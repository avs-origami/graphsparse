use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::str::FromStr;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

use anyhow::{bail, Result};
use chrono::prelude::*;
use minifb::{Key, Scale};
use num_traits::AsPrimitive;
use rayon::prelude::*;

use util::*;

use crate::api::CmdReq::{self, *};
use crate::args::Args;
use crate::{colors::*, rob, stats};
use crate::demo_win::Win;

pub use crate::api::{Cmd, Req, SimApi};

const GRID_SIZE: u32 = 30;

#[derive(Clone)]
pub struct Sim {
    queue: Arc<Mutex<VecDeque<CmdReq>>>,
    headless: bool,
    running: Arc<Mutex<bool>>,
    start: Pnt,
    pub args: Args,
}

impl Sim {
    pub fn new(args: Args, start: Pnt, headless: bool) -> Result<Self> {
        Ok(Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            headless,
            running: Arc::new(Mutex::new(false)),
            start,
            args,
        })
    }

    pub fn is_running(self: &Arc<Self>) -> bool {
        *self.running.lock().unwrap()
    }

    pub fn run(self: Arc<Self>) -> thread::JoinHandle<()> {
        *self.running.lock().unwrap() = true;
        let counter_thread = Arc::new(Mutex::new(0));
        let counter = counter_thread.clone();
        let done = Arc::new(Mutex::new(false));
        let td = done.clone();
        let src = self.running.clone();
        let sr = self.running.clone();

        thread::spawn(move || {
            loop {
                if *td.lock().unwrap() {
                    break;
                }

                if ! *src.lock().unwrap() {
                    break;
                }

                *counter_thread.lock().unwrap() += 1;
                #[allow(deprecated)]
                thread::sleep_ms(1000);
            }
        });
    
        thread::spawn(move || {
            let scale = match self.args.scale {
                Some(ref x) => match &x[..] {
                    "1x" => Scale::X1,
                    "2x" => Scale::X2,
                    "4x" => Scale::X4,
                    _ => Scale::X1,
                },
                None => Scale::X1,
            };

            let mut win = if self.headless {
                Win::new_headless(self.start, self.args.clutter, self.args.seed).unwrap()
            } else {
                Win::new_window(self.start, scale, self.args.clutter, self.args.seed).unwrap()
            };

            let mut kmask = vec![0; WIDTH * HEIGHT];
            let mut wmask = vec![0; WIDTH * HEIGHT];

            let mut score = 0;
            let mut hits = 0;
            // let mut cmd_count = 0;
            let mut hit = false;
            let mut same_move = false;

            let mem = 0.0;
            let mut time = 0.0;
            let mut total_dist = 0.0;
            let mut stats_str = String::new();

            let num_free = win.buf.iter().filter(|&n| *n == BLACK).count() as f32;
            let num_obs = win.buf.iter().filter(|&n| *n == DPURPLE).count() as f32;

            // Determine the clutteredness of the environment
            let mut num_obs_regions = 0;
            for y in 0..10 {
                for x in 0..10 {
                    for i in &win.env.obs {
                        if i.coords.0 >= (x * (WIDTH / 10)) as u32
                            && i.coords.0 < ((x + 1) * (WIDTH / 10)) as u32
                            && i.coords.1 >= (y * (WIDTH / 10)) as u32
                            && i.coords.1 < ((y + 1) * (WIDTH / 10)) as u32
                        {
                            num_obs_regions += 1;
                        }
                    }
                }
            }

            let avg_cluttering = num_obs / (WIDTH * HEIGHT) as f32 * num_obs_regions as f32;
            // dbg!(avg_cluttering);

            /******************************************************/
            /****************   BEGIN SIMULATION   ****************/
            /******************************************************/

            'o: loop {
                if let Some(ref mut x) = win.win {
                    if x.is_key_down(Key::F) {
                        let utc: DateTime<Utc> = Utc::now();
                        let fname = format!(
                            "scrot/{}-{:02}-{:02}_{:02}-{:02}-{:02}.png",
                            utc.year(),
                            utc.month(),
                            utc.day(),
                            utc.hour(),
                            utc.minute(),
                            utc.second()
                        );

                        let path = Path::new(&fname);
                        let file = File::create(path).unwrap();
                        let ref mut w = BufWriter::new(file);
                        let mut encoder = png::Encoder::new(w, W32, H32);

                        encoder.set_color(png::ColorType::Rgb);
                        encoder.set_depth(png::BitDepth::Eight);
                        let mut writer = encoder.write_header().expect("Writer failure");
                        let mut image = [0; WIDTH * HEIGHT * 3];
                        let mut idx = 0;

                        for i in &win.buf {
                            let mut decoded = [0; 3];
                            let mut hex_str = format!("{:X}", i);
                            let num_zero = 6 - hex_str.len();

                            for _ in 0..num_zero {
                                hex_str.insert(0, '0');
                            }

                            hex::decode_to_slice(hex_str, &mut decoded).expect("Hex decode error");
                            for j in decoded {
                                image[idx] = j;
                                idx += 1;
                            }
                        }

                        writer.write_image_data(&image).unwrap();
                    }
                }

                if let Some(cmd) = self.queue.lock().unwrap().pop_front() {
                    match cmd {
                        CmdStep => {
                            win.rob.step();

                            if !self.args.phasing {
                                if win.is_touching(win.rob.fb_coords() as i32, DPURPLE, false) {
                                    win.rob.step_by(-2.0);
                                    same_move = true;
                                    hit = true;
                                }
                            }

                            if win.rob.x > W32 as f32 {
                                win.rob.x = W32 as f32;
                            }

                            for i in &win.env.obs {
                                for j in i.to_fb() {
                                    if (win.is_touching(j as i32, LYELLOW, true) || win.is_touching(j as i32, YELLOW, true))
                                        && j < WIDTH * HEIGHT
                                    {
                                        wmask[j] = 1;
                                    }
                                }
                            }

                            for (n, i) in win.buf.iter().enumerate() {
                                if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                    wmask[n] = 1;
                                }
                            }

                            win.update(&wmask).unwrap();
                            // if !win.win.is_open() {
                            //     child_alg.kill().unwrap();
                            //     break 'o;
                            // }
                        },
                        CmdRot(x) => {
                            win.rob.rot(x);
                            same_move = false;
                        },
                        CmdStepBy(x) => {
                            for _ in 0..x.round() as usize {
                                win.rob.step();
                                if !self.args.phasing {
                                    if win.is_touching(win.rob.fb_coords() as i32, DPURPLE, false) {
                                        win.rob.step_by(-2.0);
                                        same_move = true;
                                        hit = true;
                                    }
                                }

                                for i in &win.env.obs {
                                    for j in i.to_fb() {
                                        if (win.is_touching(j as i32, LYELLOW, true) || win.is_touching(j as i32, YELLOW, true))
                                            && j < WIDTH * HEIGHT
                                        {
                                            wmask[j] = 1;
                                        }
                                    }
                                }

                                for (n, i) in win.buf.iter().enumerate() {
                                    if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                        wmask[n] = 1;
                                    }
                                }

                                win.update(&wmask).unwrap();
                                // if !win.win.is_open() {
                                //     child_alg.kill().unwrap();
                                //     break 'o;
                                // }
                            }
                        },
                        CmdStepBack(x) => {
                            for _ in 0..x.round() as usize {
                                win.rob.step_by(-1.0);
                                if !self.args.phasing {
                                    if win.is_touching(win.rob.fb_coords() as i32, DPURPLE, false) {
                                        win.rob.step_by(2.0);
                                        same_move = false;
                                        hit = true;
                                    }
                                }

                                for i in &win.env.obs {
                                    for j in i.to_fb() {
                                        if (win.is_touching(j as i32, LYELLOW, true) || win.is_touching(j as i32, YELLOW, true))
                                            && j < WIDTH * HEIGHT
                                        {
                                            wmask[j] = 1;
                                        }
                                    }
                                }

                                for (n, i) in win.buf.iter().enumerate() {
                                    if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                        wmask[n] = 1;
                                    }
                                }

                                win.update(&wmask).unwrap();
                                // if !win.win.is_open() {
                                //     child_alg.kill().unwrap();
                                //     break 'o;
                                // }
                            }
                        },
                        CmdRotBy(x) => {
                            same_move = false;
                            for _ in 0..10 {
                                win.rob.rot_by(x / 10.0);

                                for i in &win.env.obs {
                                    for j in i.to_fb() {
                                        if (win.is_touching(j as i32, LYELLOW, true) || win.is_touching(j as i32, YELLOW, true))
                                            && j < WIDTH * HEIGHT
                                        {
                                            wmask[j] = 1;
                                        }
                                    }
                                }

                                for (n, i) in win.buf.iter().enumerate() {
                                    if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                        wmask[n] = 1;
                                    }
                                }

                                win.update(&wmask).unwrap();
                            }
                        },
                        CmdDrawTree(x) => {
                            // win.replace(NODE, LYELLOW).unwrap();
                            win.erase_overlay().unwrap();
                            win.overlay(x).unwrap();
                            win.update(&wmask).unwrap();
                        },
                        // CmdDrawTree2 => {
                        //     let ans: Vec<u32> = cmd[2..cmd[1] as usize + 1]
                        //         .iter()
                        //         .map(|x| *x as u32)
                        //         .collect();

                        //     win.replace(BLUE, LYELLOW).unwrap();
                        //     win.draw(&ans, CYAN).unwrap();
                        //     win.update().unwrap();
                        // },
                        CmdAddScore => score += 1,
                        // CmdStatsMem => mem = cmd[2],
                        CmdStatsTime(x) => time = x,
                        CmdStatsDist(x) => total_dist = x,
                        CmdStats(x) => {
                            let num_covered = win
                                .buf
                                .iter()
                                .filter(|&n| {
                                    *n == LYELLOW || *n == YELLOW || *n == GREEN || *n == RED || *n == BLUE
                                })
                                .count() as f32;

                            stats_str.push_str(
                                &stats::Stats {
                                    mem,
                                    time,
                                    total_dist,
                                    moves: score,
                                    hits,
                                    clutter: avg_cluttering,
                                    coverage: num_covered / num_free * 100.0,
                                }
                                .dump(&format!("{x}")),
                            );
                        },
                        // CmdImgOut => {
                        //     let fname = String::from_utf8(
                        //         cmd[2..cmd[1] as usize + 1]
                        //             .to_vec()
                        //             .iter()
                        //             .map(|x| *x as u8)
                        //             .collect(),
                        //     ).unwrap();
                            
                        //     let path = Path::new(&fname);
                        //     let file = File::create(path).unwrap();
                        //     let ref mut w = BufWriter::new(file);
                        //     let mut encoder = png::Encoder::new(w, W32, H32);

                        //     encoder.set_color(png::ColorType::Rgb);
                        //     encoder.set_depth(png::BitDepth::Eight);
                        //     let mut writer = encoder.write_header().expect("Writer failure");
                        //     let mut image = [0; WIDTH * HEIGHT * 3];
                        //     let mut idx = 0;

                        //     for i in &win.buf {
                        //         let mut decoded = [0; 3];
                        //         let mut hex_str = format!("{:X}", i);
                        //         let num_zero = 6 - hex_str.len();

                        //         for _ in 0..num_zero {
                        //             hex_str.insert(0, '0');
                        //         }

                        //         hex::decode_to_slice(hex_str, &mut decoded).expect("Hex decode error");
                        //         for j in decoded {
                        //             image[idx] = j;
                        //             idx += 1;
                        //         }
                        //     }

                        //     writer.write_image_data(&image).unwrap();
                        // },
                        // CmdReplace => {
                        //     win.replace(cmd[2] as u32, cmd[3] as u32).unwrap();
                        //     win.update().unwrap();
                        // },
                        CmdDone => { *done.lock().unwrap() = true; break 'o; },
                        CmdReset => break 'o,
                        CmdPing => {
                            for i in &win.env.obs {
                                for j in i.to_fb() {
                                    if (win.is_touching(j as i32, LYELLOW, true) || win.is_touching(j as i32, YELLOW, true))
                                        && j < WIDTH * HEIGHT
                                    {
                                        kmask[j] = 1;
                                    }
                                }
                            }

                            for (n, i) in win.buf.iter().enumerate() {
                                if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                    kmask[n] = 1;
                                }
                            }
                        },
                        ReqScale(x) => x.send(vec![W32 as f32, H32 as f32]).unwrap(),
                        ReqPos(x) => x.send(vec![win.rob.x, win.rob.y]).unwrap(),
                        ReqObs(x) => {
                            let mut out = vec![];
                            for i in &win.env.obs {
                                for j in i.to_fb() {
                                    if (win.is_touching(j as i32, LYELLOW, true) || win.is_touching(j as i32, YELLOW, true))
                                        && j < WIDTH * HEIGHT
                                    {
                                        let p = inv_coords(j as u32);
                                        out.push(p.0 as f32);
                                        out.push(p.1 as f32);
                                        kmask[j] = 1;
                                    }
                                }
                            }

                            x.send(out).unwrap();
                        },
                        ReqFree(x) => {
                            let mut out = vec![];
                            for (n, i) in win.buf.iter().enumerate() {
                                if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                    let p = inv_coords(n as u32);
                                    out.push(p.0 as f32);
                                    out.push(p.1 as f32);
                                    kmask[n] = 1;
                                }
                            }

                            x.send(out).unwrap();
                        },
                        ReqFrontier(x) => {
                            let mut out = vec![];
                            for (n, i) in win.buf.iter().enumerate() {
                                if (*i == LYELLOW || *i == YELLOW) && win.is_touching(*i as i32, BLACK, false) {
                                    let p = inv_coords(n as u32);
                                    out.push(p.0 as f32);
                                    out.push(p.1 as f32);
                                }
                            }

                            x.send(out).unwrap();
                        },
                        ReqPixbuf(x) => {
                            let out: Vec<f32> = win.buf.iter().zip(&kmask).map(|(n, m)| {
                                (n * m) as f32
                            }).collect();

                            x.send(out).unwrap();
                        },
                        ReqPixbufFull(x) => {
                            let out: Vec<f32> = win.buf.iter().map(|n| {
                                *n as f32
                            }).collect();

                            x.send(out).unwrap();
                        },
                        ReqViewDist(x) => x.send(vec![rob::VIEW_DIST]).unwrap(),
                        ReqCoverage(x) => {
                            let num_covered = win.buf.iter().filter(|&n| {
                                    *n == LYELLOW || *n == YELLOW || *n == GREEN || *n == RED || *n == BLUE
                                })
                                .count() as f32;

                            x.send(vec![(num_covered / num_free) * 100.0]).unwrap();
                        },
                        ReqGridSize(x) => x.send(vec![GRID_SIZE as f32]).unwrap(),
                        ReqFacing(x) => x.send(vec![win.rob.facing]).unwrap(),
                        // ReqRet => cmd_count += 1,
                        // ReqUndef => (),
                    }
                }

                if hit && !same_move {
                    hits += 1;
                    hit = false;
                }

                if *counter.lock().unwrap() % 3 == 0 {
                    win.update(&wmask).unwrap();
                }

                if win.rob.pnt32().0 > W32 {
                    // eprintln!("There was a glitch!");
                    win.rob.x = WIDTH as f32;
                }

                if win.rob.pnt32().1 > H32 {
                    // eprintln!("There was a glitch!");
                    win.rob.y = HEIGHT as f32;
                }

                if *counter.lock().unwrap() % 10 == 0 {
                    for (idx, point) in win.buf.iter().enumerate() {
                        if *point == GREEN {
                            let mut around = vec![];
                            for n in 0..win.buf.len() {
                                if is_agent(n as i32, idx as i32, 1) {
                                    around.push(n);
                                }
                            }

                            let mut purple = 0;
                            for i in &around {
                                if *i == DPURPLE as usize {
                                    purple += 1;
                                }
                            }

                            if purple >= 5 {
                                // eprintln!("There was a glitch!");
                                break 'o;
                            }
                        }
                    }
                }

                // if *_ctrlc_exit.lock().unwrap() {break 'b; }

                // if !win.win.is_open() {
                //     child_alg.kill().unwrap();
                //     break 'o;
                // }
            }

            /****************************************************************/
            /****************   SAVE FINAL BUFFER TO IMAGE   ****************/
            /****************************************************************/

            let utc: DateTime<Utc> = Utc::now();
            let fname = format!(
                "img/{}-{:02}-{:02}_{:02}-{:02}-{:02}.png",
                utc.year(),
                utc.month(),
                utc.day(),
                utc.hour(),
                utc.minute(),
                utc.second()
            );

            let path = Path::new(&fname);
            let file = File::create(path).unwrap();
            let ref mut w = BufWriter::new(file);
            let mut encoder = png::Encoder::new(w, W32, H32);

            encoder.set_color(png::ColorType::Rgb);
            encoder.set_depth(png::BitDepth::Eight);
            let mut writer = encoder.write_header().expect("Writer failure");
            let mut image = [0; WIDTH * HEIGHT * 3];
            let mut idx = 0;

            for i in &win.buf {
                let mut decoded = [0; 3];
                let mut hex_str = format!("{:X}", i);
                let num_zero = 6 - hex_str.len();

                for _ in 0..num_zero {
                    hex_str.insert(0, '0');
                }

                hex::decode_to_slice(hex_str, &mut decoded).expect("Hex decode error");
                for j in decoded {
                    image[idx] = j;
                    idx += 1;
                }
            }

            writer.write_image_data(&image).unwrap();

            /*******************************************************/
            /****************   OUTPUT STATISTICS   ****************/
            /*******************************************************/

            let num_covered = win.buf.iter()
                .filter(|&n| *n == LYELLOW || *n == YELLOW || *n == GREEN || *n == RED || *n == BLUE)
                .count() as f32;

            // eprintln!("Number of moves: {score}");
            // eprintln!("Number of collisions: {hits}");
            // eprintln!("Clutter: {:.3}", avg_cluttering);
            // eprintln!(
            //     "Percentage of environment covered: {}% ({num_covered}/{num_free})",
            //     (num_covered / num_free) * 100.0
            // );

            // eprintln!("Number of commands recieved: {cmd_count}");

            stats_str.push_str(
                &stats::Stats {
                    mem,
                    time,
                    total_dist,
                    moves: score,
                    hits,
                    clutter: avg_cluttering,
                    coverage: num_covered / num_free * 100.0,
                }
                .dump("final"),
            );

            let fname = format!(
                "log/{}-{:02}-{:02}_{:02}-{:02}-{:02}.txt",
                utc.year(),
                utc.month(),
                utc.day(),
                utc.hour(),
                utc.minute(),
                utc.second()
            );

            let path = Path::new(&fname);
            let mut file = File::create(path).unwrap();
            write!(file, "{stats_str}").unwrap();

            *done.lock().unwrap() = true;
            *sr.lock().unwrap() = false;
            self.queue.lock().unwrap().clear();
        })
    }
}

impl SimApi for Sim {
    fn cmd(&self, s: Cmd) -> Result<()> {
        if ! *self.running.lock().unwrap() {
            bail!("Simulator not running!");
        }

        self.queue.lock().unwrap().push_back(CmdReq::from_cmd(s));
        Ok(())
    }

    fn req<T: std::str::FromStr + Copy + 'static + std::marker::Send>(&self, r: Req) -> Result<Vec<T>> where
        <T as FromStr>::Err: std::fmt::Debug,
        f32: AsPrimitive<T>
    {
        if ! *self.running.lock().unwrap() {
            bail!("Simulator not running!");
        }

        let (tx, rx) = mpsc::channel();
        self.queue.lock().unwrap().push_back(CmdReq::from_req(r, tx));
        let res = rx.recv()?;
        let res = res.iter().map(|x| (*x).as_()).collect();
        Ok(res)
    }
}