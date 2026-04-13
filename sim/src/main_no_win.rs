#![allow(deprecated)]

use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use chrono::prelude::*;
use clap::Parser;

pub mod args;
pub mod colors;
pub mod env;
pub mod win;
pub mod rob;
pub mod shmem_sim;
pub mod stats;

use args::Args;
use colors::*;
use shmem_sim::SimState;
use sim::shmem_api::CmdType::*;
use util::*;
use win::Win;

const GRID_SIZE: u32 = 30;

fn main() -> Result<()> {
    let args = Args::parse();

    let counter_thread = Arc::new(Mutex::new(0));
    let counter = counter_thread.clone();
    thread::spawn(move || {
        loop {
            *counter_thread.lock().unwrap() += 1;
            thread::sleep_ms(1000);
        }
    });

    let ctrlc_exit_handler = Arc::new(Mutex::new(false));
    let _ctrlc_exit = ctrlc_exit_handler.clone();
    // Handle Ctrl-C gracefully
    ctrlc::set_handler(move || {
        // Exit the main loop
        *ctrlc_exit_handler.lock().unwrap() = true;
    })?;

    let mut com = SimState::new()?;

    let child_alg = if let Some(x) = args.child_alg {
        match args.child_alg_args {
            Some(y) => Some(Command::new(x).args(y).spawn()?),
            None => Some(Command::new(x).spawn()?),
        }
    } else {
        None
    };

    thread::sleep_ms(10);
    com.send_fds()?;
    thread::sleep_ms(10);
    com.accept_fds()?;

    let mut done = false;
    'b: loop {
        let mut sim = Win::new_headless((0.0, 0.0), args.clutter, args.seed)?;
        let mut score = 0;
        let mut hits = 0;
        let mut cmd_count = 0;
        let mut hit = false;
        let mut same_move = false;

        let mut mem = 0.0;
        let mut time = 0.0;
        let mut total_dist = 0.0;
        let mut stats_str = String::new();

        let num_free = sim.buf.iter().filter(|&n| *n == BLACK).count() as f32;
        let num_obs = sim.buf.iter().filter(|&n| *n == DPURPLE).count() as f32;

        // Determine the clutteredness of the environment
        let mut num_obs_regions = 0;
        for y in 0..10 {
            for x in 0..10 {
                for i in &sim.env.obs {
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
        dbg!(avg_cluttering);

        /******************************************************/
        /****************   BEGIN SIMULATION   ****************/
        /******************************************************/

        'o: loop {
            if let Some(cmd) = com.next() {
                match cmd[0].into() {
                    Step => {
                        sim.rob.step();

                        if !args.phasing {
                            if sim.is_touching(sim.rob.fb_coords() as i32, DPURPLE, false) {
                                sim.rob.rot_by(180.0);
                                sim.rob.step();
                                sim.rob.rot_by(180.0);
                                same_move = true;
                                hit = true;
                            }
                        }

                        if sim.rob.x > W32 as f32 {
                            sim.rob.x = W32 as f32;
                        }

                        sim.update()?;
                        // if !sim.win.is_open() {
                        //     child_alg.kill()?;
                        //     break 'o;
                        // }
                    },
                    Rot => {
                        sim.rob.rot(cmd[2]);
                        same_move = false;
                    },
                    StepBy => {
                        for _ in 0..cmd[2].round() as usize {
                            sim.rob.step();
                            if !args.phasing {
                                if sim.is_touching(sim.rob.fb_coords() as i32, DPURPLE, false) {
                                    sim.rob.rot_by(180.0);
                                    sim.rob.step();
                                    sim.rob.rot_by(180.0);
                                    same_move = true;
                                    hit = true;
                                }
                            }

                            sim.update()?;
                            // if !sim.win.is_open() {
                            //     child_alg.kill()?;
                            //     break 'o;
                            // }
                        }
                    },
                    StepBack => {
                        for _ in 0..cmd[2].round() as usize {
                            sim.rob.step_by(-1.0);
                            if !args.phasing {
                                if sim.is_touching(sim.rob.fb_coords() as i32, DPURPLE, false) {
                                    sim.rob.step();
                                    same_move = false;
                                    hit = true;
                                }
                            }

                            sim.update()?;
                            // if !sim.win.is_open() {
                            //     child_alg.kill()?;
                            //     break 'o;
                            // }
                        }
                    },
                    RotBy => {
                        same_move = false;
                        for _ in 0..10 {
                            sim.rob.rot_by(cmd[2] / 10.0);
                            sim.update()?;
                        }
                    },
                    DrawTree => {
                        let ans: Vec<u32> = cmd[2..cmd[1] as usize + 1]
                            .iter()
                            .map(|x| *x as u32)
                            .collect();

                        sim.replace(BLUE, LYELLOW)?;
                        sim.draw(&ans, BLUE)?;
                        sim.update()?;
                    },
                    DrawTree2 => {
                        let ans: Vec<u32> = cmd[2..cmd[1] as usize + 1]
                            .iter()
                            .map(|x| *x as u32)
                            .collect();

                        sim.replace(BLUE, LYELLOW)?;
                        sim.draw(&ans, CYAN)?;
                        sim.update()?;
                    },
                    AddScore => score += 1,
                    StatsMem => mem = cmd[2],
                    StatsTime => time = cmd[2],
                    StatsDist => total_dist = cmd[2],
                    Stats => {
                        let num_covered = sim
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
                            .dump(&format!("{}", cmd[2])),
                        );
                    },
                    ImgOut => {
                        let fname = String::from_utf8(
                            cmd[2..cmd[1] as usize + 1]
                                .to_vec()
                                .iter()
                                .map(|x| *x as u8)
                                .collect(),
                        )?;
                        
                        let path = Path::new(&fname);
                        let file = File::create(path)?;
                        let ref mut w = BufWriter::new(file);
                        let mut encoder = png::Encoder::new(w, W32, H32);

                        encoder.set_color(png::ColorType::Rgb);
                        encoder.set_depth(png::BitDepth::Eight);
                        let mut writer = encoder.write_header().expect("Writer failure");
                        let mut image = [0; WIDTH * HEIGHT * 3];
                        let mut idx = 0;

                        for i in &sim.buf {
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

                        writer.write_image_data(&image)?;
                    },
                    Replace => {
                        sim.replace(cmd[2] as u32, cmd[3] as u32)?;
                        sim.update()?;
                    },
                    Done => { done = true; break 'o; },
                    Reset => break 'o,
                    ScaleReq => com.write(&[W32 as f32, H32 as f32])?,
                    PosReq => com.write(&[sim.rob.x, sim.rob.y])?,
                    ObsReq => {
                        let mut out = vec![];
                        for i in &sim.env.obs {
                            for j in i.to_fb() {
                                if sim.is_touching(j as i32, LYELLOW, true)
                                    || sim.is_touching(j as i32, YELLOW, true)
                                {
                                    let p = inv_coords(j as u32);
                                    out.push(p.0 as f32);
                                    out.push(p.1 as f32);
                                }
                            }
                        }

                        com.write(&out[..])?;
                    },
                    FreeReq => {
                        let mut out = vec![];
                        for (n, i) in sim.buf.iter().enumerate() {
                            if *i == LYELLOW || *i == YELLOW || *i == GREEN || *i == RED {
                                let p = inv_coords(n as u32);
                                out.push(p.0 as f32);
                                out.push(p.1 as f32);
                            }
                        }

                        com.write(&out[..])?;
                    },
                    FrontierReq => {
                        let mut out = vec![];
                        for (n, i) in sim.buf.iter().enumerate() {
                            if (*i == LYELLOW || *i == YELLOW) && sim.is_touching(*i as i32, BLACK, false) {
                                let p = inv_coords(n as u32);
                                out.push(p.0 as f32);
                                out.push(p.1 as f32);
                            }
                        }

                        com.write(&out[..])?;
                    },
                    PixbufReq => {
                        com.write(sim.buf.iter().map(|x| *x as f32).collect::<Vec<f32>>().as_slice())?; // This isn't used
                    },
                    ViewDistReq => com.write(&[rob::VIEW_DIST])?,
                    CoverageReq => {
                        let num_covered = sim
                            .buf
                            .iter()
                            .filter(|&n| {
                                *n == LYELLOW || *n == YELLOW || *n == GREEN || *n == RED || *n == BLUE
                            })
                            .count() as f32;

                        com.write(&[(num_covered / num_free) * 100.0])?;
                    },
                    FreeGridSizeReq => com.write(&[GRID_SIZE as f32])?,
                    FacingReq => com.write(&[sim.rob.facing])?,
                    Ret => cmd_count += 1,
                    Undef => (),
                }
            }

            if hit && !same_move {
                hits += 1;
                hit = false;
            }

            if *counter.lock().unwrap() % 3 == 0 {
                sim.update()?;
            }

            if sim.rob.pnt32().0 > W32 {
                eprintln!("There was a glitch!");
                if let Some(mut x) = child_alg { x.kill()?; }
                std::process::exit(1);
            }

            if sim.rob.pnt32().1 > H32 {
                eprintln!("There was a glitch!");
                if let Some(mut x) = child_alg { x.kill()?; }
                std::process::exit(1);
            }

            if *counter.lock().unwrap() % 10 == 0 {
                for (idx, point) in sim.buf.iter().enumerate() {
                    if *point == GREEN {
                        let mut around = vec![];
                        for n in 0..sim.buf.len() {
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
                            eprintln!("There was a glitch!");
                            std::process::exit(1);
                        }
                    }
                }
            }

            // if *_ctrlc_exit.lock().unwrap() {break 'b; }

            // if !sim.win.is_open() {
            //     child_alg.kill()?;
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
        let file = File::create(path)?;
        let ref mut w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, W32, H32);

        encoder.set_color(png::ColorType::Rgb);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().expect("Writer failure");
        let mut image = [0; WIDTH * HEIGHT * 3];
        let mut idx = 0;

        for i in &sim.buf {
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

        writer.write_image_data(&image)?;

        /*******************************************************/
        /****************   OUTPUT STATISTICS   ****************/
        /*******************************************************/

        let num_covered = sim
            .buf
            .iter()
            .filter(|&n| *n == LYELLOW || *n == YELLOW || *n == GREEN || *n == RED || *n == BLUE)
            .count() as f32;

        eprintln!("Number of moves: {score}");
        eprintln!("Number of collisions: {hits}");
        eprintln!("Clutter: {:.3}", avg_cluttering);
        eprintln!(
            "Percentage of environment covered: {}% ({num_covered}/{num_free})",
            (num_covered / num_free) * 100.0
        );

        eprintln!("Number of commands recieved: {cmd_count}");

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
        let mut file = File::create(path)?;
        write!(file, "{stats_str}")?;

        if done { break 'b; }
    }

    Ok(())
}
