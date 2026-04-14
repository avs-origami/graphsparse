use std::borrow::Cow;
use std::ffi::OsString;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::SystemTime;

use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};

use measure_time::print_time;
use rrt_fast::{Exit, RrtInc};
use sim::int_api::{Cmd::*, Req::*, SimApi};
use pysparse::py::Bridge;
use pysparse::opts::*;
use util::trees::Node;
use util::{info, Pnt, Tree, H32, HEIGHT, W32, WIDTH};
use sim::colors as col;
use plotters::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

lazy_static::lazy_static! {
    static ref HOST_OS: OsString = gethostname::gethostname();
    static ref HOST: Cow<'static, str> = HOST_OS.to_string_lossy();
}

pub const ALPHA: f32 = 5.0;
pub const BETA: f32 = 8.0;

fn randpoint() -> Pnt {
    // ((random::<f32>() * 230.0).round() + 10.0, (random::<f32>() * 230.0).round() + 10.0)
    // (10.0, 10.0)
    (132.0, 36.0)
}

fn randprune(tree: &Tree) -> (Vec<usize>, Vec<f32>) {
    let mut rng = rand::thread_rng();
    let mut ids = vec![];
    let mut probs = vec![];

    for (i, _) in tree {
        if *i == 0 { continue; } // Skip the root node
        ids.push(*i);
        probs.push(rng.gen_range(0.0..1.0));
    }

    return (ids, probs);
}

fn main() -> Result<()>  {
    let mut py = Bridge::new(GlobalOpts::parse())?;
    let (mut rithm, mut sim_thread) = RrtInc::new(Some(randpoint()), GlobalOpts::parse())?;
    let (mut rithm_eval, mut sim_thread_eval) = RrtInc::new(Some(randpoint()), GlobalOpts::parse())?;
    let mut rewards = vec![];
    let mut covs = vec![];
    let mut eval_rewards = vec![];
    let mut eval_covs = vec![];
    // let mut losses = vec![];

    export_state(&mut rithm, &mut vec![0; WIDTH * HEIGHT], 0);

    py.send_opts(&rithm.opts)?;

    if let Some(ref x) = rithm.opts.load {
        py.load(&x[0], x[1].parse()?)?;
    }

    let run_dir = py.get_dir()?;
    let run_name = run_dir.split('/').last().unwrap();

    for i in 0..rithm.opts.episodes {
        if ! rithm.opts.test {
            collect_rollout(&mut rithm, &mut py, &mut rewards, &mut covs, &mut sim_thread)?;
            eprintln!("Episode {}: Average reward: {}", i, rewards[rewards.len() - 1]);
            eprintln!("Episode {}: Average coverage: {}", i, covs[covs.len() - 1]);
            py.train(i)?;
            py.save()?;
            draw_chart(&rewards, &format!("learning_curves/{}.{run_name}.rewards", *HOST))?;
            draw_chart(&covs, &format!("learning_curves/{}.{run_name}.coverage", *HOST))?;
        }

        if i % 1 == 0 {
            py.e()?;
            collect_eval(&mut rithm_eval, &mut py, &mut eval_rewards, &mut eval_covs, &mut sim_thread_eval)?;
            eval_rewards.len();
            draw_chart(&eval_rewards, &format!("learning_curves/{}.{run_name}.rewards_eval", *HOST))?;
            draw_chart(&eval_covs, &format!("learning_curves/{}.{run_name}.coverage_eval", *HOST))?;
            eprintln!("Episode {}: Average eval reward: {}", i, eval_rewards[eval_rewards.len() - 1]);
            eprintln!("Episode {}: Average eval coverage: {}", i, eval_covs[eval_covs.len() - 1]);
            py.t()?;
        }

        if ! rithm.opts.test {
            py.plot(rewards[rewards.len() - 1], covs[covs.len() - 1], eval_rewards[eval_rewards.len() - 1], eval_covs[eval_covs.len() - 1])?;
        } else {
            py.plot_eval(eval_rewards[eval_rewards.len() - 1], eval_covs[eval_covs.len() - 1])?;
        }
    }

    // Randomize the robot's starting location to remove the bias caused by starting in the same corner all the time
    // Experiments: random pruning, cluster pruning

    py.cmd("done".into())?;
    #[allow(deprecated)]
    std::thread::sleep_ms(10000);

    rithm.sml.cmd(Done)?;
    sim_thread.join().unwrap();

    // draw_chart(&rewards, "rewards")?;
    // draw_chart(&losses, "losses")?;

    Ok(())
}

fn collect_rollout(rithm: &mut RrtInc, py: &mut Bridge, rewards: &mut Vec<f32>, covs: &mut Vec<f32>, sim_thread: &mut JoinHandle<()>) -> Result<()> {
    let pb = ProgressBar::new(rithm.opts.num_steps as u64);
    pb.set_message("Collecting experiences");
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise:.green}, {per_sec:<12.yellow}] [{bar:40.cyan/magenta}] (\x1b[33m{pos}/{len}\x1b[0m, \x1b[32mETA {eta}\x1b[0m) ({msg})",
        ).unwrap().progress_chars("##-")
    );

    if rithm.rrt.len() == 1 {
        rithm.step()?;
    }

    // let mut lens = vec![];
    let mut ep_reward = 0.0;
    let mut prev_len = rithm.rrt.len();

    let mut last_buf = vec![0; WIDTH * HEIGHT];
    // let mut env_rewards = vec![];

    let mut start = 0;

    'o: for i in 0..rithm.opts.num_steps {
        let mut gain = 0.0;
        let mut reward = 0.0;
        let mut terminal = false;
        let mut node_rewards = vec![];

        // Gets the probabilities of each node using the GMM
        let pvec = py.step(i, &rithm.rrt.tree)?;
        let mut tups = pvec.0.into_iter().zip(pvec.1).map(|(x, y)| (x, y)).collect::<Vec<_>>();
        tups.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // if i % 200 == 199 {
        //     eprintln!("{:?}", &tups.iter().map(|x| (rithm.rrt.tree[&x.0].pnt(), x.1)).collect::<Vec<_>>());
        // }

        let mut top_prune = vec![[0.0, 0.0]; rithm.opts.num_gauss];

        // Determine how many nodes to prune. Tree growth is more or less linear
        // so we aim to reduce the growth rate -- essentially, calculating the
        // number of nodes to prune based on the current slope of tree growth.
        let num_prune = rithm.opts.prune_frac * (rithm.rrt.len() - prev_len) as f32;
        for i in 0..num_prune as usize {
            let idx = tups[i].0;
            let mut nr = 0.0;
            let n = &rithm.rrt.tree[&idx];

            if rithm.frontiers.contains(n) {
                nr -= 1.0;
            } else {
                nr += 1.0;
            }

            if *n.num_child.lock().unwrap() == 1 {
                nr += 1.0;
            } else {
                nr -= 1.0;
            }

            // nr -= util::dist(centroid, rithm.rrt.tree[idx].pnt()) / 175.0;

            node_rewards.push(nr);

            if i < rithm.opts.num_gauss {
                top_prune[i] = [n.x / WIDTH as f32, n.y / HEIGHT as f32];
            }

            rithm.rrt.del(idx);
        }

        prev_len = rithm.rrt.len();

        'a: while rithm.rrt.len() <= prev_len {
            match rithm.step() {
                Ok(x) => match x.0 {
                    Exit::Ok => gain += x.1,
                    Exit::Timeout => {
                        eprintln!("Timed out!");
                        dbg!(rithm.num_visited);
                        gain += x.1;
                        terminal = true;
                        break 'a;
                    },
                    Exit::Finish => {
                        eprintln!("Completed successfully!");
                        dbg!(rithm.num_visited as f32 / rithm.rrt.len() as f32);
                        gain += x.1;
                        terminal = true;
                        break 'a;
                    }
                },
                Err(_) => break 'o,
            }
        }

        // lens.push(prev_len);

        if rithm.greg >= rithm.opts.num_moves {
            terminal = true;
        }

        // if lens.len() >= 100 {
        //     eprintln!("{lens:?}");
        //     std::process::exit(0);
        // }

        let (ac, _centroid) = export_state(rithm, &mut last_buf, 0);
        let pct = ac as f32 / (WIDTH * HEIGHT) as f32;

        reward += reward_fn(node_rewards, rithm.regen_attempts);

        if terminal {
            let reward_scale = (rithm.coverage / 100.0).exp() - 1.0;
            reward += BETA * reward_scale;
        }
        // dbg!(reward);
        // env_rewards.push(reward);
        ep_reward += reward;

        py.next(i, reward, terminal, pct, top_prune)?;

        if !rithm.sml.is_running() {
            rithm.reset(Some(randpoint()), sim_thread)?;
            rithm.step()?;
            prev_len = rithm.rrt.len();
        }

        if terminal {
            // let env_reward = env_rewards.iter().sum::<f32>();
            // ep_reward += env_reward;
            // py.reward_scale(reward_scale, start, start + env_rewards.len())?;
            // start += env_rewards.len();
            // env_rewards.clear();

            covs.push(rithm.coverage);
            rithm.reset(Some(randpoint()), sim_thread)?;
            rithm.step()?;
            export_state(rithm, &mut last_buf, 0);
            prev_len = rithm.rrt.len();
        }

        pb.inc(1);
    }

    pb.finish_with_message("Collected experiences");
    rewards.push(ep_reward / rithm.opts.num_steps as f32);

    Ok(())
}

fn collect_eval(rithm: &mut RrtInc, py: &mut Bridge, rewards: &mut Vec<f32>, covs: &mut Vec<f32>, sim_thread: &mut JoinHandle<()>) -> Result<()> {
    let pb = ProgressBar::new(rithm.opts.num_moves as u64);
    pb.set_message("Collecting experiences");
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise:.green}, {per_sec:<12.yellow}] [{bar:40.cyan/magenta}] (\x1b[33m{pos}/{len}\x1b[0m, \x1b[32mETA {eta}\x1b[0m) ({msg})",
        ).unwrap().progress_chars("##-")
    );

    rithm.reset(Some(randpoint()), sim_thread)?;
    rithm.step()?;

    // let mut lens = vec![];
    let mut ep_reward = 0.0;
    let mut prev_len = rithm.rrt.len();

    let mut last_buf = vec![0; WIDTH * HEIGHT];

    export_state(rithm, &mut last_buf, 1);

    let mut cnt = 0;

    'o: for i in 0..rithm.opts.num_moves {
        let mut gain = 0.0;
        let mut reward = 0.0;
        let mut terminal = false;
        let mut node_rewards = vec![];

        if !(rithm.opts.test && rithm.opts.no_prune) {
            // Gets the probabilities of each node using the GMM
            let pvec = if ! rithm.opts.random {
                py.step_eval(i, &rithm.rrt.tree)?
            } else {
                randprune(&rithm.rrt.tree)
            };
            
            let mut tups = pvec.0.into_iter().zip(pvec.1).map(|(x, y)| (x, y)).collect::<Vec<_>>();
            tups.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            // eprintln!("{:?}", &tups);

            // Determine how many nodes to prune. Tree growth is more or less linear
            // so we aim to reduce the growth rate -- essentially, calculating the
            // number of nodes to prune based on the current slope of tree growth.
            let num_prune = rithm.opts.prune_frac * (rithm.rrt.len() - prev_len) as f32;
            for i in 0..num_prune as usize {
                let idx = tups[i].0;
                let mut nr = 0.0;

                if rithm.frontiers.contains(&rithm.rrt.tree[&idx]) {
                    nr -= 1.0;
                } else {
                    nr += 1.0;
                }

                if *rithm.rrt.tree[&idx].num_child.lock().unwrap() == 1 {
                    nr += 1.0;
                } else {
                    nr -= 1.0;
                }

                // nr -= util::dist(centroid, rithm.rrt.tree[idx].pnt()) / 175.0;

                node_rewards.push(nr);
                rithm.rrt.del(idx);
            }
        }

        prev_len = rithm.rrt.len();

        'a: while rithm.rrt.len() <= prev_len {
            match rithm.step() {
                Ok(x) => match x.0 {
                    Exit::Ok => gain += x.1,
                    Exit::Timeout => {
                        eprintln!("Timed out!");
                        dbg!(rithm.num_visited);
                        gain += x.1;
                        terminal = true;
                        break 'a;
                    },
                    Exit::Finish => {
                        eprintln!("Completed successfully!");
                        dbg!(rithm.num_visited as f32 / rithm.rrt.len() as f32);
                        gain += x.1;
                        terminal = true;
                        break 'a;
                    }
                },
                Err(_) => break 'o,
            }
        }

        // lens.push(prev_len);

        if rithm.greg >= rithm.opts.num_moves {
            terminal = true;
        }

        // if lens.len() >= 100 {
        //     eprintln!("{lens:?}");
        //     std::process::exit(0);
        // }

        let (ac, _centroid) = export_state(rithm, &mut last_buf, 0);
        let pct = ac as f32 / (WIDTH * HEIGHT) as f32;

        reward += reward_fn(node_rewards, rithm.regen_attempts);

        if terminal {
            let reward_scale = (rithm.coverage / 100.0).exp() - 1.0;
            reward += BETA * reward_scale;
        }

        // dbg!(reward);
        ep_reward += reward;
        cnt += 1;

        py.next_eval(i, reward, terminal, pct)?;

        if !rithm.sml.is_running() {
            rithm.reset(Some(randpoint()), sim_thread)?;
            rithm.step()?;
            prev_len = rithm.rrt.len();
        }

        if terminal {
            covs.push(rithm.coverage);
            break;
        }

        if rithm.opts.test {
            pb.inc(rithm.regen_attempts as u64 + 1);
        }
    }

    pb.finish_with_message("Collected experiences");
    rewards.push(ep_reward / cnt as f32);

    Ok(())
}

fn reward_fn(node_rewards: Vec<f32>, reg_at: usize) -> f32 {
    let mut avg = node_rewards.iter().sum::<f32>() / node_rewards.len() as f32;

    if avg.is_nan() {
        avg = 0.0;
    }

    return avg - ALPHA * (reg_at as f32 / 100.0);
}

fn export_state(rithm: &mut RrtInc, last_buf: &mut Vec<u32>, count: usize) -> (usize, (f32, f32)) {
    // print_time!("export_state");
    // Do we draw the edges? Necessary for visualization, but maybe not necessary for model input?
    let _ = rithm.sml.cmd(Ping);
    let mut buf = rithm.sml.req::<u32>(Pixbuf).unwrap();
    // let mut buf2 = rithm.sml.req::<u32>(PixbufFull).unwrap();

    let alpha: Vec<bool> = buf.iter().zip(last_buf.iter()).map(|(x, y)| x - y > 0).collect();
    // let alpha = vec![false; WIDTH * HEIGHT];

    for (_, node) in &rithm.rrt.tree {
        // let i = util::coords(node.x as u32, node.y as u32) as usize;
        // if i < buf.len() { // temporary fix: whats happening?
        //     if buf[i] != col::RED {
        //         buf[i] = if rithm.frontiers.contains(node) {0xEEEEEE} else {0xDDDDDD};
        //         buf2[i] = if rithm.frontiers.contains(node) {0xEEEEEE} else {0xDDDDDD};
        //     }
        // }

        if let Some(x) = node.par.lock().unwrap().upgrade() {
            let l = util::gen_line(node.pnt32(), x.pnt32(), 2);
            let c = util::gen_circle(node.pnt32(), 2);
            for p in l {
                if p < buf.len() { if buf[p] != col::RED { buf[p] = col::EDGE; }}
            }

            for p in c {
                if p < buf.len() { if buf[p] != col::RED { buf[p] = col::NODE; }}
            }
        }
    }

    // let mut bvec = vec![];

    // for i in 0..buf.len() {
    //     if buf[i] != col::LYELLOW {
    //         continue;
    //     }
        
    //     let circle = util::gen_circle(util::inv_coords(i as u32), 3);
        
    //     if !circle.iter().any(|&x| buf[x] == col::BLACK) {
    //         bvec.push(i);
    //     }
    // }

    // for i in &bvec {
    //     buf[*i] = col::BLACK;
    // }

    let fname = format!("{count}.png");
    let path = Path::new(&fname);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, W32, H32);

    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("Writer failure");
    let mut image = [0; WIDTH * HEIGHT * 3];
    let mut idx = 0;

    for (i, a) in buf.iter().zip(&alpha) {
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

        // if *a { image[idx] = 255; } else { image[idx] = 127; }
        // idx += 1;
    }

    writer.write_image_data(&image).unwrap();

    // let fname = format!("img/B{count}.png");
    // let path = Path::new(&fname);
    // let file = File::create(path).unwrap();
    // let ref mut w = BufWriter::new(file);
    // let mut encoder = png::Encoder::new(w, W32, H32);

    // encoder.set_color(png::ColorType::Rgb);
    // encoder.set_depth(png::BitDepth::Eight);
    // let mut writer = encoder.write_header().expect("Writer failure");
    // let mut image = [0; WIDTH * HEIGHT * 3];
    // let mut idx = 0;

    // for i in &buf2 {
    //     let mut decoded = [0; 3];
    //     let mut hex_str = format!("{:X}", i);
    //     let num_zero = 6 - hex_str.len();

    //     for _ in 0..num_zero {
    //         hex_str.insert(0, '0');
    //     }

    //     hex::decode_to_slice(hex_str, &mut decoded).expect("Hex decode error");
    //     for j in decoded {
    //         image[idx] = j;
    //         idx += 1;
    //     }
    // }

    // writer.write_image_data(&image).unwrap();

    // *last_buf = buf;

    let (sx, sy, cnt) = buf.iter().enumerate()
        .filter(|(_, x)|**x != col::BLACK)
        .map(|(i, _)| util::inv_coords_f32(i as u32))
        .fold((0.0, 0.0, 0.0), |(sx, sy, cnt), (x, y)| (sx + x, sy + y, cnt + 1.0));

    return (alpha.iter().filter(|x| **x).count(), (sx / cnt, sy / cnt));
}

fn draw_chart(rewards: &Vec<f32>, name: &str) -> Result<()> {
    let fname = format!("{name}.png");
    let root_area = BitMapBackend::new(&fname, (1024, 1024)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let x_axis = (0..rewards.len()).step(1);

    let min = rewards.clone().into_iter().reduce(f32::min).unwrap();
    let max = rewards.clone().into_iter().reduce(f32::max).unwrap();

    let mut cc = ChartBuilder::on(&root_area)
        .margin(5)
        .set_left_and_bottom_label_area_size(50)
        .build_cartesian_2d(0.0..rewards.len() as f32, min..max)?;

    cc.configure_mesh()
        .x_labels(20)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    cc.draw_series(LineSeries::new(x_axis.values().map(|x| (x as f32, rewards[x])), &RED))?;
    root_area.present()?;

    Ok(())
}
