use std::fs;
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::process;
use std::thread;
use std::time::Duration;

use std::fmt;
use std::sync::Arc;
use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use measure_time::print_time;
use nix::unistd::Pid;
use nix::sys::signal::{self, Signal};
use serde_json as sj;

use util::{Pnt, Tree};

use crate::opts::GlobalOpts;

pub const TO_PY: &str = "pysparse.sock";

pub struct ChildGuard(process::Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        eprintln!("Cleaning up child process {}", self.0.id());
        signal::kill(Pid::from_raw(self.0.id() as i32), Signal::SIGTERM).unwrap();
        thread::sleep(Duration::from_millis(250));
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Wrapper to format the tree as valid JSON
pub struct TreeFmt<'a>(pub &'a HashMap<usize, Arc<util::trees::Node>>);

impl<'a> fmt::Debug for TreeFmt<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("{")?;
        let mut first = true;
        for (k, v) in self.0.iter() {
            if *k == 0 { continue; }

            if !first {
                f.write_str(", ")?;
            }

            first = false;
            write!(
                f, "\"{}\": [{}, {}]",
                k, v.x / 250.0, v.y / 250.0
            )?;
        }

        f.write_str("}")
    }
}

pub struct Bridge {
    pub pyserver: ChildGuard,
    pub stream: UnixStream,
    pub opts: GlobalOpts,
}

impl Bridge {
    pub fn new(opts: GlobalOpts) -> Result<Self> {
        if fs::metadata("pyready.txt").is_ok() { fs::remove_file("pyready.txt").context("Could not remove pyready.txt")?; }
        let pyserver = ChildGuard(process::Command::new("./pysparse.sh").spawn().context("Could not start python server")?);
        while fs::metadata("pyready.txt").is_err() {}

        Ok(Bridge {
            pyserver,
            stream: UnixStream::connect(TO_PY).context("Could not create stream")?,
            opts,
        })
    }

    pub fn cmd(&mut self, cmd: String) -> Result<String> {
        self.stream.write_all((cmd + ";").as_bytes())?;

        let mut response = String::new();
        let mut buffer = [0u8; 64];

        loop {
            match self.stream.read(&mut buffer).context("Failed to receive data")? {
                n if n > 0 => {
                    let current_str = String::from_utf8_lossy(&buffer[..n]);
                    response.push_str(&current_str);
                    if current_str.ends_with(';') {
                        return Ok(response);
                    }
                }
                _ => bail!("Server closed connection before sending terminator"),
            }
        }
    }

    pub fn send_opts(&mut self, opts: &GlobalOpts) -> Result<()> {
        let _ = self.cmd(format!("opts|{}", sj::to_string(opts)?))?;
        Ok(())        
    }

    pub fn step(&mut self, step: usize, tree: &Tree) -> Result<(Vec<usize>, Vec<f32>)> {
        // print_time!("py.step");
        let resp = self.cmd(format!("step|{step}|0|{:?}", TreeFmt(tree)))?;
        let mut parts = resp.split(&['|', ';']);
        let x = parts.next().context("No x-coord")?;
        let y = parts.next().context("No y-coord")?;
        Ok((sj::from_str(x)?, sj::from_str(y)?))
    }

    pub fn next(&mut self, step: usize, reward: f32, term: bool) -> Result<()> {
        // print_time!("py.next");
        let _ = self.cmd(format!("next|{step}|[{reward}]|[{}]", term as i32))?;
        Ok(())
    }

    pub fn step_eval(&mut self, step: usize, tree: &Tree) -> Result<(Vec<usize>, Vec<f32>)> {
        let resp = self.cmd(format!("step_eval|{step}|0|{:?}", TreeFmt(tree)))?;
        let mut parts = resp.split(&['|', ';']);
        let x = parts.next().context("No x-coord")?;
        let y = parts.next().context("No y-coord")?;
        Ok((sj::from_str(x)?, sj::from_str(y)?))
    }

    pub fn next_eval(&mut self, step: usize, reward: f32, term: bool) -> Result<()> {
        let _ = self.cmd(format!("next_eval|{step}|[{reward}]|[{}]", term as i32))?;
        Ok(())
    }

    pub fn train(&mut self, step: usize) -> Result<f32> {
        let _ = self.cmd(format!("train|{step}"))?;
        Ok(0.0)
    }

    pub fn save(&mut self) -> Result<()> {
        let _ = self.cmd(format!("save"))?;
        Ok(())
    }

    pub fn load(&mut self, name: &str, checkpoint: usize) -> Result<()> {
        let _ = self.cmd(format!("load|{name}|{checkpoint}"))?;
        Ok(())
    }

    pub fn mode(&mut self, training: bool) -> Result<()> {
        let _ = self.cmd(format!("mode|{training}"))?;
        Ok(())
    }

    pub fn get_dir(&mut self) -> Result<String> {
        let mut ret = self.cmd(format!("dir"))?;
        ret.pop();
        Ok(ret)
    }

    pub fn t(&mut self) -> Result<()> {
        let _ = self.cmd(format!("tmode"))?;
        Ok(())
    }

    pub fn e(&mut self) -> Result<()> {
        let _ = self.cmd(format!("emode"))?;
        Ok(())
    }

    pub fn plot(&mut self, r: f32, c: f32, er: f32, ec: f32) -> Result<()> {
        let _ = self.cmd(format!("plot|{r}|{c}|{er}|{ec}"))?;
        Ok(())
    }

    pub fn reward_scale(&mut self, reward: f32, start: usize, end: usize) -> Result<()> {
        let _ = self.cmd(format!("rs|{reward}|{start}|{end}"))?;
        Ok(())
    }
}