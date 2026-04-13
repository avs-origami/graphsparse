use std::collections::VecDeque;
use std::fs::{self, File};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use passfd::FdPassingExt;
use shmem_ipc::sharedring::{Sender, Receiver};

pub use sim::{BSOCK, FSOCK, CAPACITY};

pub struct SimState {
    queue: Arc<Mutex<VecDeque<Vec<f32>>>>,
    listener: UnixListener,
    sender: Option<Sender<f32>>,
}

impl SimState {
    pub fn new() -> Result<Self> {
        if fs::metadata(FSOCK).is_ok() { fs::remove_file(FSOCK)?; }

        Ok(SimState {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            listener: UnixListener::bind(FSOCK)?,
            sender: None,
        })
    }

    pub fn add_receiver(&mut self) -> Result<(File, File, File)> {
        let mut r = Receiver::new(CAPACITY)?;
        let m = r.memfd().as_file().try_clone()?;
        let e = r.empty_signal().try_clone()?;
        let f = r.full_signal().try_clone()?;

        let queue = self.queue.clone();
        thread::spawn(move || loop {
            r.block_until_readable().unwrap();
            
            unsafe {
                r.receive_trusted(|p: &[f32]| {
                    let mut q = vec![];
                    for i in p {
                        q.push(*i);
                    }

                    queue.lock().unwrap().push_back(q);
                    return p.len();
                }).unwrap();
            }
        });

        Ok((m, e, f))
    }

    pub fn send_fds(&mut self) -> Result<()> {
        let (m, e, f) = self.add_receiver()?;
        let (stream, _) = self.listener.accept()?;

        stream.send_fd(m.as_raw_fd())?;
        stream.send_fd(e.as_raw_fd())?;
        stream.send_fd(f.as_raw_fd())?;

        Ok(())
    }

    pub fn accept_fds(&mut self) -> Result<()> {
        let stream = UnixStream::connect(BSOCK)?;

        let mfd = stream.recv_fd()?;
        let m = unsafe { File::from_raw_fd(mfd) };

        let efd = stream.recv_fd()?;
        let e = unsafe { File::from_raw_fd(efd) };

        let ffd = stream.recv_fd()?;
        let f = unsafe { File::from_raw_fd(ffd) };

        self.sender = Some(Sender::open(CAPACITY, m, e, f)?);
        Ok(())
    }

    pub fn next(&mut self) -> Option<Vec<f32>> {
        self.queue.lock().unwrap().pop_front()
    }

    pub fn write(&mut self, data: &[f32]) -> Result<()> {
        unsafe {
            if let Some(s) = &mut self.sender {
                s.send_trusted(|d: &mut [f32]| {
                    d[0] = data.len() as f32;
                    for (n, i) in data.iter().enumerate() {
                        d[n + 1] = *i;
                    }

                    return d.len()
                })?;
            }

            Ok(())
        }
    }
}
