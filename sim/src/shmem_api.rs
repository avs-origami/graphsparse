use std::fs::{self, File};
use std::os::unix::io::{AsRawFd, FromRawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::thread;

use anyhow::Result;
use num_derive::FromPrimitive;
use num_traits::{AsPrimitive, FromPrimitive};
use passfd::FdPassingExt;
use shmem_ipc::sharedring::{Sender, Receiver};

pub use crate::{BSOCK, FSOCK, CAPACITY};
pub use crate::api::{Cmd, Req, SimApi};

#[derive(Clone, Copy, FromPrimitive)]
pub enum CmdType {
    Step,
    Rot,
    StepBy,
    StepBack,
    RotBy,
    DrawTree,
    DrawTree2,
    AddScore,
    StatsMem,
    StatsTime,
    StatsDist,
    Stats,
    ImgOut,
    Replace,
    Done,
    Reset,
    ScaleReq,
    PosReq,
    ObsReq,
    FreeReq,
    FrontierReq,
    ViewDistReq,
    CoverageReq,
    FreeGridSizeReq,
    FacingReq,
    PixbufReq,
    Ret,
    Undef,
}

impl CmdType {
    pub fn float(&self) -> f32 {
        return *self as usize as f32;
    }

    pub fn usize(&self) -> usize {
        return *self as usize;
    }
}

impl From<f32> for CmdType {
    fn from(value: f32) -> Self {
        <CmdType as FromPrimitive>::from_f32(value).unwrap()
    }
}

pub struct SimInner {
    pub sender: Sender<f32>,
    pub recv: Receiver<f32>,
}

pub struct Sim {
    pub inner: Arc<Mutex<SimInner>>,
}

impl Clone for Sim {
    fn clone(&self) -> Self {
        Sim { inner: self.inner.clone() }
    }
}

impl Sim {
    /// Constructor.
    pub fn new() -> Result<Sim> {
        let fstream = UnixStream::connect(FSOCK)?;

        let mfd = fstream.recv_fd()?;
        let m = unsafe { File::from_raw_fd(mfd) };

        let efd = fstream.recv_fd()?;
        let e = unsafe { File::from_raw_fd(efd) };

        let ffd = fstream.recv_fd()?;
        let f = unsafe { File::from_raw_fd(ffd) };

        let sender = Sender::open(CAPACITY, m, e, f)?;

        if fs::metadata(BSOCK).is_ok() { fs::remove_file(BSOCK)?; }
        let sock = UnixListener::bind(BSOCK)?;
        let (bstream, _) = sock.accept()?;

        let recv = Receiver::new(CAPACITY)?;
        let m = recv.memfd().as_file().try_clone()?;
        let e = recv.empty_signal().try_clone()?;
        let f = recv.full_signal().try_clone()?;

        bstream.send_fd(m.as_raw_fd())?;
        bstream.send_fd(e.as_raw_fd())?;
        bstream.send_fd(f.as_raw_fd())?;

        Ok(Sim {
            inner: Arc::new(Mutex::new(SimInner {
                sender,
                recv,
            }))
        })
    }
}

impl SimApi for Sim {
    fn cmd(&self, s: Cmd) -> Result<()> {
        let cmd = match s {
            Cmd::Step => vec![CmdType::Step.float(), 0.0],
            Cmd::Rot(x) => vec![CmdType::Rot.float(), 1.0, x],
            Cmd::StepBy(x) => vec![CmdType::StepBy.float(), 1.0, x],
            Cmd::StepBack(x) => vec![CmdType::StepBack.float(), 1.0, x],
            Cmd::RotBy(x) => vec![CmdType::RotBy.float(), 1.0, x],
            Cmd::AddScore => vec![CmdType::AddScore.float(), 0.0],
            Cmd::StatsTime(x) => vec![CmdType::StatsTime.float(), 1.0, x],
            Cmd::StatsDist(x) => vec![CmdType::StatsDist.float(), 1.0, x],
            Cmd::Stats(x) => vec![CmdType::Stats.float(), 1.0, x],
            Cmd::Reset => vec![CmdType::Reset.float(), 0.0],
            Cmd::Done => vec![CmdType::Done.float(), 0.0],
            Cmd::Ping => vec![CmdType::Undef.float(), 0.0],
            // Cmd::DrawTree(x) => {
            //     let mut cmd = vec![CmdType::DrawTree.float(), x.len() as f32];
            //     for i in x { cmd.push(util::coords(i.pnt32().0, i.pnt32().1) as f32); }
            //     cmd
            // },
            // Cmd::DrawTree2(x) => {
            //     let mut cmd = vec![CmdType::DrawTree2.float(), x.len() as f32];
            //     for i in x { cmd.push(util::coords(i.pnt32().0, i.pnt32().1) as f32); }
            //     cmd
            // },
        };

        let mut inn = self.inner.lock().unwrap();

        unsafe {
            inn.sender.send_trusted(|d: &mut [f32]| {
                for (n, i) in cmd.iter().enumerate() {
                    d[n] = *i;
                }

                return d.len()
            })?;
        }

        drop(inn);

        #[allow(deprecated)]
        thread::sleep_ms(3);

        Ok(())
    }

    fn req<T: FromStr + Copy + 'static>(&self, r: Req) -> Result<Vec<T>> where
        <T as FromStr>::Err: std::fmt::Debug,
        f32: AsPrimitive<T>
    {       
        let req = match r {
            Req::Scale => vec![CmdType::ScaleReq.float(), 0.0],
            Req::Pos => vec![CmdType::PosReq.float(), 0.0],
            Req::Obs => vec![CmdType::ObsReq.float(), 0.0],
            Req::Free => vec![CmdType::FreeReq.float(), 0.0],
            Req::Frontier => vec![CmdType::FrontierReq.float(), 0.0],
            Req::ViewDist => vec![CmdType::ViewDistReq.float(), 0.0],
            Req::Coverage => vec![CmdType::CoverageReq.float(), 0.0],
            Req::GridSize => vec![CmdType::FreeGridSizeReq.float(), 0.0],
            Req::Facing => vec![CmdType::FacingReq.float(), 0.0],
            Req::Pixbuf => vec![CmdType::PixbufReq.float(), 0.0],
            Req::PixbufFull => vec![CmdType::PixbufReq.float(), 1.0],
        };

        let mut inn = self.inner.lock().unwrap();

        unsafe {
            inn.sender.send_trusted(|d: &mut [f32]| {
                for (n, i) in req.iter().enumerate() {
                    d[n] = *i;
                }

                return d.len()
            })?;
        }

        #[allow(deprecated)]
        thread::sleep_ms(3);

        inn.recv.block_until_readable()?;

        let mut resp = vec![];
        unsafe {
            inn.recv.receive_trusted(|p: &[f32]| {
                for i in p {
                    resp.push(*i);
                }

                return p.len();
            }).unwrap();
        }

        drop(inn);

        Ok(resp[1..=resp[0] as usize].iter().map(|x| (*x).as_()).collect())
    }
}
