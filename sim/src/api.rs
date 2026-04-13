use std::{str::FromStr, sync::mpsc::Sender};

use anyhow::Result;
use num_traits::AsPrimitive;
// use util::Tree;

pub trait SimApi {
    /// Send a command to the simulator.
    /// 
    /// Available commands:
    /// - `Step` → move the robot forward one step.
    /// - `Rot(n)` → rotate the robot to `n` degrees. 0 degrees is east, going CCW.
    /// - `StepBy(n)` → move the robot `n` steps forward.
    /// - `StepBack(n)` → move the robot `n` steps backward.
    /// - `RotBy(n)` → rotate the robot by `n` degrees CCW.
    /// - `DrawTree(n)` → visualize the RRT.
    /// - `Done` → ends the simulation.
    fn cmd(&self, s: Cmd) -> Result<()>;

    /// Request info from the simulator. Data is returned as a vector of f32.
    /// 
    /// Available requests:
    /// - `Pos` → requests the robot's position. x is index 0, y is index 1.
    /// - `Obs` → requests the obstacles in the FOV. x are even indices, y are odd.
    /// - `Free` → requests the free space in the FOV. x are even indices, y are odd.
    /// - `Scale` → get the dimensions of the environment.
    /// - `ViewDist` → get the view distance of the robot.
    fn req<T: FromStr + Copy + 'static + std::marker::Send>(&self, r: Req) -> Result<Vec<T>> where
        <T as FromStr>::Err: std::fmt::Debug, f32: AsPrimitive<T>;
}

#[allow(dead_code)]
pub enum Cmd {
    Step,
    Rot(f32),
    StepBy(f32),
    StepBack(f32),
    RotBy(f32),
    DrawTree(Vec<u32>),
    // DrawTree2(&'a Tree),
    AddScore,
    StatsTime(f32),
    StatsDist(f32),
    Stats(f32),
    Reset,
    Done,
    Ping,
}

#[allow(dead_code)]
pub enum Req {
    Pos,
    Obs,
    Free,
    Frontier,
    Scale,
    ViewDist,
    Coverage,
    GridSize,
    Facing,
    Pixbuf,
    PixbufFull,
}

pub enum CmdReq {
    CmdStep,
    CmdRot(f32),
    CmdStepBy(f32),
    CmdStepBack(f32),
    CmdRotBy(f32),
    CmdDrawTree(Vec<u32>),
    // CmdDrawTree2(&'a Tree),
    CmdAddScore,
    CmdStatsTime(f32),
    CmdStatsDist(f32),
    CmdStats(f32),
    CmdReset,
    CmdDone,
    CmdPing,
    ReqPos(Sender<Vec<f32>>),
    ReqObs(Sender<Vec<f32>>),
    ReqFree(Sender<Vec<f32>>),
    ReqFrontier(Sender<Vec<f32>>),
    ReqScale(Sender<Vec<f32>>),
    ReqViewDist(Sender<Vec<f32>>),
    ReqCoverage(Sender<Vec<f32>>),
    ReqGridSize(Sender<Vec<f32>>),
    ReqFacing(Sender<Vec<f32>>),
    ReqPixbuf(Sender<Vec<f32>>),
    ReqPixbufFull(Sender<Vec<f32>>),
}

impl CmdReq {
    pub fn from_cmd(c: Cmd) -> Self {
        match c {
            Cmd::Step => CmdReq::CmdStep,
            Cmd::Rot(x) => CmdReq::CmdRot(x),
            Cmd::StepBy(x) => CmdReq::CmdStepBy(x),
            Cmd::StepBack(x) => CmdReq::CmdStepBack(x),
            Cmd::RotBy(x) => CmdReq::CmdRotBy(x),
            Cmd::DrawTree(x) => CmdReq::CmdDrawTree(x),
            // Cmd::DrawTree2(x) => CmdReq::CmdDrawTree2(x),
            Cmd::AddScore => CmdReq::CmdAddScore,
            Cmd::StatsTime(x) => CmdReq::CmdStatsTime(x),
            Cmd::StatsDist(x) => CmdReq::CmdStatsDist(x),
            Cmd::Stats(x) => CmdReq::CmdStats(x),
            Cmd::Reset => CmdReq::CmdReset,
            Cmd::Done => CmdReq::CmdDone,
            Cmd::Ping => CmdReq::CmdPing,
        }
    }

    pub fn from_req(r: Req, s: Sender<Vec<f32>>) -> Self {
        match r {
            Req::Pos => CmdReq::ReqPos(s),
            Req::Obs => CmdReq::ReqObs(s),
            Req::Free => CmdReq::ReqFree(s),
            Req::Frontier => CmdReq::ReqFrontier(s),
            Req::Scale => CmdReq::ReqScale(s),
            Req::ViewDist => CmdReq::ReqViewDist(s),
            Req::Coverage => CmdReq::ReqCoverage(s),
            Req::GridSize => CmdReq::ReqGridSize(s),
            Req::Facing => CmdReq::ReqFacing(s),
            Req::Pixbuf => CmdReq::ReqPixbuf(s),
            Req::PixbufFull => CmdReq::ReqPixbufFull(s),
        }
    }
}