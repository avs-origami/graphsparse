use clap::Parser;

use crate::env::Clutter;

/// Simulation environment for testing exploration algorithms.
#[derive(Clone, Parser)]
#[command(about)]
pub struct Args {
    /// Algorithm to spawn as a child process. If this is not used
    /// then you must make sure to start this program BEFORE
    /// calling `shmem_api::Sim:new()`.
    #[arg(short = 'C', long)]
    pub child_alg: Option<String>,
    /// Arguments to pass to the child algorithm.
    #[arg(short = 'a', long)]
    pub child_alg_args: Option<Vec<String>>,
    /// Set the amount of clutter.
    #[arg(value_enum, short, long, default_value_t = Clutter::Nah)]
    pub clutter: Clutter,
    /// Set the scale (for windowed simulation only). Supports "1x",
    /// "2x", and "4x".
    #[arg(short, long)]
    pub scale: Option<String>,
    /// Set the seed for the simulation. Defaults to a random value.
    #[arg(long)]
    pub seed: Option<u64>,
    /// Enable phasing through obstacles.
    #[arg(long)]
    pub phasing: bool,
}