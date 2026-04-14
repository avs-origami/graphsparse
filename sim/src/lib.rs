pub mod api;
pub mod args;
pub mod colors;
pub mod int_api;
pub mod shmem_api;

pub(crate) mod env;
pub(crate) mod win;
pub(crate) mod rob;
pub(crate) mod stats;

pub const CAPACITY: usize = 500_000;
pub const FSOCK: &str = "sim_forward.sock";
pub const BSOCK: &str = "sim_backward.sock";

pub use env::Clutter;

pub use rob::VIEW_DIST;