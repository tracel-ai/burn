//! The SwiGLU gate, through the autotune of cubek's SwiGLU routine.

mod tune;

pub use tune::{launch_into_tuned, launch_tuned};
