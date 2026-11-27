//! The RMS norm, through the autotune of cubek's RMS-norm routines.

mod tune;

pub use tune::{fastest_rms_norm_routine, launch_into_tuned, launch_tuned};
