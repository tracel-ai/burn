/// Dashboard module for training progress.
pub mod dashboard;

/// State module for dashboard metrics.
pub mod state;

mod acc;
mod base;
#[cfg(feature = "ui")]
mod cpu_temp;
#[cfg(feature = "ui")]
mod cpu_use;
#[cfg(feature = "ui")]
mod cuda;
#[cfg(feature = "ui")]
mod gpu_temp;
mod learning_rate;
mod loss;
#[cfg(feature = "ui")]
mod memory_use;

pub use acc::*;
pub use base::*;
#[cfg(feature = "ui")]
pub use cpu_temp::*;
#[cfg(feature = "ui")]
pub use cpu_use::*;
#[cfg(feature = "ui")]
pub use cuda::*;
#[cfg(feature = "ui")]
pub use gpu_temp::*;
pub use learning_rate::*;
pub use loss::*;
#[cfg(feature = "ui")]
pub use memory_use::*;
