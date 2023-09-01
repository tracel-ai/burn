/// Dashboard module for training progress.
pub mod dashboard;

/// State module for dashboard metrics.
pub mod state;

mod acc;
mod base;
#[cfg(feature = "cli")]
mod cpu_temp;
#[cfg(feature = "cli")]
mod cpu_use;
#[cfg(feature = "cli")]
mod cuda;
#[cfg(feature = "cli")]
mod gpu_temp;
mod learning_rate;
mod loss;
#[cfg(feature = "cli")]
mod memory_use;

pub use acc::*;
pub use base::*;
#[cfg(feature = "cli")]
pub use cpu_temp::*;
#[cfg(feature = "cli")]
pub use cpu_use::*;
#[cfg(feature = "cli")]
pub use cuda::*;
#[cfg(feature = "cli")]
pub use gpu_temp::*;
pub use learning_rate::*;
pub use loss::*;
#[cfg(feature = "cli")]
pub use memory_use::*;
