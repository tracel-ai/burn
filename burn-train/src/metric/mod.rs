/// Dashboard module for training progress.
pub mod dashboard;

/// State module for dashboard metrics.
pub mod state;

mod acc;
mod base;
#[cfg(feature = "metrics")]
mod cpu_temp;
#[cfg(feature = "metrics")]
mod cpu_use;
#[cfg(feature = "metrics")]
mod cuda;
#[cfg(feature = "metrics")]
mod gpu_temp;
mod learning_rate;
mod loss;
#[cfg(feature = "metrics")]
mod memory_use;

pub use acc::*;
pub use base::*;
#[cfg(feature = "metrics")]
pub use cpu_temp::*;
#[cfg(feature = "metrics")]
pub use cpu_use::*;
#[cfg(feature = "metrics")]
pub use cuda::*;
#[cfg(feature = "metrics")]
pub use gpu_temp::*;
pub use learning_rate::*;
pub use loss::*;
#[cfg(feature = "metrics")]
pub use memory_use::*;
