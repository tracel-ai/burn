/// Dashboard module for training progress.
pub mod dashboard;

/// State module for dashboard metrics.
pub mod state;

mod acc;
mod base;
mod cuda;
mod learning_rate;
mod loss;
mod cpu_use;
mod memory_use;
mod cpu_temp;
mod gpu_temp;

pub use acc::*;
pub use base::*;
pub use cuda::*;
pub use learning_rate::*;
pub use loss::*;
pub use cpu_use::*;
pub use memory_use::*;
pub use cpu_temp::*;
pub use gpu_temp::*;