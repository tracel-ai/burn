/// Dashboard module for training progress.
pub mod dashboard;

/// State module for dashboard metrics.
pub mod state;

mod acc;
mod base;
mod cpu_temp;
mod cpu_use;
mod cuda;
mod gpu_temp;
mod learning_rate;
mod loss;
mod memory_use;

pub use acc::*;
pub use base::*;
pub use cpu_temp::*;
pub use cpu_use::*;
pub use cuda::*;
pub use gpu_temp::*;
pub use learning_rate::*;
pub use loss::*;
pub use memory_use::*;
