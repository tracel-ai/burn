/// State module.
pub mod state;
/// Module responsible to save and exposes data collected during training.
pub mod store;

// System metrics
#[cfg(feature = "sys-metrics")]
mod cpu_temp;
#[cfg(feature = "sys-metrics")]
mod cpu_use;
#[cfg(feature = "sys-metrics")]
mod cuda;
#[cfg(feature = "sys-metrics")]
mod memory_use;
#[cfg(feature = "sys-metrics")]
pub use cpu_temp::*;
#[cfg(feature = "sys-metrics")]
pub use cpu_use::*;
#[cfg(feature = "sys-metrics")]
pub use cuda::*;
#[cfg(feature = "sys-metrics")]
pub use memory_use::*;

// Training metrics
mod acc;
mod auroc;
mod base;
mod confusion_stats;
mod fbetascore;
mod hamming;
mod iteration;
mod learning_rate;
mod loss;
mod precision;
mod recall;
mod top_k_acc;

pub use acc::*;
pub use auroc::*;
pub use base::*;
pub use confusion_stats::ConfusionStatsInput;
pub use fbetascore::*;
pub use hamming::*;
pub use iteration::*;
pub use learning_rate::*;
pub use loss::*;
pub use precision::*;
pub use recall::*;
pub use top_k_acc::*;

pub(crate) mod classification;
pub(crate) mod processor;

pub use crate::metric::classification::ClassReduction;
// Expose `ItemLazy` so it can be implemented for custom types
pub use processor::ItemLazy;
