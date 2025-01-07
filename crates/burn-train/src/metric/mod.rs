/// State module.
pub mod state;
/// Module responsible to save and exposes data collected during training.
pub mod store;

mod acc;
mod auroc;
mod base;
#[cfg(feature = "metrics")]
mod confusion_stats;
#[cfg(feature = "metrics")]
mod cpu_temp;
#[cfg(feature = "metrics")]
mod cpu_use;
#[cfg(feature = "metrics")]
mod cuda;
#[cfg(feature = "metrics")]
mod fbetascore;
mod hamming;
#[cfg(feature = "metrics")]
mod iteration;
mod learning_rate;
mod loss;
#[cfg(feature = "metrics")]
mod memory_use;
#[cfg(feature = "metrics")]
mod precision;
#[cfg(feature = "metrics")]
mod recall;
#[cfg(feature = "metrics")]
mod top_k_acc;

pub use acc::*;
pub use auroc::*;
pub use base::*;
#[cfg(feature = "metrics")]
pub use confusion_stats::ConfusionStatsInput;
#[cfg(feature = "metrics")]
pub use cpu_temp::*;
#[cfg(feature = "metrics")]
pub use cpu_use::*;
#[cfg(feature = "metrics")]
pub use cuda::*;
#[cfg(feature = "metrics")]
pub use fbetascore::*;
pub use hamming::*;
#[cfg(feature = "metrics")]
pub use iteration::*;
pub use learning_rate::*;
pub use loss::*;
#[cfg(feature = "metrics")]
pub use memory_use::*;
#[cfg(feature = "metrics")]
pub use precision::*;
#[cfg(feature = "metrics")]
pub use recall::*;
#[cfg(feature = "metrics")]
pub use top_k_acc::*;

#[cfg(feature = "metrics")]
pub(crate) mod classification;
pub(crate) mod processor;

#[cfg(feature = "metrics")]
pub use crate::metric::classification::ClassReduction;
// Expose `ItemLazy` so it can be implemented for custom types
pub use processor::ItemLazy;
