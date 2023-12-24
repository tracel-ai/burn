/// Dataloader module.
#[cfg(feature = "dataset")]
pub mod dataloader;

/// Dataset module.
#[cfg(feature = "dataset")]
pub mod dataset {
    pub use burn_dataset::*;
}
