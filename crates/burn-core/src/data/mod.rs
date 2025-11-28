/// Dataloader module.
#[cfg(feature = "dataset")]
pub mod dataloader;

/// Dataset module.
#[cfg(feature = "dataset")]
pub mod dataset {
    pub use burn_dataset::*;
}

/// Network module.
#[cfg(all(feature = "network", feature = "dataset"))]
pub mod network {
    pub use burn_dataset::network::*;
}
