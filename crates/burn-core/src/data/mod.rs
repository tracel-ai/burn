/// Dataloader module.
#[cfg(feature = "dataset")]
pub mod dataloader;

/// Dataset module.
#[cfg(feature = "dataset")]
pub mod dataset {
    pub use burn_dataset::*;
}

/// Network module.
#[cfg(all(feature = "network"))]
pub mod network {
    pub use burn_std::network::*;
}
