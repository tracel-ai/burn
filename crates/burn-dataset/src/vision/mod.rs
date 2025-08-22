#[cfg(feature = "cifar")]
mod cifar;
mod image_folder;
mod mnist;

#[cfg(feature = "cifar")]
pub use cifar::*;
pub use image_folder::*;
pub use mnist::*;
