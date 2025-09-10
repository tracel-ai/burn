#[cfg(feature = "builtin-sources")]
mod cifar;
mod image_folder;
mod mnist;

#[cfg(feature = "builtin-sources")]
pub use cifar::*;
pub use image_folder::*;
pub use mnist::*;
