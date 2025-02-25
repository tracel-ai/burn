pub(crate) mod cpu;
#[cfg(feature = "cubecl-backend")]
mod cube;

pub use cpu::{create_structuring_element, KernelShape};
