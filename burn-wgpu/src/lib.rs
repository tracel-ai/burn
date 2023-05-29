#[macro_use]
extern crate derive_new;

pub(crate) mod context;
pub(crate) mod element;
pub(crate) mod kernel;
pub(crate) mod pool;
pub(crate) mod tensor;

mod device;
pub use device::*;

mod backend;
pub use backend::*;

mod graphics;
pub use graphics::*;
