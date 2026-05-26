pub(crate) mod check;

mod autodiff;
mod base;
mod bool;
mod cartesian_grid;
mod cast;
mod float;
mod fmod;
mod int;
mod numeric;
mod options;
mod orderable;
mod pad;
pub use pad::IntoPadding;
mod take;
mod transaction;

mod trunc;

#[cfg(feature = "autodiff")]
pub use autodiff::*;
pub use base::*;
pub use cartesian_grid::cartesian_grid;
pub use cast::*;
pub use float::{DEFAULT_ATOL, DEFAULT_RTOL};
pub use options::*;
pub use transaction::*;

#[cfg(feature = "extension")]
mod extension;
#[cfg(feature = "extension")]
pub use extension::*;

pub (crate) use float::{atan2_impl, powf_impl, powf_scalar_impl};
pub (crate) use bool::{bool_not_impl, bool_and_impl, bool_or_impl};