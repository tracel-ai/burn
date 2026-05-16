pub(crate) mod check;

// #[cfg(feature = "autodiff")]
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

// #[cfg(not(feature = "autodiff"))]
// mod autodiff {
//     impl<const D: usize, K: Autodiff<InnerKind = K>> Tensor<D, K> {
//         /// Returns the inner tensor without the autodiff information.
//         pub fn inner(self) -> Tensor<D, K> {
//             compile_error!("Missing `autodiff` feature")
//         }

//         /// Convert a tensor to the autodiff backend.
//         ///
//         /// # Arguments
//         ///
//         /// * `inner` - The tensor to convert.
//         ///
//         /// # Returns
//         ///
//         /// The tensor converted to the autodiff backend.
//         pub fn from_inner(inner: Tensor<D, K>) -> Self {
//             compile_error!("Missing `autodiff` feature")
//         }
//     }
// }

// #[cfg(feature = "autodiff")]
pub use autodiff::*;
pub use base::*;
pub use cartesian_grid::cartesian_grid;
pub use cast::*;
pub use float::{DEFAULT_ATOL, DEFAULT_RTOL};
pub use options::*;
pub use transaction::*;
