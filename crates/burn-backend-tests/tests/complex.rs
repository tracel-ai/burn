use burn_tensor::{Float, Tensor};
pub type FloatTensor<const D: usize> = Tensor<D, Float>;
//it's used in the modules
#[allow(unused)]
type TestTensor<const D: usize> = burn_tensor::Tensor<D, burn_tensor::Complex>;

#[cfg(all(feature = "flex", not(any(feature = "fusion", feature = "remote"))))]
#[path = "complex/basic.rs"]
mod basic;
#[cfg(all(feature = "flex", not(any(feature = "fusion", feature = "remote"))))]
#[path = "complex/numeric.rs"]
mod numeric;
#[cfg(all(feature = "flex", not(any(feature = "fusion", feature = "remote"))))]
#[path = "complex/ops.rs"]
mod ops;
