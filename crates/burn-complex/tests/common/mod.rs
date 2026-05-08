//use burn_complex::base::split::SplitBackend;
use burn_complex::kind::ComplexKind;
use burn_complex::split::SplitComplexTensor;
use burn_tensor::TensorData;
use burn_tensor::{Float, Shape, Tensor, backend::Backend};
// #[cfg(all(
//     any(feature = "test-cpu", feature = "flex"),
//     not(any(feature = "test-wgpu", feature = "test-cuda"))
// ))]

pub type TestBackend = burn_flex::Flex;
pub type TestTensor<const D: usize> = Tensor<TestBackend, D, ComplexKind>;
//pub type TestTensor<const D: usize> = SplitComplexTensor<TestBackend, D>;
pub type FloatTensor<const D: usize> = Tensor<TestBackend, D, Float>;
