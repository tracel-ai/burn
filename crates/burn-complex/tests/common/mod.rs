//use burn_complex::base::split::SplitBackend;
use burn_tensor::{Float, Tensor};

pub type TestBackend = burn_flex::Flex;
//it's not dead, it's just sleeping
#[allow(dead_code)]
pub type FloatTensor<const D: usize> = Tensor<TestBackend, D, Float>;
