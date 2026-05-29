use burn_core::backend::FlexDevice;
//use burn_complex::base::split::SplitBackend;
use burn_tensor::{Complex, Float, Tensor};

pub type TestDevice = FlexDevice;

pub type TestTensor<const D: usize> = Tensor<D, Complex>;

pub type FloatTensor<const D: usize> = Tensor<D, Float>;
