use burn_core::backend::FlexDevice;
//use burn_complex::base::split::SplitBackend;
use burn_tensor::{ComplexKind, Float, Tensor};

pub type TestDevice = FlexDevice;

pub type TestTensor<const D: usize> = Tensor<D, ComplexKind>;

pub type FloatTensor<const D: usize> = Tensor<D, Float>;
