pub use burn_tensor::tensor::backend::autodiff::ADTensor;
use burn_tensor::tensor::backend::tch::TchTensor;
use burn_tensor::tensor::backend::TchDevice;
pub use burn_tensor::tensor::ops;
pub use burn_tensor::tensor::{Data, Element, Shape};

use burn_tensor::tensor::backend::ndarray::NdArrayTensor;
use burn_tensor::tensor::ops::*;
use burn_tensor::tensor::Tensor as TensorTrait;
use ndarray::{LinalgScalar, ScalarOperand};

pub type Tensor<E, const D: usize, B> = <B as TensorType<E, D>>::T;

pub trait Backend<E: Element>:
    TensorType<E, 1>
    + TensorType<E, 2>
    + TensorType<E, 3>
    + TensorType<E, 4>
    + TensorType<E, 5>
    + TensorType<E, 6>
{
    fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<E, D>>::T
    where
        Self: TensorType<E, D>;
}

pub trait TensorType<E: Element, const D: usize> {
    type T: TensorTrait<E, D>;

    fn from_data(data: Data<E, D>) -> Self::T;
}

#[macro_export]
macro_rules! random {
    (
        elem: $e:ident,
        shape: $shape:expr,
        distribution: $dis:expr,
        backend_ty: $b:ty
    ) => {{
        const n: usize = $shape.len();
        let shape = $crate::tensor::Shape::new($shape);
        let data = $crate::tensor::Data::<$e, n>::random(shape, $dis);

        $crate::init!($e, n, data, $b)
    }};

    (
        elem: $e:ident,
        shape: $shape:expr,
        distribution: $dis:expr,
        backend: ndarray
    ) => {{
        random!(
            elem: $e,
            shape: $shape,
            distribution: $dis,
            backend_ty: $crate::tensor::NdArrayTensorBackend<$e>
        )
    }};

    (
        elem: $e:ident,
        shape: $shape:expr,
        distribution: $dis:expr,
        backend: tch gpu $n:expr
    ) => {{
        random!(
            elem: $e,
            shape: $shape,
            distribution: $dis,
            backend_ty: $crate::tensor::TchTensorGPUBackend<$e, $n>
        )
    }};

    (
        elem: $e:ident,
        shape: $shape:expr,
        distribution: $dis:expr,
        backend: tch cpu
    ) => {{
        random!(
            elem: $e,
            shape: $shape,
            distribution: $dis,
            backend_ty: $crate::tensor::TchTensorCPUBackend<$e>
        )
    }};
}

#[macro_export]
macro_rules! init (
    ($e:ident, $n:expr, $data:expr, $b:ty) => {
        <$b as $crate::tensor::TensorType<$e, $n>>::from_data($data)
    };
    ($e:ident, $n:expr, $data:expr) => {
        <B as TensorType<$e, $n>>::from_data($data)
    };
);

pub struct Linear<E: Element, B: Backend<E>> {
    weight: Tensor<E, 2, B>,
    bias: Tensor<E, 2, B>,
}

impl<E: Element, B: Backend<E>> Linear<E, B> {
    pub fn new() -> Self {
        let weight = init!(E, 2, Data::zeros(Shape::new([2, 2])));
        let bias = init!(E, 2, Data::zeros(Shape::new([1, 2])));

        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor<E, 2, B>) -> Tensor<E, 2, B> {
        self.weight.matmul(&x).add(&self.bias)
    }
}

pub struct NdArrayTensorBackend<E> {
    _e: E,
}

impl<E: Element + ScalarOperand + LinalgScalar> Backend<E> for NdArrayTensorBackend<E> {
    fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<E, D>>::T
    where
        Self: TensorType<E, D>,
    {
        <Self as TensorType<E, D>>::from_data(data)
    }
}

impl<E: Element + ScalarOperand + LinalgScalar, const D: usize> TensorType<E, D>
    for NdArrayTensorBackend<E>
{
    type T = NdArrayTensor<E, D>;

    fn from_data(data: Data<E, D>) -> Self::T {
        let tensor = NdArrayTensor::from_data(data);
        tensor
    }
}

pub struct TchTensorGPUBackend<E, const N: usize> {
    _e: E,
}

impl<E: Element + tch::kind::Element + Into<f64>, const N: usize> Backend<E>
    for TchTensorGPUBackend<E, N>
{
    fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<E, D>>::T
    where
        Self: TensorType<E, D>,
    {
        <Self as TensorType<E, D>>::from_data(data)
    }
}

impl<E: Element + tch::kind::Element + Into<f64>, const D: usize, const N: usize> TensorType<E, D>
    for TchTensorGPUBackend<E, N>
{
    type T = TchTensor<E, D>;

    fn from_data(data: Data<E, D>) -> Self::T {
        let device = TchDevice::Cuda(N);
        let tensor = TchTensor::from_data(data, device);
        tensor
    }
}

pub struct TchTensorCPUBackend<E> {
    _e: E,
}

impl<E: Element + tch::kind::Element + Into<f64>> Backend<E> for TchTensorCPUBackend<E> {
    fn from_data<const D: usize>(data: Data<E, D>) -> <Self as TensorType<E, D>>::T
    where
        Self: TensorType<E, D>,
    {
        <Self as TensorType<E, D>>::from_data(data)
    }
}

impl<E: Element + tch::kind::Element + Into<f64>, const D: usize> TensorType<E, D>
    for TchTensorCPUBackend<E>
{
    type T = TchTensor<E, D>;

    fn from_data(data: Data<E, D>) -> Self::T {
        let device = TchDevice::Cpu;
        let tensor = TchTensor::from_data(data, device);
        tensor
    }
}
