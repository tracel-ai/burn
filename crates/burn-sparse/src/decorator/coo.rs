use burn_tensor::backend::Backend;
use burn_tensor::ops::SparseBoolOps;
use burn_tensor::ops::SparseTensorOps;
use burn_tensor::Dense;
use burn_tensor::Device;
use burn_tensor::Float;
use burn_tensor::Int;
use burn_tensor::Shape;
use burn_tensor::Sparse;
use burn_tensor::SparseRepr;
use burn_tensor::Tensor;
use burn_tensor::TensorKind;

#[derive(Clone, Debug)]
pub struct COO;

#[derive(Clone, Debug)]
pub struct SparseCOOTensor<B: Backend, K: TensorKind<B, Dense>, const D: usize> {
    pub coordinates: Option<Tensor<B, 2, Int>>,
    pub values: Option<Tensor<B, 1, K>>,
    pub shape: Shape<D>,
    pub device: Device<B>,
}

impl<B: Backend> SparseRepr<B> for COO {
    type Primitive<K: burn_tensor::TensorKind<B>, const D: usize> = SparseCOOTensor<B, K, D>;

    fn name() -> &'static str {
        "SparseCOO"
    }
}

impl<B: Backend> SparseTensorOps<COO, B> for COO {}
