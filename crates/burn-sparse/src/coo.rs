use burn_tensor::backend::Backend;
use burn_tensor::ops::SparseBoolOps;
use burn_tensor::ops::SparseTensorOps;
use burn_tensor::Dense;
use burn_tensor::Device;
use burn_tensor::Float;
use burn_tensor::Int;
use burn_tensor::Shape;
use burn_tensor::Sparse;
use burn_tensor::SparseStorage;
use burn_tensor::Tensor;
use burn_tensor::TensorData;
use burn_tensor::TensorKind;

#[derive(Clone, Debug)]
pub struct COO;

#[derive(Clone, Debug)]
pub struct SparseCOOTensor<B: Backend, K: TensorKind<B>, const D: usize> {
    pub coordinates: Option<Tensor<B, 2, Int>>,
    pub values: Option<Tensor<B, 1, K>>,
    pub shape: Shape<D>,
    pub device: Device<B>,
}

impl<B: Backend> SparseStorage<B> for COO {
    type SparsePrimitive<K: burn_tensor::TensorKind<B>, const D: usize> = SparseCOOTensor<B, K, D>;

    fn name() -> &'static str {
        "SparseCOO"
    }
}

impl<B: Backend> SparseTensorOps<COO, B> for COO {}

pub(crate) fn flatten_coordinates<B: Backend, const D: usize, const S: usize>(
    coordinates: Tensor<B, 2, Int>,
    shape: Shape<D>,
    device: &Device<B>,
) -> Tensor<B, 2, Int> {
    let mut strides_data = [[1]; D];
    for i in (0..D).rev() {
        if D - 1 - i == S {
            strides_data[i] = [1];
        } else if D - 1 - i < S {
            strides_data[i] = [0];
        } else {
            strides_data[i] = [strides_data[i + 1][0] * shape.dims[i + 1] as i64];
        }
    }
    let strides_data: TensorData = TensorData::from(strides_data);
    let strides: Tensor<B, 2, Int> = Tensor::from_data(strides_data, device);
    let flat_coordinates: Tensor<B, 1, Int> = strides.mul(coordinates).sum_dim(0).flatten(0, 1);

    flat_coordinates.unsqueeze_dim(0)
}

pub(crate) fn unflatten_coordinates<B: Backend, const D: usize>(
    flat_coordinates: Tensor<B, 2, Int>,
    new_shape: Shape<D>,
) -> Tensor<B, 2, Int> {
    let flat_coordinates = flat_coordinates.squeeze::<1>(0);
    let mut remaining_flat_coordinates = flat_coordinates.clone();
    let mut new_coordinates = Vec::with_capacity(D);

    for &dim_size in new_shape.dims.iter().rev() {
        let size = dim_size as i64;
        let new_coord = remaining_flat_coordinates.clone().remainder_scalar(size);
        new_coordinates.push(new_coord.clone());
        remaining_flat_coordinates = remaining_flat_coordinates.div_scalar(size);
    }

    new_coordinates.reverse();

    Tensor::stack(new_coordinates, 0)
}
