use crate::{element::TchElement, TchBackend, TchDevice, TchKind, TchShape, TchTensor};
use burn_tensor::{backend::Backend, ops::TensorOps, Data, Distribution, ElementConversion, Shape};
use std::ops::{Add, Div, Mul, Range, Sub};

impl<E: TchElement> TensorOps<TchBackend<E>> for TchBackend<E> {
    fn from_data<const D: usize>(data: Data<E, D>, device: TchDevice) -> TchTensor<E, D> {
        TchTensor::from_data(data, device.into())
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        device: TchDevice,
    ) -> TchTensor<bool, D> {
        TchTensor::from_data(data, device.into())
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<E>,
        device: TchDevice,
    ) -> TchTensor<E, D> {
        match distribution {
            Distribution::Standard => {
                let mut tensor = TchTensor::<E, D>::empty(shape, device);
                tensor.tensor = tensor.tensor.normal_(0.0, 1.0);
                tensor
            }
            Distribution::Bernoulli(prob) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, device);
                tensor.tensor = tensor.tensor.f_bernoulli_float_(prob).unwrap();
                tensor
            }
            Distribution::Uniform(from, to) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, device);
                tensor.tensor = tensor
                    .tensor
                    .uniform_(from.to_f64().unwrap(), to.to_f64().unwrap());
                tensor
            }
            Distribution::Normal(mean, std) => {
                let mut tensor = TchTensor::<E, D>::empty(shape, device);
                tensor.tensor = tensor.tensor.normal_(mean, std);
                tensor
            }
        }
    }

    fn zeros<const D: usize>(shape: Shape<D>, device: TchDevice) -> TchTensor<E, D> {
        let mut tensor = TchTensor::<E, D>::empty(shape, device);
        tensor.tensor = tensor.tensor.zero_();
        tensor
    }

    fn ones<const D: usize>(shape: Shape<D>, device: TchDevice) -> TchTensor<E, D> {
        let mut tensor = TchTensor::<E, D>::empty(shape, device);
        tensor.tensor = tensor.tensor.ones_like();
        tensor
    }

    fn shape<const D: usize>(tensor: &<TchBackend<E> as Backend>::TensorPrimitive<D>) -> &Shape<D> {
        &tensor.shape
    }

    fn to_data<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<TchBackend<E> as Backend>::Elem, D> {
        let values: Vec<E> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape)
    }

    fn into_data<const D: usize>(
        tensor: <TchBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<TchBackend<E> as Backend>::Elem, D> {
        let values: Vec<E> = tensor.tensor.into();
        Data::new(values, tensor.shape)
    }

    fn bool_shape<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> &Shape<D> {
        &tensor.shape
    }

    fn bool_to_data<const D: usize>(
        tensor: &<TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values: Vec<bool> = tensor.tensor.shallow_clone().into();
        Data::new(values, tensor.shape)
    }

    fn bool_into_data<const D: usize>(
        tensor: <TchBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values: Vec<bool> = tensor.tensor.into();
        Data::new(values, tensor.shape)
    }

    fn bool_to_device<const D: usize>(
        tensor: &TchTensor<bool, D>,
        device: TchDevice,
    ) -> TchTensor<bool, D> {
        TchTensor {
            kind: tensor.kind,
            tensor: tensor.tensor.to(device.into()),
            shape: tensor.shape,
        }
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: &TchTensor<bool, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<bool, D2> {
        let shape_tch: TchShape<D2> = shape.into();
        let tensor = tensor.tensor.reshape(&shape_tch.dims);
        let shape = Shape::from(tensor.size());

        TchTensor {
            tensor,
            shape,
            kind: TchKind::new(),
        }
    }

    fn device<const D: usize>(tensor: &TchTensor<E, D>) -> TchDevice {
        tensor.tensor.device().into()
    }

    fn to_device<const D: usize>(tensor: &TchTensor<E, D>, device: TchDevice) -> TchTensor<E, D> {
        TchTensor {
            kind: tensor.kind,
            tensor: tensor.tensor.to(device.into()),
            shape: tensor.shape,
        }
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: <TchBackend<E> as Backend>::Device,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D> {
        let kind = TchKind::<E>::new();
        let tensor =
            tch::Tensor::empty(&shape.dims.map(|a| a as i64), (kind.kind(), device.into()));

        to_tensor(tensor)
    }

    fn add<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).add(&rhs.tensor);

        to_tensor(tensor)
    }

    fn add_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).add(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn sub<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).sub(&rhs.tensor);
        to_tensor(tensor)
    }

    fn sub_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).sub(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn mul<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).mul(&rhs.tensor);
        to_tensor(tensor)
    }

    fn mul_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).mul(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn div<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = (&lhs.tensor).div(&rhs.tensor);
        to_tensor(tensor)
    }

    fn div_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<E, D> {
        let other: f64 = (rhs.clone()).to_elem();
        let tensor = (&lhs.tensor).div(other).to_kind(lhs.kind.kind());

        to_tensor(tensor)
    }

    fn matmul<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<E, D> {
        let tensor = lhs.tensor.matmul(&rhs.tensor);
        to_tensor(tensor)
    }

    fn neg<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        Self::mul_scalar(tensor, &(-1f32).to_elem::<E>())
    }

    fn swap_dims<const D: usize>(
        tensor: &TchTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> TchTensor<E, D> {
        let tensor = tensor.tensor.transpose(dim1 as i64, dim2 as i64);
        to_tensor(tensor)
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: &TchTensor<E, D1>,
        shape: Shape<D2>,
    ) -> TchTensor<E, D2> {
        let shape_tch: TchShape<D2> = shape.into();
        let tensor = tensor.tensor.reshape(&shape_tch.dims);

        to_tensor(tensor)
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: &TchTensor<bool, D1>,
        indexes: [Range<usize>; D2],
    ) -> TchTensor<bool, D1> {
        index(tensor, indexes)
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: &TchTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> TchTensor<E, D1> {
        index(tensor, indexes)
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: &TchTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: &TchTensor<E, D1>,
    ) -> <TchBackend<E> as Backend>::TensorPrimitive<D1> {
        let shape = tensor.shape;
        let kind = tensor.kind;
        let tensor_original = tensor.tensor.copy();
        let tch_shape = TchShape::from(tensor.shape);

        let mut tensor = tensor_original.view_(&tch_shape.dims);

        for (i, index) in indexes.into_iter().enumerate().take(D2) {
            let start = index.start as i64;
            let length = (index.end - index.start) as i64;

            tensor = tensor.narrow(i as i64, start, length);
        }

        tensor.copy_(&value.tensor);

        TchTensor {
            kind,
            tensor: tensor_original,
            shape,
        }
    }

    fn mask_fill<const D: usize>(
        tensor: &TchTensor<E, D>,
        mask: &TchTensor<bool, D>,
        value: E,
    ) -> TchTensor<E, D> {
        let value: f64 = value.to_elem();
        let tensor = tensor.tensor.f_masked_fill(&mask.tensor, value).unwrap();

        to_tensor(tensor)
    }

    fn equal<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<bool, D> {
        let tensor = lhs.tensor.eq_tensor(&rhs.tensor);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn equal_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<bool, D> {
        let other: f64 = (*rhs).to_elem();
        let tensor = lhs.tensor.eq(other);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<bool, D> {
        let tensor = lhs.tensor.greater_tensor(&rhs.tensor);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<bool, D> {
        let other: f64 = (*rhs).to_elem();
        let tensor = lhs.tensor.greater(other);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater_equal<const D: usize>(
        lhs: &TchTensor<E, D>,
        rhs: &TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        let tensor = lhs.tensor.greater_equal_tensor(&rhs.tensor);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn greater_equal_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<bool, D> {
        let other: f64 = (*rhs).to_elem();
        let tensor = lhs.tensor.greater_equal(other);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower<const D: usize>(lhs: &TchTensor<E, D>, rhs: &TchTensor<E, D>) -> TchTensor<bool, D> {
        let tensor = lhs.tensor.less_tensor(&rhs.tensor);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<bool, D> {
        let other: f64 = (*rhs).to_elem();
        let tensor = lhs.tensor.less(other);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower_equal<const D: usize>(
        lhs: &TchTensor<E, D>,
        rhs: &TchTensor<E, D>,
    ) -> TchTensor<bool, D> {
        let tensor = lhs.tensor.less_equal_tensor(&rhs.tensor);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn lower_equal_scalar<const D: usize>(lhs: &TchTensor<E, D>, rhs: &E) -> TchTensor<bool, D> {
        let other: f64 = (*rhs).to_elem();
        let tensor = lhs.tensor.less_equal(other);

        TchTensor {
            shape: lhs.shape,
            tensor,
            kind: TchKind::<bool>::new(),
        }
    }

    fn detach<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        tensor.clone()
    }

    fn mean<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.mean(tensor.kind.kind());
        to_tensor(tensor)
    }

    fn sum<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, 1> {
        let tensor = tensor.tensor.sum(tensor.kind.kind());
        to_tensor(tensor)
    }

    fn mean_dim<const D: usize>(tensor: &TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        let tensor =
            tensor
                .tensor
                .mean_dim(Some([dim as i64].as_slice()), true, tensor.kind.kind());
        to_tensor(tensor)
    }

    fn sum_dim<const D: usize>(tensor: &TchTensor<E, D>, dim: usize) -> TchTensor<E, D> {
        let tensor =
            tensor
                .tensor
                .sum_dim_intlist(Some([dim as i64].as_slice()), true, tensor.kind.kind());
        to_tensor(tensor)
    }

    fn to_full_precision<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<f32, D> {
        let tensor = tensor.tensor.to_kind(TchKind::<f32>::new().kind());
        to_tensor(tensor)
    }

    fn from_full_precision<const D: usize>(tensor: &TchTensor<f32, D>) -> TchTensor<E, D> {
        let tensor = tensor.tensor.to_kind(TchKind::<E>::new().kind());
        to_tensor(tensor)
    }

    fn argmax<const D: usize>(tensor: &TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        let tensor = tensor.tensor.argmax(dim as i64, true);
        to_tensor(tensor)
    }

    fn argmin<const D: usize>(tensor: &TchTensor<E, D>, dim: usize) -> TchTensor<i64, D> {
        let tensor = tensor.tensor.argmin(dim as i64, true);
        to_tensor(tensor)
    }

    fn exp<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        to_tensor(tensor.tensor.exp())
    }

    fn log<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        to_tensor(tensor.tensor.log())
    }

    fn powf<const D: usize>(tensor: &TchTensor<E, D>, value: f32) -> TchTensor<E, D> {
        to_tensor(tensor.tensor.pow_tensor_scalar(value as f64))
    }

    fn sqrt<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        to_tensor(tensor.tensor.sqrt())
    }

    fn erf<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        to_tensor(tensor.tensor.erf())
    }

    fn cat<const D: usize>(tensors: &[TchTensor<E, D>], dim: usize) -> TchTensor<E, D> {
        let tensors: Vec<tch::Tensor> = tensors.iter().map(|t| t.tensor.shallow_clone()).collect();
        let tensor = tch::Tensor::cat(&tensors, dim as i64);
        to_tensor(tensor)
    }

    fn relu<const D: usize>(tensor: &TchTensor<E, D>) -> TchTensor<E, D> {
        to_tensor(tensor.tensor.relu())
    }
}

fn to_tensor<const D: usize, E: TchElement>(tensor: tch::Tensor) -> TchTensor<E, D> {
    let shape = Shape::from(tensor.size());

    TchTensor {
        tensor,
        shape,
        kind: TchKind::new(),
    }
}

fn index<const D1: usize, const D2: usize, E: tch::kind::Element + Copy>(
    tensor: &TchTensor<E, D1>,
    indexes: [Range<usize>; D2],
) -> TchTensor<E, D1> {
    let shape = tensor.shape.index(indexes.clone());
    let kind = tensor.kind;

    let mut tensor = tensor.tensor.shallow_clone();

    for (i, index) in indexes.iter().enumerate().take(D2) {
        let start = index.start as i64;
        let length = (index.end - index.start) as i64;
        tensor = tensor.narrow(i as i64, start, length);
    }

    TchTensor {
        kind,
        tensor,
        shape,
    }
}
