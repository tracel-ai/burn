use crate::tensor::BatchMatrix;
use crate::{element::NdArrayElement, tensor::NdArrayTensor, NdArrayBackend};
use crate::{to_nd_array_tensor, NdArrayDevice, SEED};
use burn_tensor::Distribution;
use burn_tensor::{backend::Backend, ops::TensorOps, Data, ElementConversion, Shape};
use ndarray::{Axis, Dim, IxDyn, SliceInfoElem};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::ops::Range;

macro_rules! keepdim {
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        mean
    ) => {{
        let tensor: NdArrayTensor<E, $D> = mean_dim(&$self, $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayBackend::reshape(&tensor, shape)
    }};
    (
        $D:expr,
        $dim:expr,
        $self:expr,
        sum
    ) => {{
        let tensor: NdArrayTensor<E, $D> = sum_dim(&$self, $dim);
        let mut shape = $self.shape();
        shape.dims[$dim] = 1;
        NdArrayBackend::reshape(&tensor, shape)
    }};
}

impl<E: NdArrayElement> TensorOps<NdArrayBackend<E>> for NdArrayBackend<E> {
    fn from_data<const D: usize>(data: Data<E, D>, _device: &NdArrayDevice) -> NdArrayTensor<E, D> {
        NdArrayTensor::from_data(data)
    }

    fn from_data_bool<const D: usize>(
        data: Data<bool, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<bool, D> {
        NdArrayTensor::from_data(data)
    }

    fn random<const D: usize>(
        shape: Shape<D>,
        distribution: Distribution<E>,
        device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        let mut seed = SEED.lock().unwrap();
        let mut rng: StdRng = match seed.as_ref() {
            Some(rng) => rng.clone(),
            None => StdRng::from_entropy(),
        };
        let tensor = Self::from_data(Data::random(shape, distribution, &mut rng), device);
        *seed = Some(rng);
        tensor
    }

    fn shape<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Shape<D> {
        tensor.shape()
    }

    fn to_data<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<NdArrayBackend<E> as Backend>::Elem, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape())
    }

    fn into_data<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> Data<<NdArrayBackend<E> as Backend>::Elem, D> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        Data::new(values, shape)
    }

    fn bool_shape<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Shape<D> {
        tensor.shape()
    }

    fn bool_to_data<const D: usize>(
        tensor: &<NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let values = tensor.array.iter().map(Clone::clone).collect();
        Data::new(values, tensor.shape())
    }

    fn bool_into_data<const D: usize>(
        tensor: <NdArrayBackend<E> as Backend>::BoolTensorPrimitive<D>,
    ) -> Data<bool, D> {
        let shape = tensor.shape();
        let values = tensor.array.into_iter().collect();
        Data::new(values, shape)
    }

    fn bool_to_device<const D: usize>(
        tensor: &NdArrayTensor<bool, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<bool, D> {
        tensor.clone()
    }

    fn bool_reshape<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<bool, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<bool, D2> {
        match D2 {
            1 => to_nd_array_tensor!(bool, 1, shape, tensor.array),
            2 => to_nd_array_tensor!(bool, 2, shape, tensor.array),
            3 => to_nd_array_tensor!(bool, 3, shape, tensor.array),
            4 => to_nd_array_tensor!(bool, 4, shape, tensor.array),
            5 => to_nd_array_tensor!(bool, 5, shape, tensor.array),
            6 => to_nd_array_tensor!(bool, 6, shape, tensor.array),
            _ => panic!("NdArrayTensor support only 6 dimensions."),
        }
    }

    fn device<const D: usize>(_tensor: &NdArrayTensor<E, D>) -> NdArrayDevice {
        NdArrayDevice::Cpu
    }

    fn to_device<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
        _device: &NdArrayDevice,
    ) -> NdArrayTensor<E, D> {
        tensor.clone()
    }

    fn empty<const D: usize>(
        shape: Shape<D>,
        device: &<NdArrayBackend<E> as Backend>::Device,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        NdArrayBackend::<E>::zeros(shape, device)
    }

    fn add<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array + &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn add_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array + *rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn sub<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array - &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn sub_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array - *rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn mul<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array * &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn mul_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array * *rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn div<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array / &rhs.array;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn div_scalar<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &E,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let array = &lhs.array / *rhs;
        let array = array.into_shared();

        NdArrayTensor { array }
    }

    fn matmul<const D: usize>(
        lhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
        rhs: &<NdArrayBackend<E> as Backend>::TensorPrimitive<D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        let batch_self = BatchMatrix::from_ndarray(lhs.array.clone(), lhs.shape());
        let batch_other = BatchMatrix::from_ndarray(rhs.array.clone(), rhs.shape());
        let output = batch_self.matmul(batch_other);

        NdArrayTensor::from_bmatrix(output)
    }

    fn neg<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
    ) -> <NdArrayBackend<E> as Backend>::TensorPrimitive<D> {
        Self::mul_scalar(tensor, &(-1f32).to_elem::<E>())
    }
    fn swap_dims<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
        dim1: usize,
        dim2: usize,
    ) -> NdArrayTensor<E, D> {
        let mut array = tensor.array.clone();
        array.swap_axes(dim1, dim2);

        NdArrayTensor { array }
    }

    fn reshape<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<E, D1>,
        shape: Shape<D2>,
    ) -> NdArrayTensor<E, D2> {
        match D2 {
            1 => to_nd_array_tensor!(1, shape, tensor.array),
            2 => to_nd_array_tensor!(2, shape, tensor.array),
            3 => to_nd_array_tensor!(3, shape, tensor.array),
            4 => to_nd_array_tensor!(4, shape, tensor.array),
            5 => to_nd_array_tensor!(5, shape, tensor.array),
            6 => to_nd_array_tensor!(6, shape, tensor.array),
            _ => panic!("NdArrayTensor support only 6 dimensions."),
        }
    }

    fn bool_index<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<bool, D1>,
        indexes: [Range<usize>; D2],
    ) -> NdArrayTensor<bool, D1> {
        let slices = to_slice_args::<D1, D2>(indexes);
        let array = tensor
            .array
            .clone()
            .slice_move(slices.as_slice())
            .into_shared();

        NdArrayTensor { array }
    }

    fn index<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
    ) -> NdArrayTensor<E, D1> {
        let slices = to_slice_args::<D1, D2>(indexes);
        let array = tensor
            .array
            .clone()
            .slice_move(slices.as_slice())
            .into_shared();

        NdArrayTensor { array }
    }

    fn index_assign<const D1: usize, const D2: usize>(
        tensor: &NdArrayTensor<E, D1>,
        indexes: [Range<usize>; D2],
        value: &NdArrayTensor<E, D1>,
    ) -> NdArrayTensor<E, D1> {
        let slices = to_slice_args::<D1, D2>(indexes);
        let mut array = tensor.array.to_owned();
        array.slice_mut(slices.as_slice()).assign(&value.array);
        let array = array.into_owned().into_shared();

        NdArrayTensor { array }
    }

    fn mask_fill<const D: usize>(
        tensor: &NdArrayTensor<E, D>,
        mask: &NdArrayTensor<bool, D>,
        value: E,
    ) -> NdArrayTensor<E, D> {
        let elem = E::default();
        let mask_mul = mask.array.mapv(|x| match x {
            true => E::zeros(&elem),
            false => E::ones(&elem),
        });
        let mask_add = mask.array.mapv(|x| match x {
            true => value,
            false => E::zeros(&elem),
        });
        let array = (tensor.array.clone() * mask_mul) + mask_add;

        NdArrayTensor { array }
    }

    fn equal<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());

        Self::equal_scalar(&tensor, &zero)
    }

    fn equal_scalar<const D: usize>(lhs: &NdArrayTensor<E, D>, rhs: &E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a == *rhs).into_shared();

        NdArrayTensor { array }
    }

    fn greater<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::greater_scalar(&tensor, &zero)
    }

    fn greater_scalar<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a > *rhs).into_shared();

        NdArrayTensor { array }
    }

    fn greater_equal<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::greater_equal_scalar(&tensor, &zero)
    }

    fn greater_equal_scalar<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a >= *rhs).into_shared();

        NdArrayTensor { array }
    }

    fn lower<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::lower_scalar(&tensor, &zero)
    }

    fn lower_scalar<const D: usize>(lhs: &NdArrayTensor<E, D>, rhs: &E) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a < *rhs).into_shared();

        NdArrayTensor { array }
    }

    fn lower_equal<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &NdArrayTensor<E, D>,
    ) -> NdArrayTensor<bool, D> {
        let tensor = NdArrayBackend::<E>::sub(lhs, rhs);
        let zero = E::zeros(&E::default());
        Self::lower_equal_scalar(&tensor, &zero)
    }

    fn lower_equal_scalar<const D: usize>(
        lhs: &NdArrayTensor<E, D>,
        rhs: &E,
    ) -> NdArrayTensor<bool, D> {
        let array = lhs.array.mapv(|a| a <= *rhs).into_shared();

        NdArrayTensor { array }
    }

    fn detach<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        tensor.clone()
    }

    fn mean<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        let data = Data::from([tensor.array.mean().unwrap()]);
        NdArrayTensor::from_data(data)
    }

    fn sum<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, 1> {
        let data = Data::from([tensor.array.sum()]);
        NdArrayTensor::from_data(data)
    }

    fn mean_dim<const D: usize>(tensor: &NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<E, D> {
        match D {
            1 => keepdim!(0, dim, tensor, mean),
            2 => keepdim!(1, dim, tensor, mean),
            3 => keepdim!(2, dim, tensor, mean),
            4 => keepdim!(3, dim, tensor, mean),
            5 => keepdim!(4, dim, tensor, mean),
            6 => keepdim!(5, dim, tensor, mean),
            _ => panic!("Dim not supported {D}"),
        }
    }

    fn sum_dim<const D: usize>(tensor: &NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<E, D> {
        match D {
            1 => keepdim!(0, dim, tensor, sum),
            2 => keepdim!(1, dim, tensor, sum),
            3 => keepdim!(2, dim, tensor, sum),
            4 => keepdim!(3, dim, tensor, sum),
            5 => keepdim!(4, dim, tensor, sum),
            6 => keepdim!(5, dim, tensor, sum),
            _ => panic!("Dim not supported {D}"),
        }
    }

    fn to_full_precision<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<f32, D> {
        let array = tensor.array.mapv(|a| a.to_elem()).into_shared();

        NdArrayTensor { array }
    }

    fn from_full_precision<const D: usize>(tensor: &NdArrayTensor<f32, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv(|a| a.to_elem()).into_shared();

        NdArrayTensor { array }
    }

    fn argmax<const D: usize>(tensor: &NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<i64, D> {
        arg(tensor, dim, cmp_min)
    }

    fn argmin<const D: usize>(tensor: &NdArrayTensor<E, D>, dim: usize) -> NdArrayTensor<i64, D> {
        arg(tensor, dim, cmp_max)
    }

    fn exp<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv(|a| a.exp_elem()).into_shared();

        NdArrayTensor { array }
    }

    fn log<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv(|a| a.log_elem()).into_shared();

        NdArrayTensor { array }
    }

    fn powf<const D: usize>(tensor: &NdArrayTensor<E, D>, value: f32) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv(|a| a.pow_elem(value)).into_shared();

        NdArrayTensor { array }
    }

    fn sqrt<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor.array.mapv(|a| a.sqrt_elem()).into_shared();

        NdArrayTensor { array }
    }

    fn cos<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv(|a| a.to_f64().unwrap().cos().to_elem())
            .into_shared();

        NdArrayTensor { array }
    }

    fn sin<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv(|a| a.to_f64().unwrap().sin().to_elem())
            .into_shared();

        NdArrayTensor { array }
    }

    fn tanh<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv(|a| a.to_f64().unwrap().tanh().to_elem())
            .into_shared();

        NdArrayTensor { array }
    }

    fn erf<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let array = tensor
            .array
            .mapv(|a| libm::erf(a.to_f64().unwrap()).to_elem())
            .into_shared();

        NdArrayTensor { array }
    }

    fn cat<const D: usize>(tensors: &[NdArrayTensor<E, D>], dim: usize) -> NdArrayTensor<E, D> {
        let arrays: Vec<ndarray::ArrayView<E, IxDyn>> =
            tensors.iter().map(|t| t.array.view()).collect();
        let array = ndarray::concatenate(Axis(dim), &arrays)
            .unwrap()
            .into_shared();

        NdArrayTensor { array }
    }

    fn relu<const D: usize>(tensor: &NdArrayTensor<E, D>) -> NdArrayTensor<E, D> {
        let zero = 0.to_elem();
        let mask = Self::lower_equal_scalar(tensor, &zero);

        Self::mask_fill(tensor, &mask, zero)
    }
}

fn to_slice_args<const D1: usize, const D2: usize>(
    indexes: [Range<usize>; D2],
) -> [SliceInfoElem; D1] {
    let mut slices = [SliceInfoElem::NewAxis; D1];
    for i in 0..D1 {
        if i >= D2 {
            slices[i] = SliceInfoElem::Slice {
                start: 0,
                end: None,
                step: 1,
            }
        } else {
            slices[i] = SliceInfoElem::Slice {
                start: indexes[i].start as isize,
                end: Some(indexes[i].end as isize),
                step: 1,
            }
        }
    }
    slices
}

fn mean_dim<E: NdArrayElement, const D1: usize, const D2: usize>(
    tensor: &NdArrayTensor<E, D1>,
    dim: usize,
) -> NdArrayTensor<E, D2> {
    let array = tensor.array.mean_axis(Axis(dim)).unwrap().into_shared();

    NdArrayTensor { array }
}

fn sum_dim<E: NdArrayElement, const D1: usize, const D2: usize>(
    tensor: &NdArrayTensor<E, D1>,
    dim: usize,
) -> NdArrayTensor<E, D2> {
    let array = tensor.array.sum_axis(Axis(dim)).into_shared();

    NdArrayTensor { array }
}

fn arg<E: NdArrayElement, F, const D: usize>(
    tensor: &NdArrayTensor<E, D>,
    dim: usize,
    cmp: F,
) -> NdArrayTensor<i64, D>
where
    F: Fn(&f64, &f64) -> Ordering,
{
    let batch_size = tensor.shape().dims[dim];

    let mut data = NdArrayBackend::to_data::<D>(tensor);
    let mut start = 0;
    let mut end = tensor.shape().dims[dim];
    let mut output = Vec::new();

    while end <= data.value.len() {
        let data_dim = &mut data.value[start..end];
        let mut sorted: Vec<f64> = data_dim.iter().map(|a| a.to_elem()).collect();
        sorted.sort_by(&cmp);

        let max = sorted[0];

        let data_dim = &mut data.value[start..end];
        let mut index: i64 = 0;
        for elem in data_dim {
            let as_float: f64 = elem.to_elem();
            if as_float == max {
                break;
            }
            index += 1;
        }
        output.push(index);
        start += batch_size;
        end += batch_size;
    }
    let mut shape = tensor.shape();
    shape.dims[dim] = 1;
    NdArrayTensor::from_data(Data::new(output, shape))
}

fn cmp_max(a: &f64, b: &f64) -> Ordering {
    if a < b {
        return Ordering::Less;
    } else if a > b {
        return Ordering::Greater;
    }
    Ordering::Equal
}

fn cmp_min(a: &f64, b: &f64) -> Ordering {
    if a > b {
        return Ordering::Less;
    } else if a < b {
        return Ordering::Greater;
    }
    Ordering::Equal
}
