use burn_tensor::{Data, Shape};

use ndarray::{ArcArray, Array, Dim, IxDyn};

#[derive(Debug, Clone)]
pub struct NdArrayTensor<E, const D: usize> {
    pub array: ArcArray<E, IxDyn>,
}

impl<E, const D: usize> NdArrayTensor<E, D> {
    pub(crate) fn shape(&self) -> Shape<D> {
        Shape::from(self.array.shape().to_vec())
    }
}

#[cfg(test)]
mod utils {
    use super::*;
    use crate::{element::FloatNdArrayElement, NdArrayBackend};
    use burn_tensor::ops::TensorOps;

    impl<E, const D: usize> NdArrayTensor<E, D>
    where
        E: Default + Clone,
    {
        pub(crate) fn into_data(self) -> Data<E, D>
        where
            E: FloatNdArrayElement,
        {
            <NdArrayBackend<E> as TensorOps<NdArrayBackend<E>>>::into_data::<D>(self)
        }
    }
}

#[macro_export(local_inner_macros)]
macro_rules! to_typed_dims {
    (
        $n:expr,
        $dims:expr,
        justdim
    ) => {{
        let mut dims = [0; $n];
        for i in 0..$n {
            dims[i] = $dims[i];
        }
        let dim: Dim<[usize; $n]> = Dim(dims);
        dim
    }};
}

#[macro_export(local_inner_macros)]
macro_rules! to_nd_array_tensor {
    (
        $n:expr,
        $shape:expr,
        $array:expr
    ) => {{
        let dim = $crate::to_typed_dims!($n, $shape.dims, justdim);
        let array: ndarray::ArcArray<E, Dim<[usize; $n]>> = $array.reshape(dim);
        let array = array.into_dyn();

        NdArrayTensor { array }
    }};
    (
        bool,
        $n:expr,
        $shape:expr,
        $array:expr
    ) => {{
        let dim = $crate::to_typed_dims!($n, $shape.dims, justdim);
        let array: ndarray::ArcArray<bool, Dim<[usize; $n]>> = $array.reshape(dim);
        let array = array.into_dyn();

        NdArrayTensor { array }
    }};
}

impl<E, const D: usize> NdArrayTensor<E, D>
where
    E: Default + Clone,
{
    pub fn from_data(data: Data<E, D>) -> NdArrayTensor<E, D> {
        let shape = data.shape.clone();
        let to_array = |data: Data<E, D>| Array::from_iter(data.value.into_iter()).into_shared();

        match D {
            1 => to_nd_array_tensor!(1, shape, to_array(data)),
            2 => to_nd_array_tensor!(2, shape, to_array(data)),
            3 => to_nd_array_tensor!(3, shape, to_array(data)),
            4 => to_nd_array_tensor!(4, shape, to_array(data)),
            5 => to_nd_array_tensor!(5, shape, to_array(data)),
            6 => to_nd_array_tensor!(6, shape, to_array(data)),
            _ => panic!(""),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_common::rand::get_seeded_rng;
    use burn_tensor::Distribution;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(
            Shape::new([3]),
            Distribution::Standard,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::<f32, 2>::random(
            Shape::new([2, 3]),
            Distribution::Standard,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected = Data::<f32, 3>::random(
            Shape::new([2, 3, 4]),
            Distribution::Standard,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected = Data::<f32, 4>::random(
            Shape::new([2, 3, 4, 2]),
            Distribution::Standard,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
