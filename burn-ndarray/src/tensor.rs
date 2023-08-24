use burn_tensor::{Data, Shape};

use ndarray::{ArcArray, Array, Dim, IxDyn};

#[derive(new, Debug, Clone)]
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

/// Converts a slice of usize to a typed dimension.
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

/// Reshapes an array into a tensor.
#[macro_export(local_inner_macros)]
macro_rules! reshape {
    (
        ty $ty:ty,
        n $n:expr,
        shape $shape:expr,
        array $array:expr
    ) => {{
        let dim = $crate::to_typed_dims!($n, $shape.dims, justdim);
        let safe_into_shape =
            $array.is_standard_layout() ||
            (
                $array.ndim() > 1 &&
                $array.raw_view().reversed_axes().is_standard_layout()
            );

        let array: ndarray::ArcArray<$ty, Dim<[usize; $n]>> = match safe_into_shape {
            true => $array
                .into_shape(dim)
                .expect("Safe to change shape without relayout")
                .into_shared(),
            false => $array.reshape(dim),
        };
        let array = array.into_dyn();

        NdArrayTensor::new(array)
    }};
    (
        ty $ty:ty,
        shape $shape:expr,
        array $array:expr,
        d $D:expr
    ) => {{
        match $D {
            1 => reshape!(ty $ty, n 1, shape $shape, array $array),
            2 => reshape!(ty $ty, n 2, shape $shape, array $array),
            3 => reshape!(ty $ty, n 3, shape $shape, array $array),
            4 => reshape!(ty $ty, n 4, shape $shape, array $array),
            5 => reshape!(ty $ty, n 5, shape $shape, array $array),
            6 => reshape!(ty $ty, n 6, shape $shape, array $array),
            _ => core::panic!("NdArray supports arrays up to 6 dimensions, received: {}", $D),
        }
    }};
}

impl<E, const D: usize> NdArrayTensor<E, D>
where
    E: Default + Clone,
{
    pub fn from_data(data: Data<E, D>) -> NdArrayTensor<E, D> {
        let shape = data.shape.clone();
        let to_array = |data: Data<E, D>| Array::from_iter(data.value).into_shared();
        let array = to_array(data);

        reshape!(
            ty E,
            shape shape,
            array array,
            d D
        )
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
            Distribution::Default,
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
            Distribution::Default,
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
            Distribution::Default,
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
            Distribution::Default,
            &mut get_seeded_rng(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
