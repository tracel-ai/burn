use super::{element::NdArrayElement, NdArrayBackend};
use burn_tensor::{ops::TensorOps, Data, Shape};
use ndarray::{s, ArcArray, Array, Axis, Dim, Ix2, Ix3, IxDyn};

#[derive(Debug, Clone)]
pub struct NdArrayTensor<E, const D: usize> {
    pub array: ArcArray<E, IxDyn>,
    pub shape: Shape<D>,
}

impl<E: NdArrayElement, const D: usize> std::ops::Add for NdArrayTensor<E, D> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        NdArrayBackend::add(&self, &rhs)
    }
}

#[cfg(test)]
mod utils {
    use super::*;
    use crate::NdArrayBackend;

    impl<E, const D: usize> NdArrayTensor<E, D>
    where
        E: Default + Clone,
    {
        pub(crate) fn into_data(self) -> Data<E, D>
        where
            E: NdArrayElement,
        {
            <NdArrayBackend<E> as TensorOps<NdArrayBackend<E>>>::into_data::<D>(self)
        }
    }
}

#[derive(new)]
pub(crate) struct BatchMatrix<E, const D: usize> {
    pub arrays: Vec<ArcArray<E, Ix2>>,
    pub shape: Shape<D>,
}

impl<E, const D: usize> BatchMatrix<E, D>
where
    E: NdArrayElement,
{
    pub fn from_ndarray(array: ArcArray<E, IxDyn>, shape: Shape<D>) -> Self {
        let mut arrays = Vec::new();
        if D < 2 {
            let array = array.reshape((1, shape.dims[0]));
            arrays.push(array);
        } else {
            let batch_size = batch_size(&shape);
            let size0 = shape.dims[D - 2];
            let size1 = shape.dims[D - 1];
            let array_global = array.reshape((batch_size, size0, size1));
            for b in 0..batch_size {
                let array = array_global.slice(s!(b, .., ..));
                let array = array.into_owned().into_shared();
                arrays.push(array);
            }
        }

        Self { arrays, shape }
    }

    pub fn matmul(self, other: BatchMatrix<E, D>) -> Self {
        let self_iter = self.arrays.iter();
        let other_iter = other.arrays.iter();

        let arrays = self_iter
            .zip(other_iter)
            .map(|(lhs, rhs)| lhs.dot(rhs))
            .map(|output| output.into_shared())
            .collect();

        let mut shape = self.shape;
        shape.dims[D - 1] = other.shape.dims[D - 1];

        Self::new(arrays, shape)
    }
}

fn batch_size<const D: usize>(shape: &Shape<D>) -> usize {
    let mut num_batch = 1;
    for i in 0..D - 2 {
        num_batch *= shape.dims[i];
    }

    num_batch
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

        NdArrayTensor {
            array,
            shape: $shape,
        }
    }};
}

impl<E, const D: usize> NdArrayTensor<E, D>
where
    E: Default + Clone,
{
    pub(crate) fn from_bmatrix(bmatrix: BatchMatrix<E, D>) -> NdArrayTensor<E, D> {
        let shape = bmatrix.shape;
        let to_array = |data: BatchMatrix<E, D>| {
            let dims = data.shape.dims;
            let mut array: Array<E, Ix3> = Array::default((0, dims[D - 2], dims[D - 1]));

            for item in data.arrays {
                array.push(Axis(0), item.view()).unwrap();
            }

            array.into_shared()
        };

        match D {
            1 => to_nd_array_tensor!(1, shape, to_array(bmatrix)),
            2 => to_nd_array_tensor!(2, shape, to_array(bmatrix)),
            3 => to_nd_array_tensor!(3, shape, to_array(bmatrix)),
            4 => to_nd_array_tensor!(4, shape, to_array(bmatrix)),
            5 => to_nd_array_tensor!(5, shape, to_array(bmatrix)),
            6 => to_nd_array_tensor!(6, shape, to_array(bmatrix)),
            _ => panic!(""),
        }
    }
}
impl<E, const D: usize> NdArrayTensor<E, D>
where
    E: Default + Clone,
{
    pub fn from_data(data: Data<E, D>) -> NdArrayTensor<E, D> {
        let shape = data.shape;
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
    use burn_tensor::Distribution;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(
            Shape::new([3]),
            Distribution::Standard,
            &mut StdRng::from_entropy(),
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
            &mut StdRng::from_entropy(),
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
            &mut StdRng::from_entropy(),
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
            &mut StdRng::from_entropy(),
        );
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
