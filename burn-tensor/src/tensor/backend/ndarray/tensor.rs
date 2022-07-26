use crate::Data;
use crate::Shape;
use crate::TensorBase;
use ndarray::s;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::Ix3;
use ndarray::{ArcArray, IxDyn};

#[derive(Debug, Clone)]
pub struct NdArrayTensor<P, const D: usize> {
    pub array: ArcArray<P, IxDyn>,
    pub shape: Shape<D>,
}

impl<P, const D: usize> TensorBase<P, D> for NdArrayTensor<P, D>
where
    P: Default + Clone,
    Dim<[usize; D]>: Dimension,
{
    fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    fn into_data(self) -> Data<P, D> {
        let values = self.array.into_iter().collect();
        Data::new(values, self.shape)
    }

    fn to_data(&self) -> Data<P, D> {
        let values = self.array.clone().into_iter().collect();
        Data::new(values, self.shape)
    }
}

#[derive(new)]
pub struct BatchMatrix<P, const D: usize> {
    pub arrays: Vec<ArcArray<P, Ix2>>,
    pub shape: Shape<D>,
}

impl<P, const D: usize> BatchMatrix<P, D>
where
    P: Clone,
    Dim<[usize; D]>: Dimension,
{
    pub fn from_ndarray(array: ArcArray<P, IxDyn>, shape: Shape<D>) -> Self {
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
}

fn batch_size<const D: usize>(shape: &Shape<D>) -> usize {
    let mut num_batch = 1;
    for i in 0..D - 2 {
        num_batch *= shape.dims[i];
    }

    num_batch
}

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

macro_rules! to_nd_array_tensor {
    (
        $n:expr,
        $shape:expr,
        $array:expr
    ) => {{
        let dim = to_typed_dims!($n, $shape.dims, justdim);
        let array: ArcArray<P, Dim<[usize; $n]>> = $array.reshape(dim);
        let array = array.into_dyn();

        NdArrayTensor {
            array,
            shape: $shape,
        }
    }};
}

impl<P, const D: usize> NdArrayTensor<P, D>
where
    P: Default + Clone,
    Dim<[usize; D]>: Dimension,
{
    pub fn from_bmatrix(bmatrix: BatchMatrix<P, D>) -> NdArrayTensor<P, D> {
        let shape = bmatrix.shape;
        let to_array = |data: BatchMatrix<P, D>| {
            let dims = data.shape.dims;
            let mut array: Array<P, Ix3> = Array::default((0, dims[D - 2], dims[D - 1]));

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
impl<P, const D: usize> NdArrayTensor<P, D>
where
    P: Default + Clone,
{
    pub fn from_data(data: Data<P, D>) -> NdArrayTensor<P, D> {
        let shape = data.shape.clone();
        let to_array = |data: Data<P, D>| Array::from_iter(data.value.into_iter()).into_shared();

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

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(Shape::new([3]), crate::Distribution::Standard);
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected =
            Data::<f32, 2>::random(Shape::new([2, 3]), crate::Distribution::Standard);
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected =
            Data::<f32, 3>::random(Shape::new([2, 3, 4]), crate::Distribution::Standard);
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected =
            Data::<f32, 4>::random(Shape::new([2, 3, 4, 2]), crate::Distribution::Standard);
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
