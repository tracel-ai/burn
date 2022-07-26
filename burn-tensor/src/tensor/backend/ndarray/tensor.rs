use crate::Data;
use crate::Shape;
use crate::TensorBase;
use ndarray::s;
use ndarray::Array;
use ndarray::Dim;
use ndarray::Dimension;
use ndarray::Ix2;
use ndarray::{ArcArray, IxDyn};

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

macro_rules! define_from {
    (
        $n:expr
    ) => {
        impl<P> From<Data<P, $n>> for NdArrayTensor<P, $n>
        where
            P: Default + Clone,
        {
            fn from(data: Data<P, $n>) -> NdArrayTensor<P, $n> {
                let shape = data.shape.clone();
                let dim: Dim<[usize; $n]> = shape.clone().into();

                let array: ArcArray<P, Dim<[usize; $n]>> = Array::from_iter(data.value.into_iter())
                    .into_shared()
                    .reshape(dim);
                let array = array.into_dyn();

                NdArrayTensor { array, shape }
            }
        }
        impl<P> From<BatchMatrix<P, $n>> for NdArrayTensor<P, $n>
        where
            P: Default + Clone,
        {
            fn from(data: BatchMatrix<P, $n>) -> NdArrayTensor<P, $n> {
                let shape = data.shape;
                let dim: Dim<[usize; $n]> = shape.clone().into();
                let mut values = Vec::new();
                for array in data.arrays {
                    values.append(&mut array.into_iter().collect());
                }

                let array: ArcArray<P, Dim<[usize; $n]>> =
                    Array::from_iter(values).into_shared().reshape(dim);
                let array = array.into_dyn();

                NdArrayTensor { array, shape }
            }
        }
    };
}

define_from!(1);
define_from!(2);
define_from!(3);
define_from!(4);
define_from!(5);
define_from!(6);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(Shape::new([3]));
        let tensor = NdArrayTensor::from(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::<f32, 2>::random(Shape::new([2, 3]));
        let tensor = NdArrayTensor::from(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected = Data::<f32, 3>::random(Shape::new([2, 3, 4]));
        let tensor = NdArrayTensor::from(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected = Data::<f32, 4>::random(Shape::new([2, 3, 4, 2]));
        let tensor = NdArrayTensor::from(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
