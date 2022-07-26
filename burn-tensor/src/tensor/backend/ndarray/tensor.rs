use crate::Data;
use crate::Shape;
use crate::TensorBase;
use ndarray::s;
use ndarray::ArcArray;
use ndarray::Array;
use ndarray::Ix2;

pub struct NdArrayTensor<P, const D: usize> {
    pub arrays: Vec<ArcArray<P, Ix2>>,
    pub shape: Shape<D>,
}

impl<P: Default + Copy + std::fmt::Debug, const D: usize> TensorBase<P, D> for NdArrayTensor<P, D> {
    fn shape(&self) -> &Shape<D> {
        &self.shape
    }
    fn into_data(self) -> Data<P, D> {
        let mut arrays = self.arrays;

        if D == 1 {
            let array = arrays.remove(0);
            let values = array.into_iter().collect();
            return Data::new(values, self.shape);
        }

        let mut values = Vec::new();
        for array in arrays {
            let mut values_tmp = array.into_iter().collect();
            values.append(&mut values_tmp);
        }
        Data::new(values, self.shape)
    }
    fn to_data(&self) -> Data<P, D> {
        todo!()
    }
}
impl<P, const D: usize> NdArrayTensor<P, D>
where
    P: Default + Clone,
{
    pub fn from_data(data: Data<P, D>) -> Self {
        let shape = data.shape.clone();
        let mut arrays = Vec::new();

        if D < 2 {
            let array = Array::from_iter(data.value.into_iter())
                .into_shared()
                .reshape((1, shape.dims[0]));
            arrays.push(array);
        } else {
            let batch_size = batch_size(&shape);
            let size0 = shape.dims[D - 2];
            let size1 = shape.dims[D - 1];
            let array_global = Array::from_iter(data.value.into_iter())
                .into_shared()
                .reshape((batch_size, size0, size1));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::<f32, 1>::random(Shape::new([3]));
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::<f32, 2>::random(Shape::new([2, 3]));
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_3d() {
        let data_expected = Data::<f32, 3>::random(Shape::new([2, 3, 4]));
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_4d() {
        let data_expected = Data::<f32, 4>::random(Shape::new([2, 3, 4, 2]));
        let tensor = NdArrayTensor::from_data(data_expected.clone());

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
