use super::{device::GPUBackend, Device};
use crate::{
    backend::conversion::{Convertor, Order},
    Data, FloatTensor, Shape, TensorBase,
};
use arrayfire::{Array, ConstGenerator, Dim4, FloatingPoint, HasAfEnum};
use num_traits::Float;

pub struct ArrayfireTensor<P: HasAfEnum, const D: usize> {
    pub device: Device,
    pub array: Array<P>,
    pub shape: Shape<D>,
}

impl<P: HasAfEnum, const D: usize> ArrayfireTensor<P, D> {
    pub(crate) fn set_backend_binary_ops(&self, other: &Self) {
        if self.device != other.device {
            panic!("Not on same device");
        }
        set_backend(&self.device);
    }

    pub(crate) fn set_backend_single_ops(&self) {
        set_backend(&self.device);
    }
}

pub(crate) fn set_backend(device: &Device) {
    match device {
        Device::CPU => arrayfire::set_backend(arrayfire::Backend::CPU),
        &Device::GPU(device_num, backend) => {
            match backend {
                GPUBackend::CUDA => arrayfire::set_backend(arrayfire::Backend::CUDA),
                GPUBackend::OPENCL => arrayfire::set_backend(arrayfire::Backend::OPENCL),
            };
            arrayfire::set_device(device_num as i32);
        }
    }
}
impl<P: Float + HasAfEnum + Default + Copy + std::fmt::Debug, const D: usize> FloatTensor<P, D>
    for ArrayfireTensor<P, D>
where
    P: ConstGenerator<OutType = P> + FloatingPoint + Clone + Copy,
{
}

impl<P: HasAfEnum + Default + Copy + std::fmt::Debug, const D: usize> TensorBase<P, D>
    for ArrayfireTensor<P, D>
{
    fn empty(shape: Shape<D>) -> Self {
        let device = Device::CPU;
        set_backend(&device);

        let mut dims_arrayfire = [1; 4];

        for i in 0..D {
            dims_arrayfire[i] = shape.dims[i] as u64;
        }

        let array = Array::new_empty(Dim4::new(&dims_arrayfire));

        Self {
            array,
            shape,
            device,
        }
    }

    fn from<O: TensorBase<P, D>>(other: O) -> Self {
        Self::from_data(other.into_data(), Device::CPU)
    }

    fn shape(&self) -> &Shape<D> {
        &self.shape
    }
    fn into_data(self) -> Data<P, D> {
        let mut data = vec![P::default(); self.array.elements()];
        self.array.host(&mut data);
        let convertor = Convertor::new(&self.shape, Order::Right, Order::Left);
        let values = convertor.convert(&data);
        Data::new(values, self.shape)
    }
}

impl<P: HasAfEnum + Default + Copy + std::fmt::Debug, const D: usize> ArrayfireTensor<P, D> {
    pub fn from_data(data: Data<P, D>, device: Device) -> Self {
        set_backend(&device);

        let shape = data.shape.clone();
        let dims: Dim4 = data.shape.into();

        let convertor = Convertor::new(&shape, Order::Left, Order::Right);
        let values = convertor.convert(&data.value);
        let array = Array::new(&values, dims);

        Self {
            array,
            shape,
            device,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn should_not_create_tensor_with_more_than_4_dims() {
        let data_expected = Data::random(Shape::new([2, 3, 1, 4, 5]));
        let _tensor = ArrayfireTensor::<f64, 5>::from_data(data_expected.clone(), Device::CPU);
    }

    #[test]
    fn should_support_into_and_from_data_1d() {
        let data_expected = Data::random(Shape::new([3]));
        let tensor = ArrayfireTensor::<f64, 1>::from_data(data_expected.clone(), Device::CPU);

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }

    #[test]
    fn should_support_into_and_from_data_2d() {
        let data_expected = Data::random(Shape::new([2, 3]));
        let tensor = ArrayfireTensor::<f64, 2>::from_data(data_expected.clone(), Device::CPU);

        let data_actual = tensor.into_data();

        assert_eq!(data_expected, data_actual);
    }
}
