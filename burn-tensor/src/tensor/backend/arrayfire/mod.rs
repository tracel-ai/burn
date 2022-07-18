pub mod device;

mod ops;
mod shape;
mod tensor;

use self::device::Device;
pub use tensor::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::backend::arrayfire::device::GPUBackend;
    use crate::{Data, Shape};
    use std::thread;

    #[test]
    fn should_support_multiple_devices_on_different_thread() {
        let function = |device| {
            for _ in 0..10 {
                let data_1 = Data::random(Shape::new([1000]));
                let data_2 = Data::random(Shape::new([1000]));
                let tensor_1 = ArrayfireTensor::<f64, 1>::from_data(data_1, device);
                let tensor_2 = ArrayfireTensor::<f64, 1>::from_data(data_2, device);
                let _data = tensor_1 + tensor_2;
            }
        };

        let handler_1 = thread::spawn(move || function(Device::CPU));
        let handler_2 = thread::spawn(move || function(Device::GPU(0, GPUBackend::OPENCL)));

        handler_1.join().unwrap();
        handler_2.join().unwrap();
    }
}
