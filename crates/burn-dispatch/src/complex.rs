use burn_backend::{CBT, ComplexTensorBackend, InterleavedLayout, TypedDevice, element::Complex};

use crate::{Dispatch, DispatchDevice};

impl ComplexTensorBackend for Dispatch {
    type InnerBackend = Self;

    type Layout = InterleavedLayout;

    fn complex_from_real_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        creation_op!(Complex, device, |device| {
            burn_backend::complex_utils::interleaved_data_from_real_data(data)
        })
    }

    fn complex_from_imag_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        creation_op!(Complex, device, |device| {
            burn_backend::complex_utils::interleaved_data_from_imag_data(data)
        })
    }

    fn complex_from_interleaved_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        //creation_op!(Complex, device, |device| B::complex_from_data(data, device))
        todo!()
    }

    fn complex_from_parts_data(
        real_data: burn_backend::TensorData,
        imag_data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        creation_op!(Complex, device, |device| {
            burn_backend::complex_utils::interleaved_data_from_parts_data(real_data, imag_data)
        })
    }
}
