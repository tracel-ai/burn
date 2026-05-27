use burn_backend::{ComplexTensorBackend, InterleavedLayout};

use crate::{Dispatch, DispatchDevice};

impl ComplexTensorBackend for Dispatch {
    type InnerBackend = Self;

    type Layout = InterleavedLayout;

    fn complex_from_real_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| {
            B::complex_from_real_data(data, device)
        })
    }

    fn complex_from_imag_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| {
            B::complex_from_imag_data(data, device)
        })
    }

    fn complex_from_interleaved_data(
        data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| B::complex_from_interleaved_data(
            data, device
        ))
    }

    fn complex_from_parts_data(
        real_data: burn_backend::TensorData,
        imag_data: burn_backend::TensorData,
        device: &DispatchDevice,
    ) -> burn_backend::ComplexTensor<Self> {
        complex_creation_op!(Complex, device, |device| {
            B::complex_from_parts_data(real_data, imag_data, device)
        })
    }
}
