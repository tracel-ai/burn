use crate::ops::numeric::empty_device_dtype;
use crate::{CubeRuntime, tensor::CubeTensor};
use burn_backend::{DType, TensorMetadata};
use cubecl::prelude::*;
use cubek::fft::{irfft_launch, rfft_launch};

/// launch the fft kernel
pub fn rfft<R: CubeRuntime>(signal: CubeTensor<R>, dim: usize) -> (CubeTensor<R>, CubeTensor<R>) {
    let dtype = match signal.dtype {
        DType::F64 => f64::as_type_native_unchecked().storage_type(),
        DType::F32 => f32::as_type_native_unchecked().storage_type(),
        _ => panic!("Unsupported type {:?}", signal.dtype),
    };

    let signal_shape = signal.shape();
    let mut output_shape = signal_shape.clone();
    output_shape[dim] = output_shape[dim] / 2 + 1;

    let output_re = empty_device_dtype(
        signal.client.clone(),
        signal.device.clone(),
        output_shape.clone(),
        signal.dtype,
    );
    let output_im = empty_device_dtype(
        signal.client.clone(),
        signal.device.clone(),
        output_shape.clone(),
        signal.dtype,
    );

    rfft_launch(
        &signal.client.clone(),
        signal.binding(),
        output_re.clone().binding(),
        output_im.clone().binding(),
        dim,
        dtype,
    )
    .expect("rfft kernel launch failed");

    (output_re, output_im)
}

/// launch the irfft kernel
pub fn irfft<R: CubeRuntime>(
    spectrum_re: CubeTensor<R>,
    spectrum_im: CubeTensor<R>,
    dim: usize,
) -> CubeTensor<R> {
    let dtype = f32::as_type_native_unchecked().storage_type();

    let spectrum_shape = spectrum_re.shape();
    let mut signal_shape = spectrum_shape.clone();
    signal_shape[dim] = (signal_shape[dim] - 1) * 2;

    let signal = empty_device_dtype(
        spectrum_re.client.clone(),
        spectrum_re.device.clone(),
        signal_shape.clone(),
        spectrum_re.dtype,
    );

    irfft_launch(
        &spectrum_re.client.clone(),
        spectrum_re.binding(),
        spectrum_im.binding(),
        signal.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    signal
}
