use crate::kernel::index::slice;
use crate::ops::numeric::{empty_device_dtype, zeros};
use crate::tensor::CubeTensor;
use burn_backend::{DType, TensorMetadata};
use burn_std::Slice;
use cubecl::prelude::*;
use cubek::fft::{irfft_launch, rfft_launch};

// Materializes a padded tensor (allocate + copy) because rfft_launch/irfft_launch
// in the external cubek crate don't support virtual padding via a length parameter.
// See: https://github.com/tracel-ai/cubek/issues/194
fn pad_to_length(tensor: CubeTensor, dim: usize, target: usize) -> CubeTensor {
    let shape = tensor.shape();
    let current = shape[dim];
    if current == target {
        return tensor;
    }
    if current > target {
        let ranges: Vec<_> = shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == dim { 0..target } else { 0..s })
            .collect();
        return slice(tensor, &ranges);
    }
    let mut padded_shape = shape.clone();
    padded_shape[dim] = target;
    let padded = zeros(tensor.device.clone(), padded_shape, tensor.dtype);
    let slices: Vec<Slice> = shape.iter().map(|&s| Slice::from(0..s)).collect();
    crate::kernel::index::slice_assign(padded, &slices, tensor)
}

/// Launch the rfft kernel with optional padding for non-power-of-two sizes.
///
/// Signal is first truncated or zero-padded to `n` (when provided), then internally
/// padded to the next power of two so the kernel operates on a pow2 length.
/// Output bin count is `fft_size / 2 + 1` where `fft_size = next_pow2(n)`.
pub fn rfft(signal: CubeTensor, dim: usize, n: Option<usize>) -> (CubeTensor, CubeTensor) {
    let dtype = match signal.dtype {
        DType::F64 => f64::elem_type_native(),
        DType::F32 => f32::elem_type_native(),
        _ => panic!("Unsupported type {:?}", signal.dtype),
    };

    let input_device = signal.device.clone();
    let input_dtype = signal.dtype;
    let input_shape = signal.shape();
    let requested_n = n.unwrap_or(input_shape[dim]);
    let fft_size = requested_n.next_power_of_two();

    // Truncate/pad to requested_n, THEN pad to fft_size; otherwise for
    // requested_n < input_len < fft_size we would keep bogus samples in [n, fft_size).
    let signal = pad_to_length(signal, dim, requested_n);
    let signal = pad_to_length(signal, dim, fft_size);

    let signal_shape = signal.shape();
    let mut output_shape = signal_shape.clone();
    output_shape[dim] = fft_size / 2 + 1;

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
    .unwrap_or_else(|e| {
        panic!(
            "rfft kernel launch failed (device={input_device:?}, dtype={input_dtype:?}, \
             dim={dim}, requested_n={requested_n}, fft_size={fft_size}): {e}"
        )
    });

    (output_re, output_im)
}

/// Launch the irfft kernel with optional padding for non-power-of-two sizes.
pub fn irfft(
    spectrum_re: CubeTensor,
    spectrum_im: CubeTensor,
    dim: usize,
    n: Option<usize>,
) -> CubeTensor {
    assert!(
        spectrum_re.shape() == spectrum_im.shape(),
        "irfft: spectrum_re and spectrum_im shapes must match"
    );
    assert!(
        spectrum_re.shape()[dim] >= 1,
        "irfft: spectrum dimension cannot be empty"
    );
    assert!(
        !matches!(n, Some(0)),
        "irfft: n must be >= 1 when specified, got Some(0)"
    );

    let dtype = match spectrum_re.dtype {
        DType::F64 => f64::elem_type_native(),
        DType::F32 => f32::elem_type_native(),
        _ => panic!("Unsupported type {:?}", spectrum_re.dtype),
    };

    let input_device = spectrum_re.device.clone();
    let input_dtype = spectrum_re.dtype;
    let requested_n = n.unwrap_or((spectrum_re.shape()[dim] - 1) * 2);
    let fft_size = requested_n.next_power_of_two().max(1);
    let half_fft = fft_size / 2 + 1;

    let spectrum_re = pad_to_length(spectrum_re, dim, half_fft);
    let spectrum_im = pad_to_length(spectrum_im, dim, half_fft);

    let mut signal_shape = spectrum_re.shape().clone();
    signal_shape[dim] = fft_size;

    let signal = empty_device_dtype(
        spectrum_re.client.clone(),
        spectrum_re.device.clone(),
        signal_shape,
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
    .unwrap_or_else(|e| {
        panic!(
            "irfft kernel launch failed (device={input_device:?}, dtype={input_dtype:?}, \
             dim={dim}, requested_n={requested_n}, fft_size={fft_size}): {e}"
        )
    });

    if fft_size > requested_n {
        pad_to_length(signal, dim, requested_n)
    } else {
        signal
    }
}
