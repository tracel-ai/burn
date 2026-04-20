use burn_backend::{Backend, tensor::Float};

use crate::{Tensor, TensorCreationOptions, check, check::TensorCheck};

/// Creates a 1D Blackman window tensor.
///
/// # Arguments
/// - `size`: Size of the returned 1D window tensor.
/// - `periodic`: If `true`, the window is treated as periodic (i.e., `N = size`).
///   If `false`, the window is symmetric (i.e., `N = size - 1`).
/// - `options`: Controls the output device and optional dtype. Accepts:
///     - `&device` - uses the devices' default float dtype
///     - `(&device, DType::F32)` - uses an explicit dtype
///     - `TensorCreationOptions` directly for full control.
///
/// # Returns
/// - A 1D tensor of shape `[size]` containing the window.
///
/// # Notes
/// - If `size == 0`, the function returns an empty tensor.
/// - If `size == 1`, the returned window contains a single value 1.0 which overrides the formula.
///
/// # Panics
/// Panics if `size` exceeds `i64::MAX`.
///
/// # Example
/// ```rust,ignore
/// use burn_tensor::{backend::Backend, DType, signal::blackman_window};
///
/// fn example<B: Backend>() {
///     // Creating a window with default dtype
///     let device = B::Device::default();
///     let window_tensor = blackman_window::<B>(5, true, &device);
///     // Output: [0.0, 0.20077015, 0.84922993, 0.8492298, 0.2007701]
///
///     // Creating a window with explicit dtype
///     let device = B::Device::default();
///     let window_tensor_f64 = blackman_window::<B>(5, true, (&device, DType::F64));
///     // Output: [0.0, 0.20077015, 0.84922993, 0.8492298, 0.2007701]
/// }
/// ```
pub fn blackman_window<B: Backend>(
    size: usize,
    periodic: bool,
    options: impl Into<TensorCreationOptions<B>>,
) -> Tensor<B, 1> {
    let opt = options.into();
    let dtype = opt.resolve_dtype::<Float>();
    let shape = [size];
    check!(TensorCheck::creation_ops::<1>(
        "signal::blackman_window",
        &shape
    ));

    if size == 0 {
        return Tensor::<B, 1>::empty(shape, opt).cast(dtype);
    }

    if size == 1 {
        return Tensor::<B, 1>::ones(shape, opt).cast(dtype);
    }

    let size_i64 = i64::try_from(size)
        .expect("The argument `size` should be less than or equal to `i64::MAX`.");
    let denominator = if periodic { size } else { size - 1 };
    let angular_increment = (2.0 * core::f64::consts::PI) / denominator as f64;
    let cos_val = Tensor::arange(0..size_i64, &opt.device)
        .float()
        .mul_scalar(angular_increment)
        .cos();

    // Using the double angle property of cosine: cos(2θ) = 2cos^2(θ) - 1
    // w[n] = 0.42 - 0.5cos(2πn / (N - 1)) + 0.08cos(4πn / (N - 1))
    // w[n] = 0.42 - 0.5cos(2πn / (N - 1)) + 0.08cos(2 * (2πn / (N - 1)))
    // w[n] = 0.42 - 0.5cos(2πn / (N - 1)) + 0.08(2cos^2(2πn / (N - 1)) - 1)
    // w[n] = 0.34 - 0.5cos(2πn / (N - 1)) + 0.16cos^2(2πn / (N - 1))
    let first_cos_term = cos_val.clone().mul_scalar(-0.5);
    let second_cos_term = cos_val.powi_scalar(2).mul_scalar(0.16);
    first_cos_term
        .add(second_cos_term)
        .add_scalar(0.34)
        .cast(dtype)
}
