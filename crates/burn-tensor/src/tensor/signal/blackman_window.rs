use burn_backend::tensor::{Float, Int};

use crate::{Tensor, TensorCreationOptions, check, check::TensorCheck};

/// Creates a 1D Blackman window tensor.
///
#[cfg_attr(
    doc,
    doc = r#"
$$w_n = 0.42 - 0.5 \cos\left(\frac{2\pi n}{N}\right) + 0.08 \cos\left(\frac{4\pi n}{N}\right)$$

where $N$ = `size` when `periodic` is `true`, or $N$ = `size - 1` when `periodic` is `false`.
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`w_n = 0.42 - 0.5 * cos(2πn / N) + 0.08 * cos(4πn / N)` where N = size (periodic) or N = size-1 (symmetric)"
)]
///
/// # Arguments
/// - `size`: Size of the returned 1D window tensor.
/// - `periodic`: If `true`, the window is treated as periodic (i.e., `N = size`).
///   If `false`, the window is symmetric (i.e., `N = size - 1`).
/// - `options`: Controls the output device and optional dtype. Accepts:
///     - `&device` - uses the device's default float dtype
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
/// ```rust
/// use burn_tensor::{Device, DType, signal::blackman_window};
///
/// fn example() {
///     // Creating a window with default dtype
///     let device = Device::default();
///     let window_tensor = blackman_window(5, true, &device);
///     // Output: [0.0, 0.20077015, 0.84922993, 0.8492298, 0.2007701]
///
///     // Creating a window with explicit dtype.
///     // Note that this does not perform the computation at higher precision but it
///     // widens the storage of the returned tensor to F64.
///     let device = Device::default();
///     let window_tensor_f64 = blackman_window(5, true, (&device, DType::F64));
///     // Output: [0.0, 0.20077015, 0.84922993, 0.8492298, 0.2007701]
/// }
/// ```
pub fn blackman_window(
    size: usize,
    periodic: bool,
    options: impl Into<TensorCreationOptions>,
) -> Tensor<1> {
    let opt = options.into();
    let dtype = opt.resolve_dtype::<Float>();
    let shape = [size];
    check!(TensorCheck::creation_ops::<1>("BlackmanWindow", &shape));

    if size == 0 {
        return Tensor::<1>::empty(shape, opt).cast(dtype);
    }

    if size == 1 {
        return Tensor::<1>::ones(shape, opt).cast(dtype);
    }

    let size_i64 = i64::try_from(size).expect("BlackmanWindow size doesn't fit in i64 range.");
    let denominator = if periodic { size } else { size - 1 };
    let angular_increment = (2.0 * core::f64::consts::PI) / denominator as f64;
    let cos_val = Tensor::<1, Int>::arange(0..size_i64, &opt.device)
        .float()
        .mul_scalar(angular_increment)
        .cos();

    // Using the double angle property of cosine: cos(2θ) = 2cos^2(θ) - 1
    // w[n] = 0.42 - 0.5cos(2πn / N) + 0.08cos(4πn / N)
    // w[n] = 0.42 - 0.5cos(2πn / N) + 0.08cos(2 * (2πn / N))
    // w[n] = 0.42 - 0.5cos(2πn / N) + 0.08(2cos^2(2πn / N) - 1)
    // w[n] = 0.34 - 0.5cos(2πn / N) + 0.16cos^2(2πn / N)
    let first_cos_term = cos_val.clone().mul_scalar(-0.5);
    let second_cos_term = cos_val.powi_scalar(2).mul_scalar(0.16);
    first_cos_term
        .add(second_cos_term)
        .add_scalar(0.34)
        .cast(dtype)
}
