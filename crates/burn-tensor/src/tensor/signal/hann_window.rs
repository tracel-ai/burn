use crate::{Float, Int, Tensor, TensorCreationOptions, check, check::TensorCheck};

/// Creates a 1D Hann window.
///
#[cfg_attr(
    doc,
    doc = r#"
$$w_n = 0.5 - 0.5 \cos\left(\frac{2\pi n}{N}\right)$$

where $N$ = `size` when `periodic` is `true`, or $N$ = `size - 1` when `periodic` is `false`.
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`w_n = 0.5 - 0.5 * cos(2πn/N)` where N = size (periodic) or N = size-1 (symmetric)"
)]
///
/// # Notes
///
/// - `size == 0` returns an empty tensor.
/// - `size == 1` returns `[1.0]` regardless of `periodic`.
///
/// # Example
///
/// ```rust
/// use burn_tensor::Device;
/// use burn_tensor::signal::hann_window;
///
/// fn example() {
///     let device = Default::default();
///     let window = hann_window(8, true, &device);
///     println!("{window}");
/// }
/// ```
pub fn hann_window(
    size: usize,
    periodic: bool,
    options: impl Into<TensorCreationOptions>,
) -> Tensor<1> {
    let opt = options.into();
    let dtype = opt.resolve_dtype::<Float>();
    let shape = [size];
    check!(TensorCheck::creation_ops::<1>("HannWindow", &shape));

    if size == 0 {
        return Tensor::<1>::empty(shape, opt).cast(dtype);
    }

    if size == 1 {
        return Tensor::<1>::ones(shape, opt).cast(dtype);
    }

    let size_i64 = i64::try_from(size).expect("HannWindow size doesn't fit in i64 range.");
    let denominator = if periodic { size } else { size - 1 };
    let angular_increment = (2.0 * core::f64::consts::PI) / denominator as f64;

    Tensor::<1, Int>::arange(0..size_i64, &opt.device)
        .float()
        .mul_scalar(angular_increment)
        .cos()
        .mul_scalar(-0.5)
        .add_scalar(0.5)
        .cast(dtype)
}
