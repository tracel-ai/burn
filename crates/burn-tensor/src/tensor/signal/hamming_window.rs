use crate::{Float, Int, Tensor, TensorCreationOptions, check, check::TensorCheck};

/// Creates a 1D Hamming window.
///
#[cfg_attr(
    doc,
    doc = r#"
$$w_n = \alpha - \beta \cos\left(\frac{2\pi n}{N}\right)$$

where $\alpha = 25/46$, $\beta = 1 - \alpha$, and $N$ = `size` when `periodic` is `true`, or $N$ = `size - 1` when `periodic` is `false`.
"#
)]
#[cfg_attr(
    not(doc),
    doc = "`w_n = 25/46 - 21/46 * cos(2πn/N)` where N = size (periodic) or N = size-1 (symmetric)"
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
/// use burn_tensor::signal::hamming_window;
///
/// fn example() {
///     let device = Default::default();
///     let window = hamming_window(8, true, &device);
///     println!("{window}");
/// }
/// ```
pub fn hamming_window(
    size: usize,
    periodic: bool,
    options: impl Into<TensorCreationOptions>,
) -> Tensor<1> {
    let opt = options.into();
    let dtype = opt.resolve_dtype::<Float>();
    let shape = [size];
    check!(TensorCheck::creation_ops::<1>("HammingWindow", &shape));

    if size == 0 {
        return Tensor::<1>::empty(shape, opt).cast(dtype);
    }

    if size == 1 {
        return Tensor::<1>::ones(shape, opt).cast(dtype);
    }

    let size_i64 = i64::try_from(size).expect("HammingWindow size doesn't fit in i64 range.");
    let denominator = if periodic { size } else { size - 1 };
    let angular_increment = (2.0 * core::f64::consts::PI) / denominator as f64;

    let alpha = 25.0_f64 / 46.0_f64;
    let beta = 1.0 - alpha;

    Tensor::<1, Int>::arange(0..size_i64, &opt.device)
        .float()
        .mul_scalar(angular_increment)
        .cos()
        .mul_scalar(-beta)
        .add_scalar(alpha)
        .cast(dtype)
}
