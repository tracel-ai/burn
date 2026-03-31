use burn_backend::Backend;

use crate::Tensor;

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
/// use burn_tensor::backend::Backend;
/// use burn_tensor::signal::hann_window;
///
/// fn example<B: Backend>() {
///     let device = B::Device::default();
///     let window = hann_window::<B>(8, true, &device);
///     println!("{window}");
/// }
/// ```
pub fn rfft<B: Backend, const D: usize>(
    signal: Tensor<B, D>,
    _dim: usize,
    // options: impl Into<TensorCreationOptions<B>>,
) -> Tensor<B, D> {
    let _shape = signal.shape();
    signal
    //check!(TensorCheck::creation_ops::<1>("HannWindow", &shape));

    //Tensor::<B, 1, Int>::arange(0..size_i64, &opt.device)
    //    .float()
    //    .mul_scalar(angular_increment)
    //    .cos()
    //    .mul_scalar(-0.5)
    //    .add_scalar(0.5)
    //    .cast(dtype)
}
