use burn_core::{
    Tensor,
    tensor::{
        FloatDType,
        module::conv2d,
        ops::{ConvOptions, PadMode},
    },
};

/// Depthwise convolution of a tensor with a 2D kernel.
///
/// `images` is laid out as `[N, C, H, W]`. Every channel is filtered
/// independently — a grouped convolution with `groups = C`, so the same kernel
/// is broadcast across channels without mixing them. The output keeps the
/// input's spatial size for odd kernel dimensions; even dimensions pad
/// asymmetrically toward the top/left.
///
/// # Arguments
///
/// * `input` - A tensor shaped `[N, C, H, W]`.
/// * `kernel` - The A 2D tensor to convolve with the input.
/// * `border` - The [padding mode](PadMode) applied to the input before the
///              convolution.
///
/// The kernel may be any float dtype: it is cast to the image's
/// dtype so the convolution operands agree.
pub fn filter2d(images: Tensor<4>, kernel: Tensor<2>, border: PadMode) -> Tensor<4> {
    let [_, channels, _, _] = images.dims();
    let [kh, kw] = kernel.dims();

    // One depthwise weight per channel: `[C, 1, kh, kw]`, matched to the image's
    // float dtype so the convolution's operands agree.
    let weight = kernel
        .reshape([1, 1, kh, kw])
        .expand([channels, 1, kh, kw])
        .cast(FloatDType::from(images.dtype()));

    // "Same"-size output: pad half the kernel on each side. The `(l, r, t, b)`
    // tuple pads `(t, b)` along H and `(l, r)` along W.
    let images = images.pad((kw / 2, kw / 2, kh / 2, kh / 2), border);

    conv2d(
        images,
        weight,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::Tolerance;

    /// A 1x1 identity kernel is a no-op regardless of border (no padding).
    #[test]
    fn filter2d_identity() {
        let images = Tensor::<4>::from([[[[1., 2.], [3., 4.]]]]);
        let kernel = Tensor::<2>::from([[1.]]);
        let out = filter2d(images.clone(), kernel, PadMode::Reflect);
        images
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

    /// A 3x3 box blur with a replicate (`Edge`) border on a constant image
    /// returns the same constant (the weights sum to 1 and every neighbour
    /// equals the centre).
    #[test]
    fn filter2d_box_blur_edge_constant_image() {
        let images = Tensor::<4>::from([[[[5., 5., 5.], [5., 5., 5.], [5., 5., 5.]]]]);
        let kernel = Tensor::<2>::from([[1. / 9.; 3]; 3]);
        let out = filter2d(images.clone(), kernel, PadMode::Edge);
        images
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

    /// The same box blur with a zero (`Constant`) border darkens the edges: a
    /// pixel's window loses the cells that fall outside the image. This
    /// distinguishes the border mode from the `Edge` case above.
    #[test]
    fn filter2d_box_blur_constant_border() {
        let images = Tensor::<4>::from([[[[5., 5., 5.], [5., 5., 5.], [5., 5., 5.]]]]);
        let kernel = Tensor::<2>::from([[1. / 9.; 3]; 3]);
        let out = filter2d(images, kernel, PadMode::Constant(0.));
        // Real cells per 3x3 window (clamped to the 3x3 image), times 5/9.
        let expected = Tensor::<4>::from([[[
            [20. / 9., 30. / 9., 20. / 9.],
            [30. / 9., 45. / 9., 30. / 9.],
            [20. / 9., 30. / 9., 20. / 9.],
        ]]]);
        expected
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

    /// A `[1, 3]` kernel filters only along width, leaving the height axis
    /// untouched.
    #[test]
    fn filter2d_rectangular_kernel_width_only() {
        let images = Tensor::<4>::from([[[[1., 2., 3.]]]]);
        let kernel = Tensor::<2>::from([[1., 1., 1.]]);
        let out = filter2d(images, kernel, PadMode::Edge);
        // Edge-replicated sums: [1+1+2, 1+2+3, 2+3+3].
        let expected = Tensor::<4>::from([[[[4., 6., 8.]]]]);
        expected
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

    /// Channels are filtered independently — a `1x1` scaling kernel scales each
    /// channel on its own with no cross-channel mixing.
    #[test]
    fn filter2d_is_depthwise() {
        let images = Tensor::<4>::from([[[[1.]], [[10.]]]]); // [1, 2, 1, 1]
        let kernel = Tensor::<2>::from([[2.]]);
        let out = filter2d(images, kernel, PadMode::Reflect);
        let expected = Tensor::<4>::from([[[[2.]], [[20.]]]]);
        expected
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }
}
