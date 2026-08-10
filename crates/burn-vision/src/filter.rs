use burn_core as burn;
use burn_core::{
    Tensor,
    config::Config,
    tensor::{
        Device, FloatDType,
        module::conv2d,
        ops::{ConvOptions, PadMode},
    },
};
use burn_std::{
    rand::{self, RngExt, SeedableRng, StdRng},
    sync::Mutex,
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
/// # Retuns
/// The filtered batched images, same shape as the input.
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

/// Configuration for a [`BoxBlur`] augmentation.
#[derive(Config, Debug)]
pub struct BoxBlurConfig {
    /// The size of one side of the (square) kernel, in pixels.
    #[config(default = 3)]
    pub kernel_size: usize,
    /// Probability of the blur being applied.
    #[config(default = 1.0)]
    pub probability: f32,
}

impl BoxBlurConfig {
    /// Builds the box-filter kernel on `device` and returns the ready [`BoxBlur`].
    ///
    /// # Panics
    /// Panics unless `kernel_size` is odd and at least `3`.
    pub fn init(&self, device: &Device) -> BoxBlur {
        assert!(
            self.kernel_size >= 3 && self.kernel_size % 2 == 1,
            "kernel_size must be odd and at least 3, got {}",
            self.kernel_size
        );

        let weight = 1.0 / (self.kernel_size * self.kernel_size) as f32;
        let kernel = Tensor::<2>::full([self.kernel_size, self.kernel_size], weight, device);

        BoxBlur {
            kernel,
            probability: self.probability,
            rng: Mutex::new(StdRng::from_rng(&mut rand::get_seeded_rng())),
        }
    }
}

/// Blurs a batch of images with a uniform box filter, applied with a given
/// probability. Built from a [`BoxBlurConfig`].
#[derive(Debug)]
pub struct BoxBlur {
    /// The precomputed `[kernel_size, kernel_size]` box-filter kernel.
    kernel: Tensor<2>,
    /// Probability of the blur being applied.
    probability: f32,
    rng: Mutex<StdRng>,
}

impl BoxBlur {
    /// Draws from `seed` rather than from the thread's generator.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = Mutex::new(StdRng::seed_from_u64(seed));
        self
    }

    fn sample(&self) -> bool {
        let mut rng = self.rng.lock();

        rng.random::<f32>() < self.probability
    }

    /// Unconditionally applies the box filter to the batch of images.
    pub fn apply(&self, images: Tensor<4>) -> Tensor<4> {
        filter2d(images, self.kernel.clone(), PadMode::Reflect)
    }

    /// Applies the box filter with probability `probability`, otherwise returns
    /// the images unchanged.
    pub fn forward(&self, images: Tensor<4>) -> Tensor<4> {
        match self.sample() {
            true => self.apply(images),
            false => images,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::Tolerance;

    #[test]
    fn filter2d_identity() {
        let images = Tensor::<4>::from([[[[1., 2.], [3., 4.]]]]);
        let kernel = Tensor::<2>::from([[1.]]);
        let out = filter2d(images.clone(), kernel, PadMode::Reflect);
        images
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

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

    #[test]
    fn box_blur_constant_image_is_identity() {
        let images = Tensor::<4>::from([[[[5., 5., 5.], [5., 5., 5.], [5., 5., 5.]]]]);
        let blur = BoxBlurConfig::new().init(&Device::default());
        let out = blur.apply(images.clone());
        images
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn box_blur_matches_manual_filter2d() {
        let images = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]]);
        let kernel = Tensor::<2>::from([[1. / 9.; 3]; 3]);
        let expected = filter2d(images.clone(), kernel, PadMode::Reflect);
        let out = BoxBlurConfig::new().init(&Device::default()).apply(images);
        expected
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn box_blur_zero_probability_is_passthrough() {
        let images = Tensor::<4>::from([[[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]]);
        let blur = BoxBlurConfig::new()
            .with_probability(0.0)
            .init(&Device::default());
        let out = blur.forward(images.clone());
        images
            .to_data()
            .assert_approx_eq(&out.to_data(), Tolerance::<f32>::balanced());
    }
}
