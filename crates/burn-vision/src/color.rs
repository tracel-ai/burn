use burn_core::tensor::{DType, Tensor};

/// Weights of each RGB channel for grayscale conversion, as per
/// [OpenCV](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html).
const RGB2GRAY_WEIGHTS: [f32; 3] = [0.299, 0.587, 0.114];
const EPS: f32 = 1e-8;

/// Color space conversion tensor operations.
pub trait ColorConversion {
    /// Converts a batch of images from the RGB color space to grayscale.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`
    ///           in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The grayscale image tensor of shape `[batch, 1, height, width]`.
    fn rgb2gray(self) -> Tensor<4>;
    /// Converts a batch of grayscale images to the RGB color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, 1, height, width]`.
    ///
    /// # Returns
    /// The converted batched image tensor of shape `[batch, 3, height, width]`
    fn gray2rgb(self) -> Tensor<4>;
    /// Converts a batch of images from the RGB color space to the HSV color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`
    ///           in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The same-shape image tensor in HSV color space.
    fn rgb2hsv(self) -> Tensor<4>;
    /// Converts a batch of images from the HSV color space to the RGB color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`.
    ///           The first channel (hue) is in the `0.0..360.0` range.
    ///           The other two (saturation and value) are in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The same-shape image tensor in RGB color space.
    fn hsv2rgb(self) -> Tensor<4>;
}

/// Quantized floats aren't supported, matching the other float vision ops.
fn reject_quantized(images: &Tensor<4>) {
    if matches!(images.dtype(), DType::QFloat(_)) {
        unimplemented!("Quantized float is not supported");
    }
}

fn channel(img: &Tensor<4>, at: usize) -> Tensor<4> {
    img.clone().narrow(1, at, 1)
}

/// One channel back out of a hue, its value and its chroma.
/// `target` - target channel : `0` for red, `1` for green and `2` for blue.
fn from_hue(sixths: &Tensor<4>, target: usize, value: &Tensor<4>, chroma: &Tensor<4>) -> Tensor<4> {
    // Rotate the wheel depending on the target channel.
    let rotated = sixths
        .clone()
        .add_scalar(5.0 - (target as f32) * 2.0)
        .remainder_scalar(6.0);
    let ramp = rotated.clone().min_pair(4.0 - rotated).clamp(0.0, 1.0);

    value.clone() - chroma.clone() * ramp
}

/// Formulas derived from the [OpenCV docs](https://docs.opencv.org/4.13.0/de/d25/imgproc_color_conversions.html).
impl ColorConversion for Tensor<4> {
    fn rgb2gray(self) -> Tensor<4> {
        reject_quantized(&self);
        channel(&self, 0).mul_scalar(RGB2GRAY_WEIGHTS[0])
            + channel(&self, 1).mul_scalar(RGB2GRAY_WEIGHTS[1])
            + channel(&self, 2).mul_scalar(RGB2GRAY_WEIGHTS[2])
    }

    fn gray2rgb(self) -> Tensor<4> {
        reject_quantized(&self);
        // RGB has no more information than the single grayscale channel, so each
        // of R, G and B is just a copy of it.
        self.repeat_dim(1, 3)
    }

    fn rgb2hsv(self) -> Tensor<4> {
        reject_quantized(&self);
        let (red, green, blue) = (channel(&self, 0), channel(&self, 1), channel(&self, 2));
        let value = self.clone().max_dim(1);
        let chroma = value.clone() - self.min_dim(1);

        // Clamped values are always multiplied by 0, so this has no numerical impact.
        let by_chroma = chroma.clone().clamp_min(EPS);
        let by_value = value.clone().clamp_min(EPS);

        // OpenCV takes the first of the three that matches, so
        // red wins a tie with green and green one with blue.
        let sixths = (red.clone() - green.clone()) / by_chroma.clone() + 4.0;
        let sixths = sixths.mask_where(
            value.clone().equal(green.clone()),
            (blue.clone() - red.clone()) / by_chroma.clone() + 2.0,
        );
        let sixths = sixths.mask_where(value.clone().equal(red), (green - blue) / by_chroma);

        Tensor::cat(
            vec![
                sixths.mul_scalar(360.0 / 6.0).remainder_scalar(360.0),
                chroma / by_value,
                value,
            ],
            1,
        )
    }

    fn hsv2rgb(self) -> Tensor<4> {
        reject_quantized(&self);
        let (hue, saturation, value) = (channel(&self, 0), channel(&self, 1), channel(&self, 2));
        let sixths = hue.div_scalar(360.0 / 6.0);
        let chroma = value.clone() * saturation;

        Tensor::cat(
            vec![
                from_hue(&sixths, 0, &value, &chroma),
                from_hue(&sixths, 1, &value, &chroma),
                from_hue(&sixths, 2, &value, &chroma),
            ],
            1,
        )
    }
}
