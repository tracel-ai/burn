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
    ///   in the `0.0..=1.0` range.
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
    ///   in the `0.0..=1.0` range.
    ///
    /// # Returns
    /// The same-shape image tensor in HSV color space.
    fn rgb2hsv(self) -> Tensor<4>;
    /// Converts a batch of images from the HSV color space to the RGB color space.
    ///
    /// # Arguments
    /// * `self`: A batched image tensor of shape `[batch, channel, height, width]`.
    ///   The first channel (hue) is in the `0.0..360.0` range.
    ///   The other two (saturation and value) are in the `0.0..=1.0` range.
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::tensor::{Device, TensorData, Tolerance};

    fn pixel_to_tensor(rgb: [f32; 3]) -> Tensor<4> {
        Tensor::<4>::from_data(
            TensorData::new(rgb.to_vec(), [1, 3, 1, 1]),
            &Device::default(),
        )
    }

    fn gray_to_tensor(gray: f32) -> Tensor<4> {
        Tensor::<4>::from_data(
            TensorData::new(vec![gray], [1, 1, 1, 1]),
            &Device::default(),
        )
    }

    fn assert_gray(rgb: [f32; 3], expected: f32) {
        pixel_to_tensor(rgb)
            .rgb2gray()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(vec![expected], [1, 1, 1, 1]),
                Tolerance::default(),
            );
    }

    fn assert_rgb_from_gray(gray: f32, expected: &[f32; 3]) {
        gray_to_tensor(gray)
            .gray2rgb()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(expected.to_vec(), [1, 3, 1, 1]),
                Tolerance::default(),
            );
    }

    fn assert_hsv(rgb: [f32; 3], expected: &[f32; 3]) {
        pixel_to_tensor(rgb)
            .rgb2hsv()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(expected.to_vec(), [1, 3, 1, 1]),
                Tolerance::default(),
            );
    }

    fn assert_rgb(hsv: [f32; 3], expected: &[f32; 3]) {
        pixel_to_tensor(hsv)
            .hsv2rgb()
            .into_data()
            .assert_approx_eq::<f32>(
                &TensorData::new(expected.to_vec(), [1, 3, 1, 1]),
                Tolerance::default(),
            );
    }

    #[test]
    fn test_rgb2hsv() {
        // Expected values computed with OpenCV.
        assert_hsv([0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_hsv([1.0, 0.0, 0.0], &[0.0, 1.0, 1.0]);
        assert_hsv([0.0, 1.0, 0.0], &[120.0, 1.0, 1.0]);
        assert_hsv([0.0, 0.0, 1.0], &[240.0, 1.0, 1.0]);
        assert_hsv([1.0, 1.0, 1.0], &[0.0, 0.0, 1.0]);
        assert_hsv([0.5, 0.5, 0.5], &[0.0, 0.0, 0.5]);
        assert_hsv([0.2, 0.6, 0.9], &[205.7143, 0.7778, 0.9000]);
        assert_hsv([0.78, 0.0, 0.45], &[325.3846, 1.0000, 0.7800]);
    }

    #[test]
    fn test_hsv2rgb() {
        // Expected values computed with OpenCV.
        assert_rgb([0.0, 0.0, 0.0], &[0.0, 0.0, 0.0]);
        assert_rgb([0.0, 1.0, 1.0], &[1.0, 0.0, 0.0]);
        assert_rgb([120.0, 1.0, 1.0], &[0.0, 1.0, 0.0]);
        assert_rgb([240.0, 1.0, 1.0], &[0.0, 0.0, 1.0]);
        assert_rgb([0.0, 0.0, 1.0], &[1.0, 1.0, 1.0]);
        assert_rgb([0.0, 0.5, 1.0], &[1.0, 0.5, 0.5]);
        assert_rgb([180.0, 0.6, 0.9], &[0.36, 0.9, 0.9]);
        assert_rgb([273.0, 0.17, 0.42], &[0.3879, 0.3486, 0.42]);
    }

    #[test]
    fn rgb_hsv_roundtrip() {
        let expected = Tensor::cat(
            vec![
                pixel_to_tensor([0.8, 0.3, 0.1]),
                pixel_to_tensor([0.1, 0.7, 0.4]),
                pixel_to_tensor([0.2, 0.2, 0.9]),
                pixel_to_tensor([0.0, 0.0, 0.0]),
                pixel_to_tensor([1.0, 1.0, 1.0]),
            ],
            3,
        );
        let actual = expected.clone().rgb2hsv().hsv2rgb();
        actual
            .into_data()
            .assert_approx_eq::<f32>(&expected.into_data(), Tolerance::default());
    }

    #[test]
    fn test_rgb2gray() {
        // Luminance with the ITU-R BT.601 weights [0.299, 0.587, 0.114].
        assert_gray([0.0, 0.0, 0.0], 0.0);
        assert_gray([1.0, 0.0, 0.0], 0.299);
        assert_gray([0.0, 1.0, 0.0], 0.587);
        assert_gray([0.0, 0.0, 1.0], 0.114);
        assert_gray([1.0, 1.0, 1.0], 1.0);
        assert_gray([0.5, 0.5, 0.5], 0.5);
        assert_gray([0.2, 0.6, 0.9], 0.5146);
    }

    #[test]
    fn test_gray2rgb() {
        // Grayscale carries no color, so each channel is a copy of the gray value.
        assert_rgb_from_gray(0.0, &[0.0, 0.0, 0.0]);
        assert_rgb_from_gray(0.42, &[0.42, 0.42, 0.42]);
        assert_rgb_from_gray(1.0, &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn gray_rgb_roundtrip() {
        // The luminance weights sum to 1, so gray -> rgb -> gray is the identity.
        let gray = Tensor::<4>::from_data(
            TensorData::new(vec![0.0, 0.25, 0.5, 0.75, 1.0], [1, 1, 1, 5]),
            &Device::default(),
        );
        let actual = gray.clone().gray2rgb().rgb2gray();
        actual
            .into_data()
            .assert_approx_eq::<f32>(&gray.into_data(), Tolerance::default());
    }
}
