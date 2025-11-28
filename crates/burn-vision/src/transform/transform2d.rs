use burn_tensor::{
    Tensor,
    backend::Backend,
    grid::affine_grid_2d,
    ops::{GridSampleOptions, GridSamplePaddingMode, InterpolateMode},
};

/// 2D point transformation
///
/// Useful for resampling: rotating, scaling, translating, etc image tensors
pub struct Transform2D {
    // 2x3 transformation matrix, to be used with column vectors:
    // T(x) = Ax
    transform: [[f32; 3]; 2],
}

impl Transform2D {
    /// Transforms an image
    ///
    /// * `img` - Images tensor with shape (batch_size, channels, height, width)
    ///
    /// # Returns
    ///
    /// A tensor with the same as the input
    pub fn transform<B: Backend>(self, img: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, channels, height, width] = img.shape().dims();
        let transform = Tensor::<B, 2>::from(self.transform);
        let transform = transform.reshape([1, 2, 3]).expand([batch_size, 2, 3]);
        let grid = affine_grid_2d(transform, [batch_size, channels, height, width]);

        let options = GridSampleOptions::new(InterpolateMode::Bilinear)
            .with_padding_mode(GridSamplePaddingMode::Border)
            .with_align_corners(true);
        img.grid_sample_2d(grid, options)
    }

    /// Makes a 2d transformation composed of other transformations
    pub fn composed<I: IntoIterator<Item = Self>>(transforms: I) -> Self {
        let mut result = Self::identity();
        for t in transforms.into_iter() {
            result = result.mul(t);
        }
        result
    }

    /// Multiply two affine transforms represented as 2x3 matrices
    fn mul(self, other: Transform2D) -> Transform2D {
        let mut result = [[0.0f32; 3]; 2];

        // Row 0
        result[0][0] = self.transform[0][0] * other.transform[0][0]
            + self.transform[0][1] * other.transform[1][0];
        result[0][1] = self.transform[0][0] * other.transform[0][1]
            + self.transform[0][1] * other.transform[1][1];
        result[0][2] = self.transform[0][0] * other.transform[0][2]
            + self.transform[0][1] * other.transform[1][2]
            + self.transform[0][2];

        // Row 1
        result[1][0] = self.transform[1][0] * other.transform[0][0]
            + self.transform[1][1] * other.transform[1][0];
        result[1][1] = self.transform[1][0] * other.transform[0][1]
            + self.transform[1][1] * other.transform[1][1];
        result[1][2] = self.transform[1][0] * other.transform[0][2]
            + self.transform[1][1] * other.transform[1][2]
            + self.transform[1][2];

        Transform2D { transform: result }
    }

    /// Makes an identity transform (x = Ax)
    pub fn identity() -> Self {
        Self {
            transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        }
    }

    /// Makes a [`Transform2D`] for rotating a tensor
    ///
    /// * `theta` - In radians, the rotation
    /// * `cx` - Center of rotation, x
    /// * `cy` - Center of rotation, y
    pub fn rotation(theta: f32, cx: f32, cy: f32) -> Self {
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();

        let transform = [
            [cos_theta, -sin_theta, cx - cos_theta * cx + sin_theta * cy],
            [sin_theta, cos_theta, cy - sin_theta * cx - cos_theta * cy],
        ];

        Self { transform }
    }

    /// Makes a [`Transform2D`] for scaling an image tensor
    ///
    /// * `sx` - Scale factor in the x direction
    /// * `sy` - Scale factor in the y direction
    /// * `cx` - Center of scaling, x
    /// * `cy` - Center of scaling, y
    pub fn scale(sx: f32, sy: f32, cx: f32, cy: f32) -> Self {
        let transform = [[sx, 0.0, cx - sx * cx], [0.0, sy, cy - sy * cy]];

        Self { transform }
    }

    /// Makes a [`Transform2D`] for translating an image tensor
    ///
    /// * `tx` - Translation in the x direction
    /// * `ty` - Translation in the y direction
    pub fn translation(tx: f32, ty: f32) -> Self {
        let transform = [[1.0, 0.0, tx], [0.0, 1.0, ty]];

        Self { transform }
    }

    /// Applies a general shear transformation around the image center,
    /// combining both X and Y shear.
    ///
    /// # Arguments
    /// * `shx` - Shear factor along the X-axis.
    /// * `shy` - Shear factor along the Y-axis.
    /// * `cx`, `cy` - Coordinates of the image center.
    ///
    /// # Returns
    /// * `Self` with a combined shear transform matrix.
    pub fn shear(shx: f32, shy: f32, cx: f32, cy: f32) -> Self {
        let transform = [[1.0, shx, -shx * cy], [shy, 1.0, -shy * cx]];

        Self { transform }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use burn_tensor::Tolerance;
    type B = NdArray;

    #[test]
    fn transform_identity_translation() {
        let t = Transform2D::translation(0.0, 0.0);
        let image_original = Tensor::<B, 4>::from([[[[1., 0.], [0., 2.]]]]);
        let image_transformed = t.transform(image_original.clone());
        image_original
            .to_data()
            .assert_approx_eq(&image_transformed.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn transform_translation() {
        let t = Transform2D::translation(1., 1.);
        let image = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);
        // This result would change if the padding method is different
        let image_expected = Tensor::<B, 4>::from([[[[2.5, 3.], [3.5, 4.]]]]);
        let image = t.transform(image);
        image_expected
            .to_data()
            .assert_approx_eq(&image.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn transform_rotation_90_degrees() {
        let t = Transform2D::rotation(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        let image = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);
        let image_expected = Tensor::<B, 4>::from([[[[2., 4.], [1., 3.]]]]);
        let image = t.transform(image);
        image_expected
            .to_data()
            .assert_approx_eq(&image.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn transform_rotation_around_corner() {
        let cx = 1.;
        let cy = -1.;
        let t = Transform2D::rotation(std::f32::consts::FRAC_PI_2, cx, cy);
        let image = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);
        // This result would change if the padding method is different
        let image_expected = Tensor::<B, 4>::from([[[[2., 2.], [1., 1.]]]]);
        let image = t.transform(image);
        image_expected
            .to_data()
            .assert_approx_eq(&image.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn transform_scale() {
        let cx = 0.0;
        let cy = 0.0;
        let t = Transform2D::scale(0.5, 0.5, cx, cy);
        let image = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);
        let image_expected = Tensor::<B, 4>::from([[[[1.75, 2.25], [2.75, 3.25]]]]);
        let image = t.transform(image);
        image_expected
            .to_data()
            .assert_approx_eq(&image.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn transform_scale_around_corner() {
        let cx = 1.;
        let cy = -1.;
        let t = Transform2D::scale(0.5, 0.5, cx, cy);
        let image = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);
        let image_expected = Tensor::<B, 4>::from([[[[1.5, 2.], [2.5, 3.]]]]);
        let image = t.transform(image);
        image_expected
            .to_data()
            .assert_approx_eq(&image.to_data(), Tolerance::<f32>::balanced());
    }

    #[test]
    fn transform_combined() {
        let t1 = Transform2D::translation(0.2, -0.5);
        let t2 = Transform2D::rotation(std::f32::consts::FRAC_PI_3, 0., 0.);
        let t = Transform2D::composed([t1, t2]);

        let image = Tensor::<B, 4>::from([[[[1., 2.], [3., 4.]]]]);
        // This result would change if the padding method is different
        let image_expected =
            Tensor::<B, 4>::from([[[[1.7830127, 2.8660254], [1.1339746, 3.2830124]]]]);
        let image = t.transform(image);
        image_expected
            .to_data()
            .assert_approx_eq(&image.to_data(), Tolerance::<f32>::balanced());
    }
}
