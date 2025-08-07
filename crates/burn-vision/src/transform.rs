use burn_tensor::{Device, Int, Shape, Tensor, backend::Backend};

/// 2D point transformation
///
/// Useful for resampling: rotating, scaling, translating, etc image tensors
pub struct Transform2D<B: Backend> {
    // 3x3 transformation matrix, to be used with collumn vectors:
    // T(x) = Ax
    pub(crate) transform: Tensor<B, 2>,
}

// TODO move this somewhere
/// Generate a [width, height, 3] tensor with homogeonous coodinates of each element in the 3rd dimention: [x, y, 1.0]
fn arange_coords_2d<B: Backend>(w: usize, h: usize) -> Tensor<B, 3, Int> {
    let device = &Default::default();
    let x = Tensor::<B, 1, Int>::arange(0..w as i64, device)
        .reshape([w, 1, 1])
        .expand([w, h, 1]);
    let y = Tensor::<B, 1, Int>::arange(0..h as i64, device)
        .reshape([1, h, 1])
        .expand([w, h, 1]);
    let w = Tensor::<B, 3, Int>::ones([w, h, 1], device);

    Tensor::cat(vec![x, y, w], 2)
}

impl<B: Backend> Transform2D<B> {
    /// Apply a transform to another transform: multiplying the transforms
    pub fn mul(mut self, rhs: Self) -> Self {
        self.transform = self.transform.clone().matmul(rhs.transform);

        self
    }

    /// Transform a 2D point
    pub fn transform(&self, x: f32, y: f32) -> (f32, f32) {
        // Point is a column vector
        let point = Tensor::<B, 2>::from([[x], [y], [1.0]]);
        let new_point = self.transform.clone().matmul(point);
        let new_point = new_point.to_data().to_vec::<f32>().unwrap();
        assert!(new_point.len() == 3, "new coords invalid");
        let (x, y, w) = (new_point[0], new_point[1], new_point[2]);

        (x / w, y / w)
    }

    /// Makes a [ResampleTransform] for rotating a tensor
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
            [0.0, 0.0, 1.0],
        ];

        Self {
            transform: Tensor::<B, 2>::from(transform),
        }
    }

    /// Makes a [ResampleTransform] for scaling an image tensor
    ///
    /// * `sx` - Scale factor in the x direction
    /// * `sy` - Scale factor in the y direction
    /// * `cx` - Center of scaling, x
    /// * `cy` - Center of scaling, y
    pub fn scale(sx: f32, sy: f32, cx: f32, cy: f32) -> Self {
        let transform = [
            [sx, 0.0, cx - sx * cx],
            [0.0, sy, cy - sy * cy],
            [0.0, 0.0, 1.0],
        ];

        Self {
            transform: Tensor::<B, 2>::from(transform),
        }
    }

    /// Makes a [ResampleTransform] for translating an image tensor
    ///
    /// * `tx` - Translation in the x direction
    /// * `ty` - Translation in the y direction
    pub fn translation(tx: f32, ty: f32) -> Self {
        let transform = [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]];

        Self {
            transform: Tensor::<B, 2>::from(transform),
        }
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
        let transform = [
            [1.0, shx, -shx * cy],
            [shy, 1.0, -shy * cx],
            [0.0, 0.0, 1.0],
        ];

        Self {
            transform: Tensor::<B, 2>::from(transform),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    type B = NdArray;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_identity_translation() {
        let t = Transform2D::<B>::translation(0.0, 0.0);
        let (x, y) = t.transform(10.0, 20.0);
        assert!(approx_eq(x, 10.0));
        assert!(approx_eq(y, 20.0));
    }

    #[test]
    fn test_translation() {
        let t = Transform2D::<B>::translation(5.0, -3.0);
        let (x, y) = t.transform(10.0, 20.0);
        assert!(approx_eq(x, 15.0));
        assert!(approx_eq(y, 17.0));
    }

    #[test]
    fn test_rotation_90_degrees() {
        let t = Transform2D::<B>::rotation(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        let (x, y) = t.transform(1.0, 0.0);
        assert!(approx_eq(x, 0.0));
        assert!(approx_eq(y, 1.0));
    }

    #[test]
    fn test_rotation_around_center() {
        let cx = 50.0;
        let cy = 50.0;
        let t = Transform2D::<B>::rotation(std::f32::consts::PI, cx, cy);
        let (x, y) = t.transform(60.0, 50.0);
        assert!(approx_eq(x, 40.0));
        assert!(approx_eq(y, 50.0));
    }

    #[test]
    fn test_scale() {
        let cx = 0.0;
        let cy = 0.0;
        let t = Transform2D::<B>::scale(2.0, 3.0, cx, cy);
        let (x, y) = t.transform(1.0, 1.0);
        assert!(approx_eq(x, 2.0));
        assert!(approx_eq(y, 3.0));
    }

    #[test]
    fn test_scale_around_center() {
        let cx = 10.0;
        let cy = 10.0;
        let t = Transform2D::<B>::scale(2.0, 2.0, cx, cy);
        let (x, y) = t.transform(12.0, 10.0);
        println!("Scale: {x},{y}");
        assert!(approx_eq(x, 14.0));
        assert!(approx_eq(y, 10.0));
    }

    #[test]
    fn test_shear_x() {
        let cx = 0.0;
        let cy = 0.0;
        let t = Transform2D::<B>::shear(1.0, 0.0, cx, cy);
        let (x, y) = t.transform(1.0, 1.0);
        assert!(approx_eq(x, 2.0)); // x + 1*y
        assert!(approx_eq(y, 1.0));
    }

    #[test]
    fn test_shear_y() {
        let cx = 0.0;
        let cy = 0.0;
        let t = Transform2D::<B>::shear(0.0, 1.0, cx, cy);
        let (x, y) = t.transform(1.0, 1.0);
        assert!(approx_eq(x, 1.0));
        assert!(approx_eq(y, 2.0)); // y + 1*x
    }

    #[test]
    fn test_combined_transform() {
        let t1 = Transform2D::<B>::translation(5.0, 0.0);
        let t2 = Transform2D::<B>::scale(2.0, 2.0, 0.0, 0.0);
        let t = t1.mul(t2);
        let (x, y) = t.transform(1.0, 1.0);
        assert!(approx_eq(x, 7.0)); // (2*1 + 5)
        assert!(approx_eq(y, 2.0)); // 2*1
    }
}
