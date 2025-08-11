use burn_tensor::{ElementConversion, Int, Tensor, backend::Backend};

/// 2D point transformation
///
/// Useful for resampling: rotating, scaling, translating, etc image tensors
pub struct Transform2D {
    // 3x3 transformation matrix, to be used with collumn vectors:
    // T(x) = Ax
    transform: [[f32; 3]; 3],
}

impl Transform2D {
    /// Generate a tensor with homogeonous coodinates of each element's
    /// transformed location
    ///
    /// * `batch` - batch size
    /// * `dims` - dimensions as \[height, width\]
    ///
    /// # Returns
    ///
    /// Tensor with shape (batch, height, width, 2), where dim 2 is (x, y)
    /// All coordinates are broadcast on the batch dim
    pub fn mapping<B: Backend>(self, batch: usize, dims: [usize; 2]) -> Tensor<B, 4> {
        let height = dims[0];
        let width = dims[1];
        let device = &Default::default();

        let x = Tensor::<B, 1, Int>::arange(0..width as i64, device)
            .reshape([1, width, 1])
            .expand([height, width, 1]);
        let y = Tensor::<B, 1, Int>::arange(0..height as i64, device)
            .reshape([height, 1, 1])
            .expand([height, width, 1]);

        // from ints (0..width) and (0..height), to (-1.0..1.0)
        let x = x
            .float()
            .div_scalar((width as f32 / 2.0).elem::<f32>())
            .sub_scalar((1 as f32).elem::<f32>());
        let y = y
            .float()
            .div_scalar((height as f32 / 2.0).elem::<f32>())
            .sub_scalar((1 as f32).elem::<f32>());
        let w = Tensor::<B, 3>::ones([height, width, 1], device);

        let grid = Tensor::cat(vec![x, y, w], 2);

        // Arange coordinates as collumn vectors for transformation
        let grid = grid.reshape([height, width, 3, 1]);
        let transform = Tensor::<B, 2>::from(self.transform).reshape([1, 1, 3, 3]);
        let result = transform.matmul(grid);
        let result = result.reshape([height, width, 3]);

        // seperate (x, y) and w
        let mut grid = result.split_with_sizes(vec![2, 1], 2);
        // homogeneous coodinates stuf
        let grid_xy = grid.remove(0);
        let grid_w = grid.remove(0);
        let grid_w = grid_w.expand([height, width, 2]);
        let grid_xy = grid_xy.div(grid_w);

        // Broadcast along batches
        grid_xy
            .reshape([1, height, width, 2])
            .expand([batch, height, width, 2])
    }

    /// Apply a transform to another transform: multiplying the transforms
    pub fn mul(&self, other: Transform2D) -> Transform2D {
        let mut result = [[0.0f32; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = self.transform[i][0] * other.transform[0][j]
                    + self.transform[i][1] * other.transform[1][j]
                    + self.transform[i][2] * other.transform[2][j];
            }
        }

        Transform2D { transform: result }
    }

    /// Makes an identity transform (x = Ax)
    pub fn identity() -> Self {
        Self {
            transform: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
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
            transform,
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
            transform,
        }
    }

    /// Makes a [ResampleTransform] for translating an image tensor
    ///
    /// * `tx` - Translation in the x direction
    /// * `ty` - Translation in the y direction
    pub fn translation(tx: f32, ty: f32) -> Self {
        let transform = [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]];

        Self {
            transform,
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
            transform,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use burn_ndarray::NdArray;
//     type B = NdArray;

//     const EPS: f32 = 1e-5;

//     fn approx_eq(a: f32, b: f32) -> bool {
//         (a - b).abs() < EPS
//     }

//     #[test]
//     fn test_identity_translation() {
//         let t = Transform2D::<B>::translation(0.0, 0.0);
//         let (x, y) = t.transform(10.0, 20.0);
//         assert!(approx_eq(x, 10.0));
//         assert!(approx_eq(y, 20.0));
//     }

//     #[test]
//     fn test_translation() {
//         let t = Transform2D::<B>::translation(5.0, -3.0);
//         let (x, y) = t.transform(10.0, 20.0);
//         assert!(approx_eq(x, 15.0));
//         assert!(approx_eq(y, 17.0));
//     }

//     #[test]
//     fn test_rotation_90_degrees() {
//         let t = Transform2D::<B>::rotation(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
//         let (x, y) = t.transform(1.0, 0.0);
//         assert!(approx_eq(x, 0.0));
//         assert!(approx_eq(y, 1.0));
//     }

//     #[test]
//     fn test_rotation_around_center() {
//         let cx = 50.0;
//         let cy = 50.0;
//         let t = Transform2D::<B>::rotation(std::f32::consts::PI, cx, cy);
//         let (x, y) = t.transform(60.0, 50.0);
//         assert!(approx_eq(x, 40.0));
//         assert!(approx_eq(y, 50.0));
//     }

//     #[test]
//     fn test_scale() {
//         let cx = 0.0;
//         let cy = 0.0;
//         let t = Transform2D::<B>::scale(2.0, 3.0, cx, cy);
//         let (x, y) = t.transform(1.0, 1.0);
//         assert!(approx_eq(x, 2.0));
//         assert!(approx_eq(y, 3.0));
//     }

//     #[test]
//     fn test_scale_around_center() {
//         let cx = 10.0;
//         let cy = 10.0;
//         let t = Transform2D::<B>::scale(2.0, 2.0, cx, cy);
//         let (x, y) = t.transform(12.0, 10.0);
//         println!("Scale: {x},{y}");
//         assert!(approx_eq(x, 14.0));
//         assert!(approx_eq(y, 10.0));
//     }

//     #[test]
//     fn test_shear_x() {
//         let cx = 0.0;
//         let cy = 0.0;
//         let t = Transform2D::<B>::shear(1.0, 0.0, cx, cy);
//         let (x, y) = t.transform(1.0, 1.0);
//         assert!(approx_eq(x, 2.0)); // x + 1*y
//         assert!(approx_eq(y, 1.0));
//     }

//     #[test]
//     fn test_shear_y() {
//         let cx = 0.0;
//         let cy = 0.0;
//         let t = Transform2D::<B>::shear(0.0, 1.0, cx, cy);
//         let (x, y) = t.transform(1.0, 1.0);
//         assert!(approx_eq(x, 1.0));
//         assert!(approx_eq(y, 2.0)); // y + 1*x
//     }

//     #[test]
//     fn test_combined_transform() {
//         let t1 = Transform2D::<B>::translation(5.0, 0.0);
//         let t2 = Transform2D::<B>::scale(2.0, 2.0, 0.0, 0.0);
//         let t = t1.mul(t2);
//         let (x, y) = t.transform(1.0, 1.0);
//         assert!(approx_eq(x, 7.0)); // (2*1 + 5)
//         assert!(approx_eq(y, 2.0)); // 2*1
//     }
// }
