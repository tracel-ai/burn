use burn_tensor::{BasicOps, Tensor, TensorData, backend::Backend};

use crate::Transform2D;

/// Resample a 2D tensor using a coodinate transform and bilinear interpolation.
/// 
/// # Arguments
/// * `input` - The tensor to resample
/// * `transform` - A 3x3 transformation matrix to transform the coordinates of each value for 
///   resampling
///
/// # Returns
/// A new Tensor<B, 2> containing the rotated image.
pub fn resample<B: Backend, K: BasicOps<B>>(
    input: Tensor<B, 3, K>,
    transform: Transform2D<B>,
    default: f32,
) -> Tensor<B, 3, K> {
    let device = input.device();
    let shape = input.shape();
    let transform = transform;

    assert!(
        shape.dims[0] == 1,
        "First dimension is batch size, and batch resampling is unimplemeted"
    );

    
    let data = input.to_data();
    let src: Vec<f32> = data.to_vec::<f32>().unwrap();
    let (h, w) = (shape.dims[1] as usize, shape.dims[2] as usize);
    let mut dst = vec![0f32; src.len()];

    for y in 0..h {
        for x in 0..w {
            // Transform coordinates
            let (src_x, src_y) = transform.transform(x as f32, y as f32);

            // Bilinear interpolation
            let pixel = sample_interpolate(src_x, src_y, &src, w, h).unwrap_or(default);

            dst[y * w + x] = pixel;
        }
    }

    // Build output tensor from rotated data
    let tensor = Tensor::<B, 1, K>::from_data(TensorData::from(dst.as_slice()), &device);
    let tensor: Tensor<B, 3, K> = tensor.reshape(shape);

    //save_as_img::<B>(&image, 200, 200, "tmp_after.png").unwrap();
    //panic!("Check");

    tensor
}

/// Sample a tensor, interpolating between indices
fn sample_interpolate(x: f32, y: f32, data: &[f32], w: usize, h: usize) -> Option<f32> {
    if x >= 0.0 && x < (w - 1) as f32 && y >= 0.0 && y < (h - 1) as f32 {
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let i00 = y0 * w + x0;
        let i10 = y0 * w + (x0 + 1);
        let i01 = (y0 + 1) * w + x0;
        let i11 = (y0 + 1) * w + (x0 + 1);

        let v00 = data[i00];
        let v10 = data[i10];
        let v01 = data[i01];
        let v11 = data[i11];

        let v0 = v00 * (1.0 - dx) + v10 * dx;
        let v1 = v01 * (1.0 - dx) + v11 * dx;
        let val = v0 * (1.0 - dy) + v1 * dy;

        Some(val)
    } else {
        None
    }
}
