use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Tensor};
use image::{Rgb, RgbImage};
use std::fs;
use std::path::Path;

pub struct TensorDisplayOptions {
    pub dim_order: ImageDimOrder,
    pub color_opts: ColorDisplayOpts,
    pub batch_opts: Option<BatchDisplayOpts>,
}

pub enum ImageDimOrder {
    HW,
    CHW,
    HWC,
    NHW,
    NCHW,
    NHWC,
}

pub enum ColorDisplayOpts {
    Rgb,
    Monochrome {
        min: [f32; 3],
        max: [f32; 3],
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum BatchDisplayOpts {
    Tiled,
    Aggregated,
}

/// Save a tensor of a batch of images as an image
///
/// * `tensor` - Image batch with shape (N, height, width)
pub fn save_as_img<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    opts: TensorDisplayOptions,
    width_out: usize,
    height_out: usize,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Output file
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tensor = normalize(tensor);

    // convert to (N,C,H,W) format
    let tensor: Tensor<B, 4> = match opts.dim_order {
        ImageDimOrder::HW => {
            let [h, w] = tensor.shape().dims();
            tensor.reshape([1, 1, h, w])
        }
        ImageDimOrder::CHW => {
            let [c, h, w] = tensor.shape().dims();
            tensor.reshape([1, c, h, w])
        }
        ImageDimOrder::HWC => {
            let [h, w, c] = tensor.shape().dims();
            tensor.swap_dims(0, 2).swap_dims(1, 2).reshape([1, c, h, w])
        }
        ImageDimOrder::NHW => {
            let [n, h, w] = tensor.shape().dims();
            tensor.reshape([n, 1, h, w])
        }
        ImageDimOrder::NCHW => tensor.reshape([0, 0, 0, 0]),
        ImageDimOrder::NHWC => tensor.swap_dims(1, 3).swap_dims(2, 3).reshape([0, 0, 0, 0]),
    };

    let data = tensor.to_data();
    let shape = data.shape.clone();
    let (batch, channels, src_height, src_width) = (shape[0], shape[1], shape[2], shape[3]);

    let mut img = if let Some(opts) = opts.batch_opts
        && BatchDisplayOpts::Tiled == opts
    {
        RgbImage::new(width_out as u32, (height_out * batch) as u32)
    } else {
        RgbImage::new(width_out as u32, height_out as u32)
    };

    let data_vec = data.to_vec::<f32>().unwrap();

    let mut channel_vals = vec![0 as f32; channels]; // value for each channel in a given pixel
    for n in 0..batch {
        for x in 0..width_out {
            for y in 0..height_out {
                let i = ((x as f32) / (width_out as f32) * (src_width as f32))
                    .floor()
                    .clamp(0.0, src_width as f32) as usize;
                let j = ((y as f32) / (height_out as f32) * (src_height as f32))
                    .floor()
                    .clamp(0.0, src_height as f32) as usize;

                for c in 0..channels {
                    channel_vals[c] =
                        data_vec[i + (j + (n * channels + c) * src_height) * src_width];
                }

                let (x, y) = if let Some(opts) = opts.batch_opts
                    && BatchDisplayOpts::Tiled == opts
                {
                    let batch_x = 0;
                    let batch_y = n as u32 * height_out as u32;
                    (x as u32 + batch_x, y as u32 + batch_y)
                } else {
                    (x as u32, y as u32)
                };

                let mut pixel = [0 as f32; 3];
                match opts.color_opts {
                    ColorDisplayOpts::Rgb => match channels {
                        1 => {
                            pixel[0] = channel_vals[0];
                            pixel[1] = 0.0;
                            pixel[2] = 0.0;
                        }
                        2 => {
                            pixel[0] = channel_vals[0];
                            pixel[1] = channel_vals[1];
                            pixel[2] = 0.0;
                        }
                        3 => {
                            pixel[0] = channel_vals[0];
                            pixel[1] = channel_vals[1];
                            pixel[2] = channel_vals[2];
                        }
                        _ => unimplemented!("More than 3 channels not supported ({channels})"),
                    },
                    ColorDisplayOpts::Monochrome { min, max } => {
                        let val: f32 = channel_vals.iter().sum();
                        pixel[0] = min[0] * (1.0 - val) + max[0] * val;
                        pixel[1] = min[1] * (1.0 - val) + max[1] * val;
                        pixel[2] = min[2] * (1.0 - val) + max[2] * val;
                    }
                }

                let pixel = [
                    (pixel[0] * 255.0) as u8,
                    (pixel[1] * 255.0) as u8,
                    (pixel[2] * 255.0) as u8,
                ];
                img.put_pixel(x, y, Rgb(pixel));
            }
        }
    }

    img.save(path)?;
    Ok(())
}

/// Normalize values in 2D tensor from 0 to 1
fn normalize<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let min = tensor.clone().min().to_data().to_vec::<f32>().unwrap()[0];
    let max = tensor.clone().max().to_data().to_vec::<f32>().unwrap()[0];
    let range = if max - min == 0.0 { 1.0 } else { max - min };

    tensor
        .sub_scalar(min.elem::<f32>())
        .div_scalar(range.elem::<f32>())
}
