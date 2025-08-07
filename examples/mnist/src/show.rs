use burn::prelude::Backend;
use burn::tensor::{ElementConversion, Tensor};
use image::{GrayImage, Luma};
use std::fs;
use std::path::Path;

/// Save a tensor as an image
pub fn save_as_img<B: Backend>(
    tensor: &Tensor<B, 2>,
    width: u32,
    height: u32,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(path);

    // Ensure the output directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let tensor = normalize(tensor.clone());
    let data = tensor.to_data();
    let mut img = GrayImage::new(width, height);

    let shape = data.shape.clone();
    let data_vec = data.to_vec::<f32>().unwrap();

    for x in 0..width {
        for y in 0..height {
            let i = ((x as f32) / (width as f32) * (shape[0] as f32))
                .floor()
                .clamp(0.0, shape[0] as f32) as usize;
            let j = ((y as f32) / (height as f32) * (shape[1] as f32))
                .floor()
                .clamp(0.0, shape[1] as f32) as usize;
            let pixel = data_vec[i + j * shape[0]];
            img.put_pixel(x, y, Luma([pixel as u8]));
        }
    }

    img.save(path)?;
    Ok(())
}

/// Normalize values in 2D tensor from 0 to 255
fn normalize<B: Backend>(tensor: Tensor<B, 2>) -> Tensor<B, 2> {
    let min = tensor.clone().min().to_data().to_vec::<f32>().unwrap()[0];
    let max = tensor.clone().max().to_data().to_vec::<f32>().unwrap()[0];
    let range = if max - min == 0.0 { 1.0 } else { max - min };

    tensor
        .sub_scalar(min.elem::<f32>())
        .div_scalar(range.elem::<f32>())
        .mul_scalar(255.elem::<f32>())
}
