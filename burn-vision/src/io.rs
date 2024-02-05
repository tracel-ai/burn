use std::fmt;
use std::{fmt::Display, io};
use std::path::Path; 
use std::fs::File;
use std::error::Error;

use burn_core::tensor::{backend::Backend, Tensor, Data, Shape};
use image::{DynamicImage, GenericImage, GenericImageView, Rgba};
use image::ColorType::Rgba8;

#[derive(Debug)]
pub enum ImageReaderError {
    Io(io::Error),
    Image(image::ImageError),
}

impl Display for ImageReaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ImageReaderError::Io(err) => write!(f, "IO error: {}", err),
            ImageReaderError::Image(err) => write!(f, "Image error: {}", err),
        }
    }
}

impl Error for ImageReaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ImageReaderError::Io(err) => Some(err),
            ImageReaderError::Image(err) => Some(err),
        }
    }
} 


pub struct ImageReader<B: Backend> {
    device: B::Device,
}

impl<B: Backend> ImageReader<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Tensor<B, 3>, ImageReaderError> {
        let img = image::open(path).map_err(ImageReaderError::Image)?;
        let (width, height) = img.dimensions();

        let raw_pixels: Vec<f32> = img.to_rgb8().into_iter().map(|p| {
            p.clone() as f32
        }).collect::<Vec<f32>>();

        let data = Data {
            value: raw_pixels,
            shape: Shape::new([height as usize, width as usize, 3]),
        };
        
        Ok(Tensor::<B, 3>::from_data(data.convert(), &self.device))
    }

    pub fn write_image<P: AsRef<Path>>(&self, tensor: &Tensor<B, 3>, path: P) -> Result<(), ImageReaderError> {
        let data = tensor.data().to_host();
        let shape = data.shape;
        let width = shape[1];
        let height = shape[0];

        let mut img = DynamicImage::new_rgb8(width as u32, height as u32);
        for y in 0..height {
            for x in 0..width {
                let r = data.value[(y * width + x) * 3 + 0] as u8;
                let g = data.value[(y * width + x) * 3 + 1] as u8;
                let b = data.value[(y * width + x) * 3 + 2] as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }

        let mut file = File::create(path).map_err(ImageReaderError::Io)?;
        img.write_to(&mut file, image::ImageOutputFormat::Png).map_err(ImageReaderError::Image)?;
        Ok(())
    }

}