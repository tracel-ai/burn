use std::error::Error;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::{fmt::Display, io};

use burn_core::tensor::{backend::Backend, Data, Shape, Tensor};
use image::{DynamicImage, GenericImage, GenericImageView, Rgba};

#[derive(Debug)]
pub enum ImageReaderError {
    Io(io::Error),
    Image(image::ImageError),
    Custom(String),
}

impl Display for ImageReaderError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ImageReaderError::Io(err) => write!(f, "IO error: {}", err),
            ImageReaderError::Image(err) => write!(f, "Image error: {}", err),
            ImageReaderError::Custom(err) => write!(f, "Custom error: {}", err),
        }
    }
}

impl Error for ImageReaderError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ImageReaderError::Io(err) => Some(err),
            ImageReaderError::Image(err) => Some(err),
            ImageReaderError::Custom(_) => None,
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

    pub fn image_to_tensor(&self, img: DynamicImage) -> Result<Tensor<B, 3>, ImageReaderError> {
        let (width, height) = img.dimensions();

        let mut red_values = Vec::with_capacity((width * height) as usize);
        let mut green_values = Vec::with_capacity((width * height) as usize);
        let mut blue_values = Vec::with_capacity((width * height) as usize);

        for pixel in img.pixels() {
            let rgba = pixel.2;
            red_values.push(rgba[0] as f32);
            green_values.push(rgba[1] as f32);
            blue_values.push(rgba[2] as f32);
        }

        let raw_pixels: Vec<f32> = red_values
            .into_iter()
            .chain(green_values.into_iter())
            .chain(blue_values.into_iter())
            .collect();

        let data = Data {
            value: raw_pixels,
            shape: Shape::new([3, height as usize, width as usize]),
        };

        Ok(Tensor::<B, 3>::from_data(data.convert(), &self.device))
    }

    pub fn tensor_to_image(tensor: &Tensor<B, 3>) -> Result<DynamicImage, ImageReaderError> {
        let tensor = tensor.clone();
        let min = tensor.clone().min().into_scalar();

        // Bring the minimum value to zero.
        let tensor = tensor.sub_scalar(min);

        // Bring the max to 255
        let max = tensor.clone().max().into_scalar();
        let tensor = tensor.div_scalar(max).mul_scalar(255);

        let data = tensor.to_data().convert::<u8>();
        let shape = data.shape;
        let width = shape.dims[2];
        let height = shape.dims[1];
        let mut img = DynamicImage::new_rgb8(width as u32, height as u32);

        // slice the data into the three channels
        let red = data.value[0..(width * height) as usize].to_vec();
        let green = data.value[(width * height) as usize..(2 * width * height) as usize].to_vec();
        let blue =
            data.value[(2 * width * height) as usize..(3 * width * height) as usize].to_vec();

        for y in 0..height {
            for x in 0..width {
                let r = red[(y * width + x) as usize];
                let g = green[(y * width + x) as usize];
                let b = blue[(y * width + x) as usize];
                img.put_pixel(x as u32, y as u32, Rgba([r, g, b, 255]));
            }
        }

        Ok(img)
    }

    pub fn read_image<P: AsRef<Path>>(&self, path: P) -> Result<Tensor<B, 3>, ImageReaderError> {
        let img = image::open(path).map_err(ImageReaderError::Image)?;
        self.image_to_tensor(img)
    }

    pub fn write_image<P: AsRef<Path>>(
        &self,
        tensor: &Tensor<B, 3>,
        path: P,
    ) -> Result<(), ImageReaderError> {
        let img = Self::tensor_to_image(tensor)?;

        let mut file = File::create(path).map_err(ImageReaderError::Io)?;
        img.write_to(&mut file, image::ImageOutputFormat::Png)
            .map_err(ImageReaderError::Image)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_core::backend::wgpu::{Wgpu, WgpuDevice};
    use std::env;
    use std::path::PathBuf;

    // Test reading and writing an image using the ImageReader.
    #[test]
    fn test_image_io() {
        let device = WgpuDevice::default();
        let reader: ImageReader<Wgpu> = ImageReader::new(device.clone());

        let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        path.push("src");
        path.push("tests");
        path.push("helicopter.png");
        println!("Reading image from: {:?}", path);

        let tensor = reader.read_image(path).unwrap();
        let mut path = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        path.push("src");
        path.push("tests");
        path.push("helicopter_out.png");
        reader.write_image(&tensor, path).unwrap();
    }

    // Test writing just one channel by slicing the tensor.
    // We should see that channel 1 produces red, 2 produces green, and 3 produces blue.
    #[test]
    fn test_image_io_single_channel() {
        let device = WgpuDevice::default();
        let reader: ImageReader<Wgpu> = ImageReader::new(device.clone());

        // Create a 10x10 image with all red values as 0, green as 1, and blue as 2.
        let mut img = DynamicImage::new_rgb8(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                img.put_pixel(x, y, Rgba([0, 1, 2, 255]));
            }
        }

        let tensor = reader.image_to_tensor(img).unwrap();
        let red_slice = tensor.clone().slice([0..1, 0..10, 0..10]);
        let red_check = Tensor::<Wgpu, 3>::ones(Shape::new([1, 10, 10]), &device).mul_scalar(0);
        red_check
            .to_data()
            .assert_approx_eq(&red_slice.to_data(), 3);

        let green_slice = tensor.clone().slice([1..2, 0..10, 0..10]);
        let green_check = Tensor::<Wgpu, 3>::ones(Shape::new([1, 10, 10]), &device);
        green_check
            .to_data()
            .assert_approx_eq(&green_slice.to_data(), 3);

        let blue_slice = tensor.clone().slice([2..3, 0..10, 0..10]);
        let blue_check = Tensor::<Wgpu, 3>::ones(Shape::new([1, 10, 10]), &device).mul_scalar(2);
        blue_check
            .to_data()
            .assert_approx_eq(&blue_slice.to_data(), 3);
    }
}
