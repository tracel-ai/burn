use std::path::PathBuf;

use burn_core::tensor::{Device, Shape, TensorData};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};

use burn_core::tensor::{Bool, Int};

#[allow(unused)]
#[cfg(all(test, feature = "flex", not(any(feature = "wgpu", feature = "cuda"))))]
pub type TestDevice = burn_core::backend::FlexDevice;

#[cfg(all(test, feature = "wgpu"))]
pub type TestDevice = burn_core::backend::WgpuDevice;

#[cfg(all(test, feature = "cuda"))]
pub type TestDevice = burn_core::backend::CudaDevice;

pub use burn_core::tensor::Tensor;
pub type TestTensorInt<const D: usize> = Tensor<D, Int>;
#[allow(unused)]
pub type TestTensorBool<const D: usize> = Tensor<D, Bool>;

#[allow(unused)]
pub fn test_image(name: &str, device: &Device, luma: bool) -> Tensor<3> {
    let file = PathBuf::from("tests/images").join(name);
    let image = image::open(file).unwrap();
    if luma {
        let image = image.to_luma32f();
        let h = image.height() as usize;
        let w = image.width() as usize;
        let data = TensorData::new(image.into_vec(), Shape::new([h, w, 1]));
        Tensor::from_data(data, device)
    } else {
        let image = image.to_rgb32f();
        let h = image.height() as usize;
        let w = image.width() as usize;
        let data = TensorData::new(image.into_vec(), Shape::new([h, w, 3]));
        Tensor::from_data(data, device)
    }
}

#[allow(unused)]
pub fn save_test_image(name: &str, tensor: Tensor<3>, luma: bool) {
    let file = PathBuf::from("tests/images").join(name);
    let [h, w, _] = tensor.shape().dims();
    let data = tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap();
    if luma {
        let image = ImageBuffer::<Luma<f32>, _>::from_raw(w as u32, h as u32, data).unwrap();
        DynamicImage::from(image).to_luma8().save(file).unwrap();
    } else {
        let image = ImageBuffer::<Rgb<f32>, _>::from_raw(w as u32, h as u32, data).unwrap();
        DynamicImage::from(image).to_rgb8().save(file).unwrap();
    }
}
