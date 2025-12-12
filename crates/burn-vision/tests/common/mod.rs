use std::path::PathBuf;

use burn_tensor::{Shape, Tensor, TensorData, backend::Backend};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};

use burn_tensor::{Bool, Int};

#[cfg(all(
    any(feature = "test-cpu", feature = "ndarray"),
    not(any(feature = "test-wgpu", feature = "test-cuda"))
))]
pub type TestBackend = burn_ndarray::NdArray<f32, i32>;

#[cfg(all(test, feature = "test-wgpu"))]
pub type TestBackend = burn_wgpu::Wgpu;

#[cfg(all(test, feature = "test-cuda"))]
pub type TestBackend = burn_cuda::Cuda;

#[allow(unused)]
pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, Int>;
pub type TestTensorBool<const D: usize> = burn_tensor::Tensor<TestBackend, D, Bool>;

#[allow(unused)]
pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;

#[allow(missing_docs)]
#[macro_export]
macro_rules! as_type {
    ($ty:ident: [$($elem:tt),*]) => {
        [$($crate::as_type![$ty: $elem]),*]
    };
    ($ty:ident: [$($elem:tt,)*]) => {
        [$($crate::as_type![$ty: $elem]),*]
    };
    ($ty:ident: $elem:expr) => {
        {
            use cubecl::prelude::*;

            $ty::new($elem)
        }
    };
}

#[allow(unused)]
pub fn test_image<B: Backend>(name: &str, device: &B::Device, luma: bool) -> Tensor<B, 3> {
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
pub fn save_test_image<B: Backend>(name: &str, tensor: Tensor<B, 3>, luma: bool) {
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
