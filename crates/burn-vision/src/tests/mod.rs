use std::path::PathBuf;

use burn_tensor::{Shape, Tensor, TensorData, backend::Backend};
use image::{DynamicImage, ImageBuffer, Luma, Rgb};

mod connected_components;
mod morphology;

#[macro_export]
macro_rules! testgen_all {
    () => {
        use burn_tensor::{Bool, Float, Int};

        pub type TestTensor<const D: usize> = burn_tensor::Tensor<TestBackend, D>;
        pub type TestTensorInt<const D: usize> = burn_tensor::Tensor<TestBackend, D, Int>;
        pub type TestTensorBool<const D: usize> = burn_tensor::Tensor<TestBackend, D, Bool>;

        pub mod vision {
            pub use super::*;

            pub type IntType = <TestBackend as burn_tensor::backend::Backend>::IntElem;

            burn_vision::testgen_connected_components!();
            burn_vision::testgen_morphology!();
        }
    };
}

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
