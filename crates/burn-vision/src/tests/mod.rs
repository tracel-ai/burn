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
