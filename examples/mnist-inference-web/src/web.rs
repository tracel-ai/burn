#![allow(clippy::new_without_default)]

use alloc::string::String;
use js_sys::Array;

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use crate::model::Model;
use crate::state::{build_and_load_model, Backend};

use burn::tensor::Tensor;

#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Mnist structure that corresponds to JavaScript class.
/// See:[exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Mnist {
    model: Option<Model<Backend>>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Mnist {
    /// Constructor called by JavaScripts with the new keyword.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self { model: None }
    }

    /// Returns the inference results.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 28x28 image
    ///
    /// See bindgen support types for passing and returning arrays:
    /// * [number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html)
    /// * [boxed-number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html)
    ///
    pub async fn inference(&mut self, input: &[f32]) -> Result<Array, String> {
        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }

        let model = self.model.as_ref().unwrap();

        let device = Default::default();
        // Reshape from the 1D array to 3d tensor [batch, height, width]
        let input: Tensor<Backend, 3> = Tensor::from_floats(input, &device).reshape([1, 28, 28]);

        // Normalize input: make between [0,1] and make the mean=0 and std=1
        // values mean=0.1307,std=0.3081 were copied from Pytorch Mist Example
        // https://github.com/pytorch/examples/blob/54f4572509891883a947411fd7239237dd2a39c3/mnist/main.py#L122

        let input = ((input / 255) - 0.1307) / 0.3081;

        // Run the tensor input through the model
        let output: Tensor<Backend, 2> = model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let output = burn::tensor::activation::softmax(output, 1);

        // Flatten output tensor with [1, 10] shape into boxed slice of [f32]
        #[cfg(not(target_family = "wasm"))]
        let output = output.into_data().convert::<f32>().value;

        #[cfg(target_family = "wasm")]
        let output = output.into_data().await.convert::<f32>().value;

        let array = Array::new();
        for value in output {
            array.push(&value.into());
        }

        Ok(array)
    }
}
