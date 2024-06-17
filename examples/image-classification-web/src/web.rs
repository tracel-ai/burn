#![allow(clippy::new_without_default)]

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::convert::Into;

use crate::model::{label::LABELS, normalizer::Normalizer, squeezenet::Model as SqueezenetModel};

use burn::{backend::NdArray, prelude::*, tensor::activation::softmax};

use burn_candle::Candle;
use burn_wgpu::{init_async, AutoGraphicsApi, Wgpu, WgpuDevice};

use serde::Serialize;
use wasm_bindgen::prelude::*;
use wasm_timer::Instant;

#[wasm_bindgen(start)]
pub fn start() {
    // Initialize the logger so that the logs are printed to the console
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
}

#[allow(clippy::large_enum_variant)]
/// The model is loaded to a specific backend
pub enum ModelType {
    /// The model is loaded to the Candle backend
    WithCandleBackend(Model<Candle<f32, i64>>),

    /// The model is loaded to the NdArray backend
    WithNdArrayBackend(Model<NdArray<f32>>),

    /// The model is loaded to the Wgpu backend
    WithWgpuBackend(Model<Wgpu<f32, i32>>),
}

/// The image is 224x224 pixels with 3 channels (RGB)
const HEIGHT: usize = 224;
const WIDTH: usize = 224;
const CHANNELS: usize = 3;

/// The image classifier
#[wasm_bindgen]
pub struct ImageClassifier {
    model: ModelType,
}

#[wasm_bindgen]
impl ImageClassifier {
    /// Constructor called by JavaScripts with the new keyword.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log::info!("Initializing the image classifier");
        let device = Default::default();
        Self {
            model: ModelType::WithNdArrayBackend(Model::new(&device)),
        }
    }

    /// Runs inference on the image
    pub async fn inference(&self, input: &[f32]) -> Result<JsValue, JsValue> {
        log::info!("Running inference on the image");

        let start = Instant::now();

        let result = match self.model {
            ModelType::WithCandleBackend(ref model) => model.forward(input).await,
            ModelType::WithNdArrayBackend(ref model) => model.forward(input).await,
            ModelType::WithWgpuBackend(ref model) => model.forward(input).await,
        };

        let duration = start.elapsed();

        log::debug!("Inference is completed in {:?}", duration);

        top_5_classes(result)
    }

    /// Sets the backend to Candle
    pub async fn set_backend_candle(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Candle backend");
        let start = Instant::now();
        let device = Default::default();
        self.model = ModelType::WithCandleBackend(Model::new(&device));
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Candle backend in {:?}", duration);
        Ok(())
    }

    /// Sets the backend to NdArray
    pub async fn set_backend_ndarray(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the NdArray backend");
        let start = Instant::now();
        let device = Default::default();
        self.model = ModelType::WithNdArrayBackend(Model::new(&device));
        let duration = start.elapsed();
        log::debug!("Model is loaded to the NdArray backend in {:?}", duration);
        Ok(())
    }

    /// Sets the backend to Wgpu
    pub async fn set_backend_wgpu(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Wgpu backend");
        let start = Instant::now();
        let device = WgpuDevice::default();
        init_async::<AutoGraphicsApi>(&device, Default::default()).await;
        self.model = ModelType::WithWgpuBackend(Model::new(&device));
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Wgpu backend in {:?}", duration);

        log::debug!("Warming up the model");
        let start = Instant::now();
        let _ = self.inference(&[0.0; HEIGHT * WIDTH * CHANNELS]).await;
        let duration = start.elapsed();
        log::debug!("Warming up is completed in {:?}", duration);
        Ok(())
    }
}

/// The image classifier model
pub struct Model<B: Backend> {
    model: SqueezenetModel<B>,
    normalizer: Normalizer<B>,
}

impl<B: Backend> Model<B> {
    /// Constructor
    pub fn new(device: &B::Device) -> Self {
        Self {
            model: SqueezenetModel::from_embedded(device),
            normalizer: Normalizer::new(device),
        }
    }

    /// Normalizes input and runs inference on the image
    pub async fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Reshape from the 1D array to 3d tensor [ width, height, channels]
        let input: Tensor<B, 4> =
            Tensor::from_floats(input, &B::Device::default()).reshape([1, CHANNELS, HEIGHT, WIDTH]);

        // Normalize input: make between [-1,1] and make the mean=0 and std=1
        let input = self.normalizer.normalize(input);

        // Run the tensor input through the model
        let output = self.model.forward(input);

        // Convert the model output into probability distribution using softmax formula
        let probabilities = softmax(output, 1);

        #[cfg(not(target_family = "wasm"))]
        let result = probabilities.into_data().convert::<f32>().value;

        // Forces the result to be computed
        #[cfg(target_family = "wasm")]
        let result = probabilities.into_data().await.convert::<f32>().value;

        result
    }
}

#[wasm_bindgen]
#[derive(Serialize)]
pub struct InferenceResult {
    index: usize,
    probability: f32,
    label: String,
}

/// Returns the top 5 classes and convert them into a JsValue
fn top_5_classes(probabilities: Vec<f32>) -> Result<JsValue, JsValue> {
    // Convert the probabilities into a vector of (index, probability)
    let mut probabilities: Vec<_> = probabilities.iter().enumerate().collect();

    // Sort the probabilities in descending order
    probabilities.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

    // Take the top 5 probabilities
    probabilities.truncate(5);

    // Convert the probabilities into InferenceResult
    let result: Vec<InferenceResult> = probabilities
        .into_iter()
        .map(|(index, probability)| InferenceResult {
            index,
            probability: *probability,
            label: LABELS[index].to_string(),
        })
        .collect();

    // Convert the InferenceResult into a JsValue
    Ok(serde_wasm_bindgen::to_value(&result)?)
}
