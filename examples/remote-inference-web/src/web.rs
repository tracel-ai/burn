//! Browser entry point: a MNIST classifier whose tensor operations run on a remote Iroh compute
//! peer. The model is defined client-side, but every op executes on the peer; only the input and
//! output probabilities cross the wire.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use wasm_bindgen::prelude::*;

use burn::backend::remote::{EndpointId, RemoteSecret};
use burn::module::Module;
use burn::store::ModuleRecord;
use burn::tensor::{Bytes, Device, Tensor, activation::softmax};
use iroh::{Endpoint, endpoint::presets};

use crate::model::Model;

/// Trained MNIST parameters in the burnpack format, produced by the `mnist` example.
static STATE_ENCODED: &[u8] = include_bytes!("../model.bpk");

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Derive the server's identity from the shared topic; both ends compute the same id from the same
/// string (a demo convenience — see the native example for the security note).
fn server_id(topic: &str) -> EndpointId {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    RemoteSecret::from_bytes(*hash.as_bytes()).id()
}

#[wasm_bindgen]
pub struct RemoteMnist {
    device: Device,
    model: Model,
}

#[wasm_bindgen]
impl RemoteMnist {
    /// Connect to the peer reachable under `topic` and load the model onto it.
    pub async fn connect(topic: String) -> Result<RemoteMnist, String> {
        console_error_panic_hook::set_once();

        let endpoint = Endpoint::builder(presets::N0)
            .bind()
            .await
            .map_err(|err| err.to_string())?;
        let device = Device::remote_iroh_async(&endpoint, server_id(&topic), 0).await;

        let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(STATE_ENCODED.to_vec()))
            .map_err(|err| format!("Failed to decode model weights: {err}"))?;
        let model = Model::new(&device).load_record(record);

        Ok(Self { device, model })
    }

    /// Classify a 28x28 grayscale image (row-major length-784 `f32`, pixels in `[0, 255]`),
    /// returning the 10 class probabilities.
    pub async fn inference(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let input = Tensor::<1>::from_floats(input, &self.device).reshape([1, 28, 28]);

        // MNIST training mean/std (from the PyTorch example).
        let input = ((input / 255) - 0.1307) / 0.3081;

        let output = softmax(self.model.forward(input), 1);
        let data = output
            .into_data_async()
            .await
            .map_err(|err| format!("Failed to read inference result: {err:?}"))?;

        Ok(data.iter::<f32>().collect())
    }
}
