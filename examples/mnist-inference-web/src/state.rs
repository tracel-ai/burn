use crate::model::Model;
use burn::{module::Module, prelude::Device, store::ModuleRecord, tensor::Bytes};

// Trained parameters in the burnpack format, produced by the `mnist` example
// (`model.into_record().save(..)`) and copied here. Regenerate with the same command if the
// model architecture changes.
static STATE_ENCODED: &[u8] = include_bytes!("../model.bpk");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model {
    #[cfg(all(feature = "flex", not(feature = "wgpu")))]
    let device = Device::flex();
    #[cfg(feature = "wgpu")]
    // Calls init_setup_async
    let device = Device::wgpu_async(Default::default()).await;

    let model = Model::new(&device);
    let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(STATE_ENCODED.to_vec()))
        .expect("Failed to decode state");

    model.load_record(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_decodes_into_architecture() {
        let device = Device::flex();
        let model = Model::new(&device);
        // `load_record` validates that every model parameter is present with a matching shape; a
        // stale/mismatched asset would panic here.
        let record = ModuleRecord::from_bytes(Bytes::from_bytes_vec(STATE_ENCODED.to_vec()))
            .expect("Embedded model.bpk should decode as burnpack");
        let _model = model.load_record(record);
    }
}
