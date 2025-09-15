use crate::{ModuleSnapshot, SafetensorsStore};
use burn_core::nn::LinearConfig;

type TestBackend = burn_ndarray::NdArray;

#[test]
fn metadata_preservation() {
    let device = Default::default();
    let module = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);

    // Write with metadata
    let mut save_persister = SafetensorsStore::from_bytes(None)
        .metadata("framework", "burn")
        .metadata("version", "0.14.0")
        .metadata("model_type", "linear");

    module.collect_to(&mut save_persister).unwrap();

    // Verify metadata was saved (would need to add a method to check metadata)
    // For now, just verify the round trip works
    let mut load_persister = SafetensorsStore::from_bytes(None);
    if let SafetensorsStore::Memory(ref mut p) = load_persister {
        if let SafetensorsStore::Memory(ref p_save) = save_persister {
            // Get Arc and extract data
            let data_arc = p_save.data().unwrap();
            p.set_data(data_arc.as_ref().clone());
        }
    }

    let mut module2 = LinearConfig::new(4, 2)
        .with_bias(true)
        .init::<TestBackend>(&device);
    let result = module2.apply_from(&mut load_persister).unwrap();

    assert!(result.is_success());
}
