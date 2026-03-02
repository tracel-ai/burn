//! Example: Save and load a model with half-precision (F32 <-> F16)
//!
//! Demonstrates using HalfPrecisionAdapter to automatically convert between
//! F32 and F16 during saving/loading. The same adapter instance handles both
//! directions.
//!
//! Usage:
//!   cargo run -p burn-store --example half_precision

use burn_core as burn;
use burn_core::module::Module;
use burn_ndarray::NdArray;
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn_store::{BurnpackStore, HalfPrecisionAdapter, ModuleSnapshot};
use burn_tensor::backend::Backend;

// A model with mixed layer types to show selective conversion
#[derive(Module, Debug)]
struct DemoModel<B: Backend> {
    linear1: Linear<B>,
    norm: LayerNorm<B>,
    linear2: Linear<B>,
}

impl<B: Backend> DemoModel<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            linear1: LinearConfig::new(128, 64).init(device),
            norm: LayerNormConfig::new(64).init(device),
            linear2: LinearConfig::new(64, 10).init(device),
        }
    }
}

fn main() {
    type B = NdArray<f32>;
    let device = Default::default();
    let model = DemoModel::<B>::new(&device);

    // 1) Save at full F32 precision (baseline)
    let path_f32 = "/tmp/model_f32";
    let mut store = BurnpackStore::from_file(path_f32).overwrite(true);
    model.save_into(&mut store).expect("Failed to save F32");
    let size_f32 = std::fs::metadata(format!("{}.bpk", path_f32))
        .map(|m| m.len())
        .unwrap_or(0);

    // 2) Save with default half-precision (all default modules get F16)
    let path_f16 = "/tmp/model_f16";
    let adapter = HalfPrecisionAdapter::new();
    let mut store = BurnpackStore::from_file(path_f16)
        .overwrite(true)
        .with_to_adapter(adapter.clone());
    model.save_into(&mut store).expect("Failed to save F16");
    let size_f16 = std::fs::metadata(format!("{}.bpk", path_f16))
        .map(|m| m.len())
        .unwrap_or(0);

    // 3) Save with without_module: keep LayerNorm at F32
    let path_mixed = "/tmp/model_mixed";
    let adapter_no_norm = HalfPrecisionAdapter::new().without_module("LayerNorm");
    let mut store = BurnpackStore::from_file(path_mixed)
        .overwrite(true)
        .with_to_adapter(adapter_no_norm);
    model.save_into(&mut store).expect("Failed to save mixed");
    let size_mixed = std::fs::metadata(format!("{}.bpk", path_mixed))
        .map(|m| m.len())
        .unwrap_or(0);

    println!("F32 (full precision):    {} bytes", size_f32);
    println!("F16 (default modules):   {} bytes", size_f16);
    println!("Mixed (norm stays F32):  {} bytes", size_mixed);
    println!(
        "F16 savings: {:.1}%",
        (1.0 - size_f16 as f64 / size_f32 as f64) * 100.0
    );

    // 4) Round-trip: load the F16 file back to F32 with the same adapter
    let mut load_store = BurnpackStore::from_file(path_f16)
        .with_from_adapter(adapter)
        .validate(false);
    let mut model2 = DemoModel::<B>::new(&device);
    let result = model2.load_from(&mut load_store).expect("Failed to load");
    println!(
        "\nRound-trip loaded {} tensors successfully",
        result.applied.len()
    );
}
