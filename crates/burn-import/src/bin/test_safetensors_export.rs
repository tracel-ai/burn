use std::io::Write;
use std::{collections::HashMap, fs::File};

use burn::{
    backend::{libtorch::LibTorchDevice, LibTorch},
    module::{Module, ModuleVisitor, ParamId},
    nn::{Linear, LinearConfig},
    prelude::Backend,
    record::{
        serde::data::NestedValue, FullPrecisionSettings, PrecisionSettings, Record, Recorder,
    },
    tensor::{Tensor, TensorData},
};
use burn_import::safetensors::export::to_safetensors;

#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    l1: Linear<B>,
    l2: Linear<B>,
}

type Back = LibTorch;

fn main() {
    use serde::Serialize;
    let device = LibTorchDevice::Cpu;
    let model = Mlp {
        l1: LinearConfig::new(3, 2).init::<Back>(&device),
        l2: LinearConfig::new(7, 1).init::<Back>(&device),
    };

    let safetensors_serialization = to_safetensors(model).unwrap();

    let mut file = File::create("output.bin").unwrap();
    file.write_all(&safetensors_serialization).unwrap();
}
