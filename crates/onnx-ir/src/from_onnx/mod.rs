//! ONNX to IR conversion
//!
//! This module handles the conversion of ONNX protobuf models into the internal
//! intermediate representation (IR) used by the burn framework.

mod constant_builder;
mod conversion;
mod graph_builder;
mod graph_data;
mod identity_elimination;
mod tensor_store;
mod type_inference;

// Public exports
pub use conversion::{convert_constant_value, element_type_from_proto};
pub use graph_builder::parse_onnx;

// Internal exports - used by other modules in onnx-ir
pub(crate) use graph_data::GraphData;

// Processor registry singleton
use crate::processor::ProcessorRegistry;
use std::sync::OnceLock;

static PROCESSOR_REGISTRY: OnceLock<ProcessorRegistry> = OnceLock::new();

/// Get the processor registry singleton
pub(crate) fn get_processor_registry() -> &'static ProcessorRegistry {
    PROCESSOR_REGISTRY.get_or_init(ProcessorRegistry::with_standard_processors)
}
