//! ONNX to IR conversion
//!
//! This module handles the conversion of ONNX protobuf models into the internal
//! intermediate representation (IR) used by the burn framework.
//!
//! ## Pipeline Overview
//!
//! The conversion happens in clear phases (see `pipeline.rs`):
//! 1. Initialization - Create state, process initializers
//! 2. Node Conversion - Convert ONNX nodes to IR
//! 3. Type Inference - Infer types iteratively
//! 4. Post-processing - Identity elimination
//! 5. Finalization - Cleanup and build graph
//!
//! ## Module Organization
//!
//! - `pipeline.rs` - Main orchestrator showing the entire flow
//! - `phases/` - All conversion phases and their helpers
//! - `graph_state.rs` - Mutable state container
//! - `conversion.rs` - Shared conversion utilities (public API)
//! - `tensor_store.rs` - Tensor data storage infrastructure

mod conversion;
mod graph_state;
mod phases;
mod pipeline;
mod tensor_store;

// Public exports
pub use conversion::{convert_constant_value, element_type_from_proto};
pub use pipeline::parse_onnx;

// Internal exports - used by other modules in onnx-ir
pub(crate) use graph_state::GraphState;

// Processor registry singleton
use crate::processor::ProcessorRegistry;
use std::sync::OnceLock;

static PROCESSOR_REGISTRY: OnceLock<ProcessorRegistry> = OnceLock::new();

/// Get the processor registry singleton
pub(crate) fn get_processor_registry() -> &'static ProcessorRegistry {
    PROCESSOR_REGISTRY.get_or_init(ProcessorRegistry::with_standard_processors)
}
