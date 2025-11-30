//! ONNX to IR conversion phases
//!
//! This module contains the distinct phases of ONNX graph conversion:
//!
//! 1. **initialization** - Create GraphState, process initializers
//! 2. **node_conversion** - Convert ONNX nodes, extract constants
//! 3. **type_inference** - Infer types iteratively with preferences
//! 4. **post_processing** - Identity elimination, constant re-lifting
//! 5. **finalization** - Remove unused constants, build final graph
//!
//! All phase-specific helper functions are inlined into their respective phase modules.

pub(crate) mod finalization;
pub(crate) mod initialization;
pub(crate) mod node_conversion;
pub(crate) mod post_processing;
pub(crate) mod type_inference;
