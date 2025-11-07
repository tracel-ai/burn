//! Node processor trait and infrastructure for node-centric processing.
//!
//! This module defines the `NodeProcessor` trait with support for type preferences
//! and proper error handling.

use crate::ir::{Node, NodeConfig};
use std::collections::HashMap;

// Re-export registry types for backward compatibility
pub use crate::registry::get_processor_registry;

// ============================================================================
// Node Specification System
// ============================================================================

/// Specification for input count validation
#[derive(Debug, Clone)]
pub enum InputSpec {
    /// Exact count required
    Exact(usize),
    /// Minimum count required
    AtLeast(usize),
    /// Range of valid counts (min, max inclusive)
    Range(usize, usize),
    /// Different specs for different opset versions (opset_version, spec)
    OpsetDependent(Vec<(usize, InputSpec)>),
}

/// Specification for output count validation
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum OutputSpec {
    /// Exact count required
    Exact(usize),
    /// Range of valid counts (min, max inclusive)
    Range(usize, usize),
    /// Different specs for different opset versions (opset_version, spec)
    OpsetDependent(Vec<(usize, OutputSpec)>),
}

/// Specification for node validation and metadata
#[derive(Debug, Clone)]
pub struct NodeSpec {
    /// Minimum supported opset version
    pub min_opset: usize,
    /// Maximum supported opset version (None = no max)
    pub max_opset: Option<usize>,
    /// Input count validation
    pub inputs: InputSpec,
    /// Output count validation
    pub outputs: OutputSpec,
}

impl Default for NodeSpec {
    fn default() -> Self {
        Self {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(0),
            outputs: OutputSpec::Range(0, 2147483647),
        }
    }
}

/// Type preferences for node inputs
#[derive(Debug, Default, Clone)]
pub struct InputPreferences {
    preferences: HashMap<String, Vec<ArgPreference>>,
}

#[derive(Debug, Clone)]
pub enum ArgPreference {
    Scalar,
    Shape,
    Tensor,
}

impl InputPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(mut self, input_name: impl Into<String>, ty: ArgPreference) -> Self {
        self.preferences
            .entry(input_name.into())
            .or_default()
            .push(ty);
        self
    }

    pub fn get(&self, input_name: &str) -> &[ArgPreference] {
        self.preferences
            .get(input_name)
            .map_or(&[], |v| v.as_slice())
    }
}

/// Type preferences requested by consumers for a node's outputs
#[derive(Debug, Default, Clone)]
pub struct OutputPreferences {
    // output_name -> [(consumer_name, requested_type)]
    requests: HashMap<String, Vec<(String, ArgPreference)>>,
}

impl OutputPreferences {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        output_name: impl Into<String>,
        consumer: impl Into<String>,
        ty: ArgPreference,
    ) {
        self.requests
            .entry(output_name.into())
            .or_default()
            .push((consumer.into(), ty));
    }

    pub fn get(&self, output_name: &str) -> &[(String, ArgPreference)] {
        self.requests.get(output_name).map_or(&[], |v| v.as_slice())
    }
}

/// Errors that can occur during node processing
#[derive(Debug, Clone)]
pub enum ProcessError {
    UnsupportedOpset {
        required: usize,
        actual: usize,
    },
    MissingInput(String),
    MissingOutput(String),
    InvalidInputCount {
        expected: usize,
        actual: usize,
    },
    InvalidOutputCount {
        expected: usize,
        actual: usize,
    },
    TypeMismatch {
        expected: String,
        actual: String,
    },
    ConflictingPreferences {
        output: String,
        details: Vec<String>,
    },
    MissingAttribute(String),
    InvalidAttribute {
        name: String,
        reason: String,
    },
    Custom(String),
}

/// Node-specific processing logic for type inference and configuration extraction
pub trait NodeProcessor: Send + Sync {
    /// Return the node specification for validation
    ///
    /// This defines the supported opset versions, input/output counts, etc.
    /// Validation is automatically performed before processing.
    fn spec(&self) -> NodeSpec {
        NodeSpec::default()
    }

    /// Declare preferred types for inputs (propagated to producers as `output_preferences`)
    ///
    /// Preferences are requests, not requirements. Producers may honor them (e.g., Constant→Shape).
    fn input_preferences(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        Ok(None)
    }

    /// Convert constant inputs to static values (embedded in config, unreferenced constants removed later)
    fn lift_constants(&self, _node: &mut Node, _opset: usize) -> Result<(), ProcessError> {
        Ok(())
    }

    /// Infer output types given preferences from consumers
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError>;

    /// Extract config for codegen
    fn extract_config(
        &self,
        _node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        Ok(None)
    }
}

/// Default processor for nodes without specific implementations.
///
/// This processor passes through the input type to output.
pub(crate) struct DefaultProcessor;

impl NodeProcessor for DefaultProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Default: preserve input type
        same_as_input(node);
        Ok(())
    }
}

// ============================================================================
// Processor Utilities
// ============================================================================

/// Validate opset version
pub fn validate_opset(opset: usize, min_version: usize) -> Result<(), ProcessError> {
    if opset < min_version {
        Err(ProcessError::UnsupportedOpset {
            required: min_version,
            actual: opset,
        })
    } else {
        Ok(())
    }
}

// ============================================================================
// NodeSpec Validation
// ============================================================================

/// Validate node against its specification
pub fn validate_node_spec(node: &Node, opset: usize, spec: &NodeSpec) -> Result<(), ProcessError> {
    // Validate opset version
    if opset < spec.min_opset {
        return Err(ProcessError::UnsupportedOpset {
            required: spec.min_opset,
            actual: opset,
        });
    }
    if let Some(max) = spec.max_opset
        && opset > max
    {
        return Err(ProcessError::UnsupportedOpset {
            required: max,
            actual: opset,
        });
    }

    // Validate input count
    validate_input_spec(node, opset, &spec.inputs)?;

    // Validate output count
    validate_output_spec(node, opset, &spec.outputs)?;

    Ok(())
}

/// Validate input count against specification
fn validate_input_spec(node: &Node, opset: usize, spec: &InputSpec) -> Result<(), ProcessError> {
    let actual = node.inputs.len();

    match spec {
        InputSpec::Exact(expected) => {
            if actual != *expected {
                return Err(ProcessError::InvalidInputCount {
                    expected: *expected,
                    actual,
                });
            }
        }
        InputSpec::AtLeast(min) => {
            if actual < *min {
                return Err(ProcessError::InvalidInputCount {
                    expected: *min,
                    actual,
                });
            }
        }
        InputSpec::Range(min, max) => {
            if actual < *min || actual > *max {
                return Err(ProcessError::InvalidInputCount {
                    expected: *min, // Report minimum as expected
                    actual,
                });
            }
        }
        InputSpec::OpsetDependent(specs) => {
            // Find the applicable spec for this opset (use the highest opset <= current)
            let applicable_spec = specs
                .iter()
                .filter(|(opset_version, _)| *opset_version <= opset)
                .max_by_key(|(opset_version, _)| opset_version)
                .map(|(_, spec)| spec);

            if let Some(spec) = applicable_spec {
                validate_input_spec(node, opset, spec)?;
            }
        }
    }

    Ok(())
}

/// Validate output count against specification
fn validate_output_spec(node: &Node, opset: usize, spec: &OutputSpec) -> Result<(), ProcessError> {
    let actual = node.outputs.len();

    match spec {
        OutputSpec::Exact(expected) => {
            if actual != *expected {
                return Err(ProcessError::InvalidOutputCount {
                    expected: *expected,
                    actual,
                });
            }
        }
        OutputSpec::Range(min, max) => {
            if actual < *min || actual > *max {
                return Err(ProcessError::InvalidOutputCount {
                    expected: *min, // Report minimum as expected
                    actual,
                });
            }
        }
        OutputSpec::OpsetDependent(specs) => {
            // Find the applicable spec for this opset (use the highest opset <= current)
            let applicable_spec = specs
                .iter()
                .filter(|(opset_version, _)| *opset_version <= opset)
                .max_by_key(|(opset_version, _)| opset_version)
                .map(|(_, spec)| spec);

            if let Some(spec) = applicable_spec {
                validate_output_spec(node, opset, spec)?;
            }
        }
    }

    Ok(())
}

/// Copy input type to output (for operations that preserve type)
pub fn same_as_input(node: &mut Node) {
    node.outputs[0].ty = node.inputs[0].ty.clone();
}

/// Compute broadcast output rank from multiple inputs
pub fn compute_broadcast_rank(inputs: &[crate::ir::Argument]) -> usize {
    use crate::ir::ArgType;
    use core::cmp::max;

    inputs.iter().fold(0, |acc, input| match &input.ty {
        ArgType::Tensor(tensor) => max(acc, tensor.rank),
        ArgType::Scalar(_) => acc,
        ArgType::Shape(_) => max(acc, 1),
    })
}

/// Compute broadcast static shape from multiple inputs (NumPy-style broadcasting)
pub fn compute_broadcast_static_shape(inputs: &[crate::ir::Argument]) -> Option<Vec<usize>> {
    let static_shapes: Vec<_> = inputs
        .iter()
        .filter_map(|input| input.ty.static_shape().cloned())
        .collect();

    if static_shapes.is_empty() {
        return None;
    }

    if static_shapes.len() == 1 {
        return Some(static_shapes[0].clone());
    }

    if static_shapes.windows(2).all(|w| w[0] == w[1]) {
        return Some(static_shapes[0].clone());
    }

    let max_rank = static_shapes.iter().map(|s| s.len()).max()?;
    let mut result = vec![1; max_rank];

    for shape in &static_shapes {
        let offset = max_rank - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let result_idx = offset + i;
            let current_dim = result[result_idx];

            if current_dim == 1 {
                result[result_idx] = dim;
            } else if dim != 1 && dim != current_dim {
                return None;
            }
        }
    }

    Some(result)
}

/// Update output type for broadcasting operations to max input rank
pub fn same_as_input_broadcast(node: &mut Node) {
    use crate::ir::ArgType;

    let has_tensor_input = node
        .inputs
        .iter()
        .any(|input| matches!(&input.ty, ArgType::Tensor(_)));

    let has_shape_input = node
        .inputs
        .iter()
        .any(|input| matches!(&input.ty, ArgType::Shape(_)));

    if has_shape_input && !has_tensor_input {
        let shape_rank = node
            .inputs
            .iter()
            .find_map(|input| match &input.ty {
                ArgType::Shape(rank) => Some(*rank),
                _ => None,
            })
            .expect("Shape input must exist");

        node.outputs[0].ty = ArgType::Shape(shape_rank);
        return;
    }

    let max_rank = compute_broadcast_rank(&node.inputs);

    if max_rank == 0 {
        node.outputs[0].ty = ArgType::Scalar(node.inputs[0].ty.elem_type());
    } else {
        let elem_type = node
            .inputs
            .iter()
            .find_map(|input| match &input.ty {
                ArgType::Tensor(tensor) => Some(tensor.dtype),
                _ => None,
            })
            .unwrap_or_else(|| node.inputs[0].ty.elem_type());

        let static_shape = compute_broadcast_static_shape(&node.inputs);

        node.outputs[0].ty = ArgType::Tensor(crate::ir::TensorType {
            dtype: elem_type,
            rank: max_rank,
            static_shape,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ArgType, Argument, DType, Node, NodeType, TensorType};
    use crate::registry::ProcessorRegistry;

    struct TestProcessor;

    impl NodeProcessor for TestProcessor {
        fn infer_types(
            &self,
            node: &mut Node,
            _opset: usize,
            _output_preferences: &OutputPreferences,
        ) -> Result<(), ProcessError> {
            // Simple test: copy input type to output
            if !node.inputs.is_empty() && !node.outputs.is_empty() {
                node.outputs[0].ty = node.inputs[0].ty.clone();
            }
            Ok(())
        }
    }

    #[test]
    fn test_infer_outputs() {
        let processor = TestProcessor;
        let prefs = OutputPreferences::new();
        let mut node = Node {
            node_type: NodeType::Add,
            name: "test_node".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 2,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Output should match input type
        assert_eq!(node.outputs[0].ty, node.inputs[0].ty);
    }

    #[test]
    fn test_processor_registry() {
        let mut registry = ProcessorRegistry::new();

        // Register a processor
        registry.register(NodeType::Add, Box::new(TestProcessor));

        // Verify the processor is registered by checking the type
        let add_processor = registry.get(&NodeType::Add);
        let sub_processor = registry.get(&NodeType::Sub);

        // Add should return our TestProcessor (we can't directly check type, but can verify behavior)
        // Sub should return DefaultProcessor since it's not registered
        // Both should be valid processor references
        assert!(std::ptr::addr_of!(*add_processor) != std::ptr::addr_of!(*sub_processor));
    }

    #[test]
    fn test_default_processor() {
        let processor = DefaultProcessor;
        let prefs = OutputPreferences::new();
        let mut node = Node {
            node_type: NodeType::Relu,
            name: "test_relu".to_string(),
            inputs: vec![Argument {
                name: "input".to_string(),
                ty: ArgType::Tensor(TensorType {
                    dtype: DType::F32,
                    rank: 3,
                    static_shape: None,
                }),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            outputs: vec![Argument {
                name: "output".to_string(),
                ty: ArgType::default(),
                value_source: crate::ir::ValueSource::Dynamic,
                value_store: None,
            }],
            attrs: Default::default(),
            config: None,
        };

        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Default processor should preserve input type
        match &node.outputs[0].ty {
            ArgType::Tensor(t) => {
                assert_eq!(t.rank, 3);
                assert_eq!(t.dtype, DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
