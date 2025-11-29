//! # Constant
//!
//! Produces a constant tensor with fixed values.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Constant.html>
//!
//! ## Type Constraints
//!
//! - T: Any ONNX type
//!
//! ## Opset Versions
//!
//! - **Opset 1-8**: Basic constant with value attribute only
//! - **Opset 9-10**: Updated type constraints, same functionality
//! - **Opset 11-12**: Added sparse_value attribute for sparse tensor support
//! - **Opset 13+**: Added value_* attribute family (value_float, value_floats, value_int, value_ints, value_string, value_strings)

use onnx_ir_derive::NodeBuilder;

use crate::ir::{ArgType, Argument, Node, RawNode, TensorDataExt, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};

/// Node representation for Constant operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct ConstantNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
}

pub(crate) struct ConstantProcessor;

impl NodeProcessor for ConstantProcessor {
    type Config = ();

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            inputs: InputSpec::AtLeast(0),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        _opset: usize,
        output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Add test for empty tensor constant - edge case that may fail

        // Validate that the Constant node has an input
        if node.inputs.is_empty() {
            return Err(ProcessError::MissingAttribute(
                "Constant node must have an input with value data".to_string(),
            ));
        }

        // Get tensor data from central store via input's data_id
        let input = &node.inputs[0];
        let tensor_data = input.value().ok_or_else(|| {
            ProcessError::MissingAttribute("value (from central store)".to_string())
        })?;

        // First, determine the base type from the tensor data
        let base_type = if tensor_data.shape.is_empty() {
            ArgType::Scalar(tensor_data.elem_type())
        } else {
            ArgType::Tensor(TensorType {
                dtype: tensor_data.elem_type(),
                rank: tensor_data.shape.len(),
                static_shape: Some(tensor_data.shape.to_vec()),
            })
        };

        // Check output preferences to see if consumers want this converted
        let output_name = &node.outputs[0].name;
        let preferences = output_preferences.get(output_name);

        // Apply preferences if any exist
        node.outputs[0].ty = if !preferences.is_empty() {
            // Check if any consumer wants Shape type
            let wants_shape = preferences
                .iter()
                .any(|(_, ty)| matches!(ty, crate::processor::ArgPreference::Shape));

            // Check if any consumer wants Scalar type
            let wants_scalar = preferences
                .iter()
                .any(|(_, ty)| matches!(ty, crate::processor::ArgPreference::Scalar));

            match &base_type {
                // Convert 1D tensor to Shape if requested and we have static shape info
                ArgType::Tensor(tensor) if tensor.rank == 1 && wants_shape => {
                    if let Some(shape) = tensor.static_shape.as_ref() {
                        if let Some(&shape_rank) = shape.first() {
                            ArgType::Shape(shape_rank)
                        } else {
                            // Empty shape for rank-1 tensor is invalid, keep as Tensor
                            base_type
                        }
                    } else {
                        // No static shape info, keep as Tensor
                        base_type
                    }
                }
                // Convert scalar-compatible tensor to Scalar if requested
                ArgType::Tensor(tensor) if tensor.rank == 0 && wants_scalar => {
                    ArgType::Scalar(tensor.dtype)
                }
                // Otherwise keep base type
                _ => base_type,
            }
        } else {
            base_type
        };

        Ok(())
    }

    fn build_node(&self, builder: RawNode, _opset: usize) -> Node {
        Node::Constant(ConstantNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{DType, NodeType};
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node_with_data(tensor_data: crate::ir::TensorData) -> RawNode {
        use crate::graph_state::GraphState;
        use crate::ir::Argument;

        let elem_type = tensor_data.elem_type();
        let shape = tensor_data.shape.to_vec();

        // Create GraphState and register the constant
        let mut graph_state = GraphState::new(&[], &[], &[], &[]);
        graph_state.register_test_constant("test_value".to_string(), tensor_data);

        // Get the data_id from the registered constant
        let data_id = graph_state
            .get_constant_data_id("test_value")
            .expect("Test constant should have data_id");

        // Create type based on shape
        let ty = if shape.is_empty() {
            crate::ir::ArgType::Scalar(elem_type)
        } else {
            crate::ir::ArgType::Tensor(crate::ir::TensorType {
                dtype: elem_type,
                rank: shape.len(),
                static_shape: Some(shape),
            })
        };

        // Build ValueStore from GraphState
        let value_store = graph_state.build_value_store();

        // Create constant node with input containing the data_id
        let mut node = TestNodeBuilder::new(NodeType::Constant, "test_constant")
            .output_tensor_f32("output", 0, None)
            .build();

        // Create input with Static value
        let mut input_arg = Argument {
            name: String::new(),
            ty: ty.clone(),
            value_source: crate::ir::ValueSource::Static(data_id),
            value_store: None,
        };
        input_arg.set_value_store(value_store.clone());
        node.inputs.push(input_arg);

        // Attach value_store to output
        node.outputs[0].set_value_store(value_store);
        node.outputs[0].value_source = crate::ir::ValueSource::Constant;
        node.outputs[0].ty = ty;

        node
    }

    fn create_test_node() -> RawNode {
        // Create a node without data for testing missing value case
        TestNodeBuilder::new(NodeType::Constant, "test_constant")
            .output_tensor_f32("output", 0, None)
            .build()
    }

    #[test]
    fn test_constant_scalar_float() {
        let mut node =
            create_test_node_with_data(crate::ir::TensorData::new(vec![6.14f32], vec![]));

        let processor = ConstantProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::F32);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    fn test_constant_tensor() {
        let mut node = create_test_node_with_data(crate::ir::TensorData::new(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ));

        let processor = ConstantProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_constant_missing_value() {
        let mut node = create_test_node();

        let processor = ConstantProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::MissingAttribute { .. })));
    }

    #[test]
    fn test_constant_1d_tensor_to_shape_with_preferences() {
        let mut node = create_test_node_with_data(
            crate::ir::TensorData::new(vec![10i64, 20, 30], vec![3]), // 1D tensor with 3 elements
        );

        // Create preferences requesting Shape type
        let mut prefs = OutputPreferences::new();
        prefs.add(
            node.outputs[0].name.clone(),
            "consumer_node",
            crate::processor::ArgPreference::Shape,
        );

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should be converted to Shape(3) since we have 3 elements
        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 3);
            }
            other => panic!("Expected Shape output, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_1d_tensor_without_preferences() {
        let mut node =
            create_test_node_with_data(crate::ir::TensorData::new(vec![10i64, 20, 30], vec![3]));

        // No preferences
        let prefs = OutputPreferences::new();

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should remain as Tensor
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.dtype, DType::I64);
            }
            other => panic!("Expected Tensor output, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_rank0_tensor_to_scalar_with_preferences() {
        let mut node = create_test_node_with_data(
            crate::ir::TensorData::new(vec![42.0f32], vec![]), // rank 0 tensor
        );

        // Create preferences requesting Scalar type
        let mut prefs = OutputPreferences::new();
        prefs.add(
            node.outputs[0].name.clone(),
            "consumer_node",
            crate::processor::ArgPreference::Scalar,
        );

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should already be Scalar (rank 0 tensor is treated as scalar by default)
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::F32);
            }
            other => panic!("Expected Scalar output, got {:?}", other),
        }
    }

    #[test]
    fn test_constant_2d_tensor_ignores_shape_preference() {
        let mut node = create_test_node_with_data(
            crate::ir::TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]), // 2D tensor - cannot convert to Shape
        );

        // Create preferences requesting Shape type (which shouldn't apply to 2D tensor)
        let mut prefs = OutputPreferences::new();
        prefs.add(
            node.outputs[0].name.clone(),
            "consumer_node",
            crate::processor::ArgPreference::Shape,
        );

        let processor = ConstantProcessor;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Should remain as Tensor (2D tensor cannot be converted to Shape)
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.dtype, DType::F32);
            }
            other => panic!("Expected Tensor output, got {:?}", other),
        }
    }
}
