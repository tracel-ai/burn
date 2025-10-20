//! # Cast
//!
//! Casts the elements of a given input tensor to a specified data type.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Cast.html>
//!
//! ## Attributes
//! - `to` (required): Target data type specified as an integer from the TensorProto DataType enum.
//!   Supported types include: bool, float16, float32, float64, int8, int16, int32, int64,
//!   uint8, uint16, uint32, uint64, string, and various float8 formats.
//! - `saturate` (optional, since opset 19): Defines conversion behavior when input value is out of
//!   range of destination type. Only applies for float8 conversions. Default: true.
//! - `round_mode` (optional, since opset 21): Rounding mode for float8e8m0 conversion.
//!   Values: "up" (default), "down", "nearest".
//!
//! ## Inputs
//! - `input` (T1): Input tensor to be cast. Supports heterogeneous types including all numeric types,
//!   bool, and string. Casting from complex types is not supported.
//!
//! ## Outputs
//! - `output` (T2): Output tensor with the same shape as input but with the target data type.
//!   Supports casting to all numeric types, bool, and string. Casting to complex types is not supported.
//!
//! ## Opset Versions
//! - **Opset 21+**: Added `round_mode` attribute for float8e8m0 conversion
//! - **Opset 19+**: Added `saturate` attribute for float8 conversions
//! - **Opset 13+**: Added support for bfloat16
//! - **Opset 9+**: Added float8 types (e4m3fn, e4m3fnuz, e5m2, e5m2fnuz)
//! - **Opset 6+**: Extended type support
//! - **Opset 1+**: Basic cast operation
//!
//! ## Special Features
//! - Supports casting from string tensor in plain (e.g., "3.14", "1000") and scientific notation
//!   (e.g., "1e-5", "1E8") to float types.
//! - The 'to' argument must match one of the data types in the TensorProto DataType enum.

use crate::ir::{ArgType, AttributeValue, ElementType, Node, NodeConfig, TensorType};
use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::proto_conversion::element_type_from_proto;
use std::any::Any;

/// Configuration for Cast operations
#[derive(Debug, Clone)]
pub struct CastConfig {
    /// Target element type to cast to
    pub to: ElementType,
}

impl CastConfig {
    /// Create a new CastConfig
    pub fn new(to: ElementType) -> Self {
        Self { to }
    }
}

impl NodeConfig for CastConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct CastProcessor;

impl NodeProcessor for CastProcessor {
    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset
        crate::processor::validate_opset(opset, 1)?;

        // Validate input count
        crate::processor::validate_input_count(node, 1)?;

        // Validate output count
        crate::processor::validate_output_count(node, 1)?;

        // Get reference to config for type inference
        let config = node.config::<CastConfig>();
        let elem_type = config.to.clone();

        // Infer output type based on input type
        let input = &mut node.inputs[0];
        let output = &mut node.outputs[0];

        match input.ty.clone() {
            ArgType::Tensor(tensor) => {
                if tensor.rank == 0 {
                    // treat 0-dim tensor as scalar
                    output.ty = ArgType::Scalar(elem_type);
                    input.ty = ArgType::Scalar(tensor.elem_type);
                } else {
                    // Cast input and output are the same shape, but possibly different types
                    output.ty = ArgType::Tensor(TensorType {
                        elem_type,
                        rank: tensor.rank,
                        static_shape: tensor.static_shape, // keep it
                    });
                }
            }
            ArgType::Scalar(_) => output.ty = ArgType::Scalar(elem_type),
            ArgType::Shape(rank) => {
                // When casting Shape to float or bool types, convert to 1D tensor
                // This allows Shape values to be used in tensor operations
                match elem_type {
                    ElementType::Float32
                    | ElementType::Float64
                    | ElementType::Float16
                    | ElementType::Bool => {
                        output.ty = ArgType::Tensor(TensorType {
                            elem_type: elem_type.clone(),
                            rank: 1,
                            static_shape: Some(vec![rank]),
                        });
                    }
                    _ => {
                        // For int types, keep as Shape
                        // This matches Burn's representation where shapes are always [i64; N]
                        output.ty = ArgType::Shape(rank);
                    }
                }
            }
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract the target element type from attributes
        let elem_type = match node.attrs.get("to") {
            Some(AttributeValue::Int64(type_id)) => element_type_from_proto(*type_id as i32)
                .map_err(|_| ProcessError::InvalidAttribute {
                    name: "to".to_string(),
                    reason: format!("unsupported dtype: {}", type_id),
                })?,
            Some(_) => {
                return Err(ProcessError::InvalidAttribute {
                    name: "to".to_string(),
                    reason: "must be Int64".to_string(),
                });
            }
            None => {
                return Err(ProcessError::MissingAttribute("to".to_string()));
            }
        };

        let config = CastConfig::new(elem_type);
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Argument, NodeType, TensorType};
    use crate::node::test_utils::NodeBuilder;
    use crate::protos::tensor_proto::DataType;
    use protobuf::Enum;
    fn create_test_node(input_rank: usize, to_type: i64) -> Node {
        NodeBuilder::new(NodeType::Cast, "test_cast")
            .input_tensor_f32("X", input_rank, None)
            .output_tensor_f32("Y", input_rank, None) // Element type will be overwritten
            .attr_int("to", to_type)
            .build()
    }

    // Additional test function to demonstrate scalar inputs
    fn create_scalar_test_node(to_type: i64) -> Node {
        NodeBuilder::new(NodeType::Cast, "test_cast")
            .input_scalar_f32("X")
            .output_scalar_f32("Y") // Element type will be overwritten
            .attr_int("to", to_type)
            .build()
    }

    #[test]
    fn test_cast_config() {
        let mut node = create_test_node(2, DataType::INT64.value() as i64);

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<CastConfig>();
        assert_eq!(config.to, ElementType::Int64);

        let mut node = create_test_node(2, DataType::FLOAT.value() as i64);

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<CastConfig>();
        assert_eq!(config.to, ElementType::Float32);

        let mut node = create_test_node(2, DataType::BOOL.value() as i64);

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<CastConfig>();
        assert_eq!(config.to, ElementType::Bool);
    }

    #[test]
    fn test_cast_float_to_int64() {
        let mut node = create_test_node(2, DataType::INT64.value() as i64);

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_cast_scalar_handling() {
        let mut node = create_test_node(0, DataType::BOOL.value() as i64);

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Bool);
            }
            _ => panic!("Expected scalar output for 0-rank tensor"),
        }

        match &node.inputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Input should have been converted to scalar"),
        }
    }

    #[test]
    fn test_cast_multiple_inputs() {
        let mut node = create_test_node(2, DataType::INT64.value() as i64);
        node.inputs.push(Argument {
            name: "extra".to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 1,
                static_shape: None,
            }),
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn test_cast_scalar_to_bool() {
        let mut node = create_scalar_test_node(DataType::BOOL.value() as i64);

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Bool);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    fn test_cast_shape_to_float32() {
        let mut node = NodeBuilder::new(NodeType::Cast, "test_cast")
            .input_shape("shape_input", 3)
            .output_shape("output", 3) // Will be overwritten
            .attr_int("to", DataType::FLOAT.value() as i64)
            .build();

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.static_shape, Some(vec![3]));
            }
            _ => panic!("Expected rank-1 tensor output when casting Shape to float"),
        }
    }

    #[test]
    fn test_cast_shape_to_int64_remains_shape() {
        let mut node = NodeBuilder::new(NodeType::Cast, "test_cast")
            .input_shape("shape_input", 4)
            .output_shape("output", 4) // Will be preserved
            .attr_int("to", DataType::INT64.value() as i64)
            .build();

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 4);
            }
            _ => panic!("Expected Shape output when casting Shape to int64"),
        }
    }

    #[test]
    fn test_cast_shape_to_bool() {
        let mut node = NodeBuilder::new(NodeType::Cast, "test_cast")
            .input_shape("shape_input", 3)
            .output_shape("output", 3) // Will be overwritten
            .attr_int("to", DataType::BOOL.value() as i64)
            .build();

        let processor = CastProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Bool);
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.static_shape, Some(vec![3]));
            }
            _ => panic!("Expected rank-1 bool tensor output when casting Shape to bool"),
        }
    }
}
