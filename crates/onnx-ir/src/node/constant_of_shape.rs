use crate::processor::{NodeProcessor, OutputPreferences, ProcessError};
use crate::{
    TensorData,
    ir::{ArgType, Data, ElementType, Node, NodeConfig, RuntimeInputRef, TensorType},
};
use std::any::Any;

/// Shape information for the ConstantOfShape operation.
#[derive(Debug, Clone)]
pub enum ConstantOfShapeShape {
    /// Static shape information known at compile time.
    Static(Vec<i64>),
    /// Runtime shape that will be determined during execution .
    Runtime(RuntimeInputRef),
}

impl NodeConfig for ConstantOfShapeShape {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ConstantOfShapeProcessor;

impl NodeProcessor for ConstantOfShapeProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        // The shape input can be lifted if it's a constant
        // This allows ConstantOfShape to use static shape values when available
        if !node.inputs.is_empty() {
            Ok(vec![node.inputs[0].name.clone()])
        } else {
            Ok(Vec::new())
        }
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        crate::util::validate_opset(opset, 9)?;
        crate::util::validate_input_count(node, 1)?;
        crate::util::validate_output_count(node, 1)?;

        // Validate input type
        match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => {
                // For tensor inputs representing shapes, the rank should be 1
                if tensor.rank != 1 {
                    return Err(ProcessError::Custom(
                        "ConstantOfShape: shape tensor must be 1D".to_string(),
                    ));
                }
                if !matches!(tensor.elem_type, ElementType::Int64) {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Int64".to_string(),
                        actual: format!("{:?}", tensor.elem_type),
                    });
                }
            }
            ArgType::Shape(_) => {
                // Shapes are always 1-D int64 data, so nothing to validate here
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        }

        log::debug!("ConstantOfShape rank inference for node {}", node.name);

        let value_type = node
            .attrs
            .get("value")
            .map(|v| v.clone().into_tensor().elem_type())
            .unwrap_or(ElementType::Float32); // If not given, defaults to 0 as float32
        log::debug!(
            "ConstantOfShape value type for {}: {:?}",
            node.name,
            value_type
        );

        let rank = match &node.inputs[0].ty {
            ArgType::Shape(rank) => {
                log::debug!(
                    "ConstantOfShape input is Shape with rank {} for {}",
                    rank,
                    node.name
                );
                *rank
            }
            ArgType::Tensor(tensor_type) => {
                log::debug!("ConstantOfShape input is Tensor for {}", node.name);

                // First check if we have a lifted constant value (most reliable)
                if let Some(tensor_data) = node.inputs[0].into_value() {
                    // We have the actual constant values that were lifted
                    log::debug!(
                        "ConstantOfShape extracting rank from lifted constant value for {}",
                        node.name
                    );

                    // The tensor data contains the shape values
                    // For a shape tensor, the length of the data is the output rank
                    match &tensor_data.data {
                        crate::ir::Data::Int64s(vals) => {
                            let r = vals.len();
                            log::debug!(
                                "ConstantOfShape derived rank from Int64s constant data: {} for {}",
                                r,
                                node.name
                            );
                            r
                        }
                        _ => {
                            return Err(ProcessError::Custom(format!(
                                "ConstantOfShape node {} requires Int64 shape input, found {:?}",
                                node.name, tensor_data.data
                            )));
                        }
                    }
                } else if let Some(shape) = &tensor_type.static_shape {
                    // Fall back to static shape if no constant value
                    let r = shape.first().copied().ok_or_else(|| {
                        ProcessError::Custom(
                            "ConstantOfShape node must have a non-empty static shape value"
                                .to_string(),
                        )
                    })?;
                    log::debug!(
                        "ConstantOfShape derived rank from static shape: {} for {}",
                        r,
                        node.name
                    );
                    r
                } else {
                    return Err(ProcessError::Custom(format!(
                        "ConstantOfShape node {} must have either a constant value or a static shape",
                        node.name
                    )));
                }
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // Update the input type to be a shape
        node.inputs[0].ty = ArgType::Shape(rank);
        log::debug!(
            "ConstantOfShape updated input to Shape({}) for {}",
            rank,
            node.name
        );

        // Optimization: When input is Shape(1) and value type is Int64,
        // output Shape(1) directly instead of a tensor. This is a common pattern
        // in ONNX models where ConstantOfShape is used to create shape arrays.
        // Downstream operations can cast to tensor if needed.
        // This optimization improves performance by keeping shape operations in the Shape domain.
        if rank == 1 && value_type == ElementType::Int64 {
            // Special optimization for Shape(1) with Int64 values
            node.outputs[0].ty = ArgType::Shape(1);
            log::debug!(
                "ConstantOfShape optimization: Shape(1) -> Shape(1) with Int64 value for {}",
                node.name
            );
        } else if rank == 0 {
            // When rank is 0, output should be a scalar
            node.outputs[0].ty = ArgType::Scalar(value_type);
            log::debug!("ConstantOfShape output is Scalar for {}", node.name);
        } else {
            // General case: output is a tensor
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: value_type,
                rank,
                static_shape: None,
            });
            log::debug!("ConstantOfShape output rank for {}: {}", node.name, rank);
        }

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Check if we have static values or need runtime resolution
        let config = match node.inputs[0].into_value() {
            Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) => ConstantOfShapeShape::Static(shape.clone()),
            None => {
                // Runtime input - store reference instead of cloning the argument
                ConstantOfShapeShape::Runtime(RuntimeInputRef::new(
                    node.inputs[0].name.clone(),
                    0,
                ))
            }
            _ => {
                return Err(ProcessError::Custom(format!(
                    "ConstantOfShape node {} requires Int64 shape data",
                    node.name
                )));
            }
        };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{RuntimeInputRef, AttributeValue, Data, NodeType, TensorData};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(input_ty: ArgType) -> NodeBuilder {
        NodeBuilder::new(NodeType::ConstantOfShape, "test_constantofshape")
            .add_input("shape", input_ty)
            .output_tensor_f32("output", 0, None) // Will be updated
    }

    #[test]
    fn test_shape_input() {
        let mut node = create_test_node(ArgType::Shape(3)).build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_tensor_input_with_static_shape() {
        let mut node = create_test_node(ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 1,
            static_shape: Some(vec![4]),
        }))
        .build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_custom_value_type() {
        let mut node = create_test_node(ArgType::Shape(2)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Int64s(vec![7]), // Int64 value
            }),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
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
    fn test_invalid_input_type() {
        let mut node = create_test_node(ArgType::Scalar(ElementType::Float32)).build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(result, Err(ProcessError::TypeMismatch { .. })));
    }

    #[test]
    fn test_no_static_shapes_with_value_attr() {
        // Simulates the scenario after constant lifting where the input has a value

        let mut node = NodeBuilder::new(NodeType::ConstantOfShape, "constantofshape1")
            .input_tensor_i64_data("constant180_out1", vec![2, 3, 4], vec![3])
            .output_default("/model/encoder/patch_encoder/ConstantOfShape_output_0")
            .attr_tensor(
                "value",
                TensorData {
                    data: Data::Int64s(vec![1]),
                    shape: vec![1],
                },
            )
            .build_with_graph_data(16);

        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        // Verify the output has the correct rank
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Int64);
                assert_eq!(tensor.rank, 3); // Output rank should be 3
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_scalar_output_with_shape_0() {
        // Test when input is Shape(0), output should be Scalar
        let mut node = create_test_node(ArgType::Shape(0)).build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_scalar_output_with_tensor_shape_0() {
        // Test when input is a tensor with static shape [0], output should be Scalar
        let mut node = create_test_node(ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 1,
            static_shape: Some(vec![0]), // Shape is [0], meaning rank-0 output
        }))
        .build();
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_scalar_output_with_custom_value() {
        // Test scalar output with custom value type
        let mut node = create_test_node(ArgType::Shape(0)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Int64s(vec![42]), // Custom Int64 value
            }),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Int64);
            }
            _ => panic!("Expected scalar output for rank 0 input"),
        }
    }

    #[test]
    fn test_shape_optimization_with_int64() {
        // Test Shape(1) -> Shape(1) optimization when value type is Int64
        let mut node = create_test_node(ArgType::Shape(1)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Int64s(vec![5]), // Int64 value
            }),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Shape(rank) => {
                assert_eq!(*rank, 1);
            }
            _ => panic!("Expected Shape(1) output for Shape(1) input with Int64 value"),
        }
    }

    #[test]
    fn test_shape_1_with_float_no_optimization() {
        // Test that Shape(1) with Float32 does NOT get optimized (outputs Tensor)
        let mut node = create_test_node(ArgType::Shape(1)).build();
        node.attrs.insert(
            "value".to_string(),
            AttributeValue::Tensor(TensorData {
                shape: vec![],
                data: Data::Float32s(vec![1.5]), // Float32 value
            }),
        );
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected Tensor output for Shape(1) input with Float32 value"),
        }
    }

    #[test]
    fn test_shape_1_default_value_no_optimization() {
        // Test that Shape(1) with default value (Float32) does NOT get optimized
        let mut node = create_test_node(ArgType::Shape(1)).build();
        // No value attribute means default Float32
        let processor = ConstantOfShapeProcessor;
        let prefs = OutputPreferences::new();
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 1);
            }
            _ => panic!("Expected Tensor output for Shape(1) input with default Float32 value"),
        }
    }
}
