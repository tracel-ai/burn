use crate::processor::{NodeProcessor, ProcessorContext};
use crate::{
    ElementType, TensorData,
    ir::{ArgType, Data, Node, NodeConfig, TensorType},
};
use std::any::Any;

/// Shape information for the Expand operation.
#[derive(Debug, Clone)]
pub enum ExpandShape {
    /// Static shape information known at compile time.
    Static(Vec<i64>),
    /// Runtime shape that will be determined during execution .
    Runtime(crate::ir::Argument),
}

impl NodeConfig for ExpandShape {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

pub struct ExpandProcessor;

impl NodeProcessor for ExpandProcessor {
    fn supported_opset_range(&self) -> (i64, Option<i64>) {
        (8, None)
    }

    fn process_config(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1, "Expand: shape tensor must be 1D");
                assert!(
                    matches!(tensor.elem_type, ElementType::Int64),
                    "Expand: shape tensor must have element type int64"
                );
            }
            ArgType::Shape(_) => {
                // Shapes are always 1-D int64 data, so nothing to assert here
            }
            _ => panic!("Only tensor input is valid for shape"),
        }

        let config = match node.inputs[1].into_value() {
            Some(TensorData {
                data: Data::Int64s(shape),
                ..
            }) => ExpandShape::Static(shape.clone()),
            None => {
                // we were unable to statically determine the input value, so we'll need to fetch it at runtime
                let mut runtime_arg = node.inputs[1].clone();
                runtime_arg.value_store = None;
                ExpandShape::Runtime(runtime_arg)
            }
            _ => panic!(
                "Shape data type must be int64, is {:?}",
                &node.inputs[1].into_value()
            ),
        };
        node.config = Some(Box::new(config));
    }

    fn process_forward(
        &self,
        node: &mut Node,
        _context: &ProcessorContext,
        graph_data: &mut crate::from_onnx::GraphData,
    ) {
        log::debug!("Expand node {} has {} inputs", node.name, node.inputs.len());
        if node.inputs.len() >= 2 {
            log::debug!(
                "Expand node {} input[0]: {:?}",
                node.name,
                node.inputs[0].ty
            );
            log::debug!(
                "Expand node {} input[1]: {:?}",
                node.name,
                node.inputs[1].ty
            );
        }

        let shape = if node.inputs.len() == 2 {
            match node.inputs[1].into_value() {
                Some(value) => match &value.data {
                    Data::Int64s(shape) => Some(shape.clone()),
                    _ => panic!("Expand operation encountered invalid input types"),
                },
                None => None,
            }
        } else {
            panic!("Expand operation requires exactly two inputs");
        };

        // Get input element type - Expand should preserve the input's element type
        let input_elem_type = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.elem_type.clone(),
            _ => panic!("Expand operation requires first input to be a tensor"),
        };

        let output = match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => panic!("Expand operation encountered invalid output types"),
        };

        if let Some(shape) = shape {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: input_elem_type.clone(),
                rank: shape.len(),
                static_shape: Some(shape.into_iter().map(|dim| dim as usize).collect()),
            });
        } else {
            // When the shape cannot be determined statically (i.e., the second argument 'shape' is passed dynamically),
            // infer the rank from the shape input.
            let output_rank = match &node.inputs[1].ty {
                ArgType::Shape(rank) => {
                    // Shape type directly gives us the output rank
                    *rank
                }
                ArgType::Tensor(tensor) => {
                    // For tensor inputs representing shapes, the rank should be 1
                    // and the output rank is determined by the number of elements
                    if tensor.rank == 1 {
                        // If we have a static shape, use it to get the exact output rank
                        if let Some(static_shape) = &tensor.static_shape {
                            static_shape
                                .first()
                                .copied()
                                .expect("Static shape must contain at least one element")
                        } else {
                            // For dynamic rank-1 tensors without static shape, we need to make an assumption
                            // or get the information from elsewhere.
                            // Check if we have a value that can tell us the rank
                            if let Some(value) = node.inputs[1].into_value() {
                                if let Data::Int64s(shape_data) = &value.data {
                                    // We have the actual shape values, so the output rank is the number of elements
                                    shape_data.len()
                                } else {
                                    panic!(
                                        "Expand shape tensor has unexpected data type: {:?}",
                                        value.data
                                    )
                                }
                            } else {
                                // No static shape and no value - this is truly dynamic
                                // We need to look at the output type if it's been set
                                log::warn!(
                                    "Expand node {} has dynamic shape tensor without static shape info. Using output rank if available.",
                                    node.name
                                );
                                // Use the current output rank if it's already a tensor
                                match &output {
                                    TensorType { rank, .. } if *rank > 0 => *rank,
                                    _ => panic!(
                                        "Cannot determine output rank for Expand node {} with fully dynamic shape tensor. Please provide static shape or use Shape type.",
                                        node.name
                                    ),
                                }
                            }
                        }
                    } else {
                        panic!(
                            "Shape tensor for Expand must be 1-dimensional, got rank {}",
                            tensor.rank
                        )
                    }
                }
                _ => panic!("Shape input must be of tensor or shape type"),
            };

            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: input_elem_type,
                rank: output_rank,
                static_shape: None, // The exact shape cannot be determined statically
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ElementType, NodeType, TensorData};
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(
        input_rank: usize,
        shape_value: Option<Vec<i64>>,
        shape_type: Option<ArgType>,
    ) -> NodeBuilder {
        let mut builder = NodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", input_rank, None)
            .output_tensor_f32("output", 0, None); // Rank 0 will be updated

        if let Some(shape) = shape_value {
            builder = builder.input_tensor_i64_data("shape", shape.clone(), vec![shape.len()]);
        } else if let Some(st) = shape_type {
            // Use the provided custom shape type
            builder = builder.add_input("shape", st);
        } else {
            // Default case with dynamic shape
            builder = builder.input_tensor_i64("shape", 1, Some(vec![3]));
        }

        builder
    }

    #[test]
    fn test_expand_with_constant_shape() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node =
            create_test_node(2, Some(vec![2, 3, 4]), None).build_with_graph_data(&mut graph_data);

        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_forward(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, Some(vec![2, 3, 4]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_with_dynamic_shape() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = create_test_node(2, None, None).build();

        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_forward(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, None);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    #[should_panic(expected = "Expand operation requires exactly two inputs")]
    fn test_expand_with_incorrect_inputs() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node =
            create_test_node(2, Some(vec![2, 3, 4]), None).build_with_graph_data(&mut graph_data);
        node.inputs.pop(); // Remove one input

        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_forward(&mut node, &context, &mut graph_data);
    }

    // Tests for expand_config function

    #[test]
    fn test_expand_config_with_static_shape() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node =
            create_test_node(2, Some(vec![2, 3, 4]), None).build_with_graph_data(&mut graph_data);
        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<ExpandShape>()
            .unwrap();

        match config {
            ExpandShape::Static(shape) => {
                assert_eq!(*shape, vec![2, 3, 4]);
            }
            ExpandShape::Runtime(_) => panic!("Expected Static config, got Runtime"),
        }
    }

    #[test]
    fn test_expand_config_with_runtime_shape() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(2, None, None).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<ExpandShape>()
            .unwrap();

        match config {
            ExpandShape::Static(_) => panic!("Expected Runtime config, got Static"),
            ExpandShape::Runtime(name) => {
                assert_eq!(name.name, "shape");
            }
        }
    }

    #[test]
    fn test_expand_config_with_shape_type() {
        let shape_type = ArgType::Shape(3);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(2, None, Some(shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
        let config = node
            .config
            .as_ref()
            .unwrap()
            .as_any()
            .downcast_ref::<ExpandShape>()
            .unwrap();

        match config {
            ExpandShape::Static(_) => panic!("Expected Runtime config, got Static"),
            ExpandShape::Runtime(name) => {
                assert_eq!(name.name, "shape");
            }
        }
    }

    #[test]
    #[should_panic(expected = "Expand: shape tensor must be 1D")]
    fn test_expand_config_with_invalid_shape_rank() {
        let invalid_shape_type = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int64,
            rank: 2, // Invalid rank, should be 1
            static_shape: None,
        });
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(2, None, Some(invalid_shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
    }

    #[test]
    #[should_panic(expected = "Expand: shape tensor must have element type int64")]
    fn test_expand_config_with_invalid_shape_type() {
        let invalid_shape_type = ArgType::Tensor(TensorType {
            elem_type: ElementType::Float32, // Invalid element type, should be Int64
            rank: 1,
            static_shape: None,
        });
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(2, None, Some(invalid_shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
    }

    #[test]
    #[should_panic(expected = "Only tensor input is valid for shape")]
    fn test_expand_config_with_invalid_input_type() {
        let invalid_shape_type = ArgType::Scalar(ElementType::Int64);
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let node = create_test_node(2, None, Some(invalid_shape_type)).build();
        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
    }

    #[test]
    #[should_panic(expected = "Expand: shape tensor must have element type int64")]
    fn test_expand_config_with_invalid_value_type() {
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);

        // Create a node with shape input that has Float32 type instead of Int64
        let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", 2, None)
            .input_tensor_f32_data("shape", vec![2.0, 3.0, 4.0], vec![3]) // Wrong type - Float32 instead of Int64
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(&mut graph_data);

        let mut node = node;
        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_config(&mut node, &context, &mut graph_data);
    }

    #[test]
    fn test_expand_update_outputs_with_shape_input() {
        // Test Expand with Shape type as shape input
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = create_test_node(2, None, Some(ArgType::Shape(4))).build();

        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_forward(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 4); // Shape(4) means output will be rank 4
                assert_eq!(tensor.static_shape, None); // Dynamic shape
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_update_outputs_with_shape_input_static_value() {
        // Test Expand with shape input that has static values
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_f32("input", 2, None)
            .input_tensor_i64_data("shape", vec![5, 10, 15], vec![3]) // Static shape values
            .output_tensor_f32("output", 0, None)
            .build_with_graph_data(&mut graph_data);

        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_forward(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 3);
                assert_eq!(tensor.static_shape, Some(vec![5, 10, 15]));
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_expand_preserves_input_element_type() {
        // Test that Expand preserves the input element type for different types

        // Test Float32 -> Float32
        {
            let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
            let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_f32("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_f32("output", 0, None)
                .build_with_graph_data(&mut graph_data);

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64, // Wrong type
                rank: 0,
                static_shape: None,
            });

            let processor = ExpandProcessor;
            let context = ProcessorContext::new(16);
            processor.process_forward(&mut node, &context, &mut graph_data);

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.elem_type,
                        ElementType::Float32,
                        "Expand should preserve Float32 input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }

        // Test Int64 -> Int64
        {
            let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
            let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_i64("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_i64("output", 0, None)
                .build_with_graph_data(&mut graph_data);

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32, // Wrong type
                rank: 0,
                static_shape: None,
            });

            let processor = ExpandProcessor;
            let context = ProcessorContext::new(16);
            processor.process_forward(&mut node, &context, &mut graph_data);

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.elem_type,
                        ElementType::Int64,
                        "Expand should preserve Int64 input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }

        // Test Bool -> Bool
        {
            let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
            let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
                .input_tensor_bool("input", 2, None)
                .input_tensor_i64_data("shape", vec![2, 3, 4], vec![3])
                .output_tensor_bool("output", 0, None)
                .build_with_graph_data(&mut graph_data);

            // Initially set output to wrong type
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32, // Wrong type
                rank: 0,
                static_shape: None,
            });

            let processor = ExpandProcessor;
            let context = ProcessorContext::new(16);
            processor.process_forward(&mut node, &context, &mut graph_data);

            match &node.outputs[0].ty {
                ArgType::Tensor(tensor) => {
                    assert_eq!(
                        tensor.elem_type,
                        ElementType::Bool,
                        "Expand should preserve Bool input type"
                    );
                    assert_eq!(tensor.rank, 3);
                }
                _ => panic!("Expected tensor output"),
            }
        }
    }

    #[test]
    fn test_expand_with_mismatched_output_type() {
        // Test that Expand corrects output type even when initially set incorrectly
        // This simulates the case where ONNX might have wrong type info
        let mut graph_data = crate::from_onnx::GraphData::new(&[], &[], &[]);
        let mut node = NodeBuilder::new(NodeType::Expand, "test_expand")
            .input_tensor_i64("input", 2, None) // Input is Int64
            .input_tensor_i64_data("shape", vec![2, 3], vec![2])
            .output_tensor_f32("output", 0, None) // Output incorrectly set to Float32
            .build_with_graph_data(&mut graph_data);

        let processor = ExpandProcessor;
        let context = ProcessorContext::new(16);
        processor.process_forward(&mut node, &context, &mut graph_data);

        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(
                    tensor.elem_type,
                    ElementType::Int64,
                    "Expand should use input type (Int64) not initial output type (Float32)"
                );
                assert_eq!(tensor.rank, 2);
                assert_eq!(tensor.static_shape, Some(vec![2, 3]));
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
