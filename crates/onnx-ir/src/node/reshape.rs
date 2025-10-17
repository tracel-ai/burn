use crate::ir::{
    ArgType, Argument, Data, Node, NodeConfig, RuntimeInputRef, TensorData, TensorType,
};
use crate::processor::{InputPreferences, NodeProcessor, OutputPreferences, ProcessError};
use std::any::Any;

/// Configuration for the Reshape operation.
#[derive(Debug, Clone)]
pub struct ReshapeConfig {
    pub shape: ReshapeInput,
}

impl NodeConfig for ReshapeConfig {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn NodeConfig> {
        Box::new(self.clone())
    }
}

/// Represents either a static value or a runtime argument for reshape shape.
#[derive(Debug, Clone)]
pub enum ReshapeInput {
    /// Static shape known at compile time.
    Static(Vec<i64>),
    /// Runtime shape determined during execution - references node.inputs[input_index].
    Runtime(RuntimeInputRef),
}

/// Update output rank for Reshape based on shape input if constant, otherwise use input rank.
pub fn reshape_update_outputs(node: &mut Node) {
    // Extract input information
    let input_info = extract_input_info(&node.inputs[0]);

    // Determine output rank
    let output_rank = infer_reshape_output_rank(node);

    // Get static shape if available
    let static_shape = match &node.outputs[0].ty {
        ArgType::Tensor(t) => t.static_shape.clone(),
        _ => None,
    };

    // Set output type based on rank and input type
    node.outputs[0].ty = determine_output_type(&input_info, output_rank, static_shape, node);
}

/// Extract relevant information from input argument
struct InputInfo {
    elem_type: crate::ElementType,
    is_shape: bool,
    shape_size: Option<usize>,
}

fn extract_input_info(input: &Argument) -> InputInfo {
    match &input.ty {
        ArgType::Tensor(tensor) => InputInfo {
            elem_type: tensor.elem_type.clone(),
            is_shape: false,
            shape_size: None,
        },
        ArgType::Shape(size) => InputInfo {
            elem_type: crate::ElementType::Int64,
            is_shape: true,
            shape_size: Some(*size),
        },
        _ => panic!(
            "Reshape: invalid input type - expected Tensor or Shape, got {:?}",
            input.ty
        ),
    }
}

/// Determine the output type based on input and output characteristics
fn determine_output_type(
    input_info: &InputInfo,
    output_rank: usize,
    static_shape: Option<Vec<usize>>,
    node: &Node,
) -> ArgType {
    // Case 1: Scalar output (rank 0)
    if output_rank == 0 {
        log::debug!("Reshape node {} outputs a scalar", node.name);
        return ArgType::Scalar(input_info.elem_type.clone());
    }

    // Case 2: Shape input -> Shape output (optimization)
    if input_info.is_shape && output_rank == 1 && input_info.elem_type == crate::ElementType::Int64
    {
        let output_size =
            calculate_shape_output_size(input_info.shape_size.unwrap_or(1), node, &static_shape);

        log::debug!(
            "Reshape node {} with Shape({}) input outputs Shape({})",
            node.name,
            input_info.shape_size.unwrap_or(1),
            output_size
        );
        return ArgType::Shape(output_size);
    }

    // Case 3: Regular tensor output
    ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape,
        elem_type: input_info.elem_type.clone(),
    })
}

/// Calculate the output size for Shape type outputs
fn calculate_shape_output_size(
    input_size: usize,
    node: &Node,
    static_shape: &Option<Vec<usize>>,
) -> usize {
    // Try to get size from static reshape parameter
    if let Some(shape_values) = get_static_shape(node)
        && shape_values.len() == 1
    {
        return match shape_values[0] {
            -1 => input_size, // Infer dimension
            n if n > 0 => n as usize,
            _ => 1, // Invalid value, default to 1
        };
    }

    // Try to get size from output's static shape
    if let Some(shape) = static_shape
        && shape.len() == 1
    {
        return shape[0];
    }

    // Default: preserve input size
    input_size
}

/// Infer output rank for reshape operation from available information
fn infer_reshape_output_rank(node: &Node) -> usize {
    // Try sources in order of preference

    // 1. Static shape from constant shape input
    if let Some(shape) = get_static_shape(node) {
        return shape.len();
    }

    // 2. Dynamic shape from shape input type
    if let Some(rank) = get_rank_from_shape_input(node) {
        return rank;
    }

    // 3. Output's static shape if available
    if let Some(rank) = get_rank_from_output(node) {
        return rank;
    }

    // No rank information available
    panic!(
        "Reshape node {} has dynamic shape with no rank information available. \
         Cannot determine output rank.",
        node.name
    )
}

/// Get rank from shape input if available
fn get_rank_from_shape_input(node: &Node) -> Option<usize> {
    if node.inputs.len() != 2 {
        return None;
    }

    match &node.inputs[1].ty {
        ArgType::Shape(rank) => Some(*rank),
        ArgType::Tensor(tensor) => tensor
            .static_shape
            .as_ref()
            .filter(|dims| !dims.is_empty())
            .map(|dims| dims[0]),
        _ => None,
    }
}

/// Get rank from output tensor if available
fn get_rank_from_output(node: &Node) -> Option<usize> {
    match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => tensor.static_shape.as_ref().map(|shape| shape.len()),
        _ => None,
    }
}

/// Extract static shape from reshape node if available
fn get_static_shape(node: &Node) -> Option<Vec<i64>> {
    // Check shape input
    if node.inputs.len() == 2
        && let Some(value) = node.inputs[1].into_value()
        && let Data::Int64s(shape) = &value.data
    {
        return Some(shape.clone());
    }

    None
}

/// Extract shape input as either static or runtime
fn extract_shape_input(node: &Node) -> ReshapeInput {
    match &node.inputs[1].ty {
        ArgType::Tensor(_) => extract_tensor_shape(node),
        ArgType::Shape(_) => {
            // Runtime input - store reference instead of cloning the argument
            ReshapeInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
        }
        _ => panic!("Reshape: second input must be either a Tensor or Shape type"),
    }
}

/// Extract shape from tensor input
fn extract_tensor_shape(node: &Node) -> ReshapeInput {
    match node.inputs[1].into_value() {
        Some(TensorData { data, shape, .. }) => {
            assert_eq!(shape.len(), 1, "Reshape: shape tensor must be 1D");
            ReshapeInput::Static(data.clone().into_i64s())
        }
        None => {
            // Runtime input - store reference instead of cloning the argument
            ReshapeInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
        }
    }
}

/// Node processor for Reshape operation
pub struct ReshapeProcessor;

impl NodeProcessor for ReshapeProcessor {
    fn lift_constants(&self, node: &mut Node, _opset: usize) -> Result<Vec<String>, ProcessError> {
        let mut lifted = Vec::new();

        // Lift shape input (input[1]) if present
        // Note: This might be a runtime argument, but we lift it anyway
        // The extract_config method will handle whether it's static or runtime
        if node.inputs.len() > 1 {
            lifted.push(node.inputs[1].name.clone());
        }

        Ok(lifted)
    }

    fn input_preferences(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<InputPreferences>, ProcessError> {
        use crate::processor::ArgPreference;

        if node.inputs.len() != 2 {
            return Ok(None);
        }

        // Prefer Shape type for shape input (second input)
        Ok(Some(
            InputPreferences::new().add(&node.inputs[1].name, ArgPreference::Shape),
        ))
    }

    fn infer_types(
        &self,
        node: &mut Node,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate opset, input count, and output count
        crate::util::validate_opset(opset, 5)?;
        crate::util::validate_input_count(node, 2)?;
        crate::util::validate_output_count(node, 1)?;

        // Extract and validate shape input
        let shape = extract_shape_input(node);
        let config = ReshapeConfig { shape };
        node.config = Some(Box::new(config));

        // Extract input information
        let input_info = extract_input_info(&node.inputs[0]);

        // Determine output rank
        let output_rank = infer_reshape_output_rank(node);

        // Get static shape if available
        let static_shape = match &node.outputs[0].ty {
            ArgType::Tensor(t) => t.static_shape.clone(),
            _ => None,
        };

        // Set output type
        node.outputs[0].ty = determine_output_type(&input_info, output_rank, static_shape, node);

        Ok(())
    }

    fn extract_config(
        &self,
        node: &Node,
        _opset: usize,
    ) -> Result<Option<Box<dyn NodeConfig>>, ProcessError> {
        // Extract shape input as either static or runtime
        let shape = match &node.inputs[1].ty {
            ArgType::Tensor(_tensor) => {
                // Extract shape from tensor input
                // Note: We don't validate rank here because extract_config runs before type inference
                // The rank might be 0 initially and will be updated during type inference
                match node.inputs[1].into_value() {
                    Some(TensorData { data, shape, .. }) => {
                        // Only validate when we have actual tensor data
                        assert_eq!(shape.len(), 1, "Reshape: shape tensor must be 1D");
                        ReshapeInput::Static(data.clone().into_i64s())
                    }
                    None => {
                        // Runtime input - store reference instead of cloning the argument
                        ReshapeInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
                    }
                }
            }
            ArgType::Shape(_) => {
                // Runtime input - store reference instead of cloning the argument
                ReshapeInput::Runtime(RuntimeInputRef::new(node.inputs[1].name.clone(), 1))
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        let config = ReshapeConfig { shape };
        Ok(Some(Box::new(config)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;
    use crate::ir::NodeType;
    use crate::node::test_utils::NodeBuilder;

    fn create_test_node(allowzero: i64, shape_vec: Vec<i64>) -> NodeBuilder {
        let mut builder = NodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_i64_data("shape", shape_vec.clone(), vec![shape_vec.len()])
            .output_tensor_f32("reshaped", 2, None);

        if allowzero != 0 {
            builder = builder.attr_int("allowzero", allowzero);
        }

        builder
    }

    fn create_runtime_reshape_node() -> NodeBuilder {
        NodeBuilder::new(NodeType::Reshape, "test_runtime_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("shape", 0, None) // No static value - runtime input
            .output_tensor_f32("reshaped", 2, None)
    }

    fn create_reshape_with_shape_input() -> NodeBuilder {
        NodeBuilder::new(NodeType::Reshape, "test_reshape_with_shape")
            .input_tensor_f32("data", 4, None)
            .add_input("shape", ArgType::Shape(2))
            .output_tensor_f32("reshaped", 2, None)
    }

    #[test]
    fn test_reshape_config_basic() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);
        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReshapeConfig>();
        match &config.shape {
            ReshapeInput::Static(shape) => assert_eq!(shape, &vec![2, 3]),
            _ => panic!("Expected static shape"),
        }
    }

    #[test]
    fn test_reshape_config_allowzero_supported() {
        let mut node = create_test_node(1, vec![2, 3]).build_with_graph_data(16);
        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
    }

    #[test]
    #[ignore] // TODO: Test needs redesign - runtime reshape requires rank information from output or Shape type input
    fn test_reshape_config_runtime() {
        let mut node = create_runtime_reshape_node().build();
        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReshapeConfig>();
        match &config.shape {
            ReshapeInput::Runtime(runtime_ref) => assert_eq!(runtime_ref.name, "shape"),
            _ => panic!("Expected runtime shape"),
        }
    }

    #[test]
    fn test_reshape_config_no_shape_input() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);
        node.inputs.pop(); // Remove the shape input
        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();

        let result = processor.infer_types(&mut node, 16, &prefs);
        assert!(matches!(
            result,
            Err(ProcessError::InvalidInputCount {
                expected: 2,
                actual: 1
            })
        ));
    }

    #[test]
    #[should_panic(expected = "shape tensor must be 1D")]
    fn test_reshape_config_invalid_shape_dim() {
        // Create a node with 2D shape tensor (should trigger panic)
        let node = NodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_with_data(
                "shape",
                ElementType::Int64,
                2, // 2D tensor (rank 2)
                crate::ir::Data::Int64s(vec![2, 3]),
                vec![2, 1], // 2D shape - this should cause panic
            )
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);

        let mut node = node;
        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();
    }

    #[test]
    fn test_reshape_update_outputs_basic() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_update_outputs_int() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            elem_type: ElementType::Int32,
            rank: 4,
            static_shape: None,
        });

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.elem_type, ElementType::Int32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_config_with_shape_type() {
        let mut node = create_reshape_with_shape_input().build();
        let processor = ReshapeProcessor;
        let prefs = OutputPreferences::new();
        let config = processor.extract_config(&node, 16).unwrap();
        node.config = config;
        processor.infer_types(&mut node, 16, &prefs).unwrap();

        let config = node.config::<ReshapeConfig>();
        match &config.shape {
            ReshapeInput::Runtime(runtime_ref) => assert_eq!(runtime_ref.name, "shape"),
            _ => panic!("Expected runtime shape"),
        }
    }

    #[test]
    fn test_reshape_update_outputs_with_shape_type() {
        let mut node = create_reshape_with_shape_input().build();

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.elem_type, ElementType::Float32);
                assert_eq!(tensor.rank, 2); // Should get rank from Shape(2) input
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_to_scalar() {
        // Test reshaping to a scalar (rank 0)
        let mut node = NodeBuilder::new(NodeType::Reshape, "test_reshape_scalar")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("shape", vec![], vec![0]) // Empty shape = scalar
            .output_tensor_f32("reshaped", 0, None)
            .build_with_graph_data(16);

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, ElementType::Float32);
            }
            _ => panic!("Expected scalar output"),
        }
    }
}
