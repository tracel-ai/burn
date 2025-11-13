//! # Reshape
//!
//! Reshapes the input tensor to a new shape specified by the shape input.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__Reshape.html>
//!
//! ## Special Features
//! - The `shape` input can contain special values:
//!   - `-1`: At most one dimension can be -1, which will be inferred from the tensor size
//!     and remaining dimensions
//!   - `0`: When allowzero=0 (default), copies the corresponding dimension from input tensor.
//!     When allowzero=1, sets the dimension to zero explicitly
//!   - Empty shape: Converts tensor to a scalar
//!
//! **NOTE**: The `allowzero` attribute (opset 14+) IS now validated in infer_types (lines 346-363).
//! When allowzero=1, the implementation correctly checks that shape cannot contain both 0 and -1.
//! However, the actual reshape logic respecting allowzero=1 behavior needs verification in codegen.
//!
//! ## Opset Versions
//! - **Opset 1-4**: Used 'shape' attribute (not supported in this implementation).
//! - **Opset 5**: Changed shape from attribute to input, enabling dynamic reshaping.
//! - **Opset 13**: Added support for more data types including bfloat16.
//! - **Opset 14**: Added 'allowzero' attribute to control zero-dimension handling.
//! - **Opset 19**: Clarified behavior and type constraints.
//! - **Opset 21**: Added support for 8-bit integer types (int4, uint4).
//!
//! **Implementation Note**: This implementation requires opset 5+ (shape as input). The allowzero attribute is mentioned in the spec but not currently validated or used in the implementation.

use crate::ir::{
    ArgType, Argument, Node, NodeBuilder, NodeConfig, RuntimeInputRef, TensorDataExt, TensorType,
};
use crate::processor::{
    InputPreferences, InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec,
    ProcessError,
};
use std::any::Any;

/// Configuration for the Reshape operation.
#[derive(Debug, Clone, Default)]
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
    /// Runtime shape determined during execution - references node.inputs\[input_index\].
    Runtime(RuntimeInputRef),
}

impl Default for ReshapeInput {
    fn default() -> Self {
        Self::Static(Vec::new())
    }
}

/// Update output rank for Reshape based on shape input if constant, otherwise use input rank.
pub fn reshape_update_outputs(node: &mut NodeBuilder) {
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
    dtype: crate::DType,
    is_shape: bool,
    shape_size: Option<usize>,
}

fn extract_input_info(input: &Argument) -> InputInfo {
    match &input.ty {
        ArgType::Tensor(tensor) => InputInfo {
            dtype: tensor.dtype,
            is_shape: false,
            shape_size: None,
        },
        ArgType::Shape(size) => InputInfo {
            dtype: crate::DType::I64,
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
    node: &NodeBuilder,
) -> ArgType {
    // Case 1: Scalar output (rank 0)
    if output_rank == 0 {
        return ArgType::Scalar(input_info.dtype);
    }

    // Case 2: Shape input -> Shape output (optimization)
    if input_info.is_shape && output_rank == 1 && input_info.dtype == crate::DType::I64 {
        let output_size =
            calculate_shape_output_size(input_info.shape_size.unwrap_or(1), node, &static_shape);

        return ArgType::Shape(output_size);
    }

    // Case 3: Regular tensor output
    ArgType::Tensor(TensorType {
        rank: output_rank,
        static_shape,
        dtype: input_info.dtype,
    })
}

/// Calculate the output size for Shape type outputs
fn calculate_shape_output_size(
    input_size: usize,
    node: &NodeBuilder,
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
fn infer_reshape_output_rank(node: &NodeBuilder) -> usize {
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
fn get_rank_from_shape_input(node: &NodeBuilder) -> Option<usize> {
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
fn get_rank_from_output(node: &NodeBuilder) -> Option<usize> {
    match &node.outputs[0].ty {
        ArgType::Tensor(tensor) => Some(tensor.rank),
        ArgType::Scalar(_) => Some(0),
        _ => None,
    }
}

/// Extract static shape from reshape node if available
fn get_static_shape(node: &NodeBuilder) -> Option<Vec<i64>> {
    // Check shape input
    if node.inputs.len() == 2
        && let Some(value) = node.inputs[1].value()
    {
        return value.to_i64_vec().ok();
    }

    None
}

/// Node processor for Reshape operation
pub struct ReshapeProcessor;

impl NodeProcessor for ReshapeProcessor {
    type Config = ReshapeConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 5,
            max_opset: None,
            inputs: InputSpec::Exact(2),
            outputs: OutputSpec::Exact(1),
        }
    }

    fn lift_constants(&self, node: &mut NodeBuilder, _opset: usize) -> Result<(), ProcessError> {
        // Only lift shape input (input[1]) if it has a static value
        // If it's a runtime argument (no value), it should remain in the graph
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }

        Ok(())
    }

    fn input_preferences(
        &self,
        node: &NodeBuilder,
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
        node: &mut NodeBuilder,
        _opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // TODO: Missing test coverage for allowzero=1 behavior
        // While allowzero attribute is validated (lines 346-363), there's no test that verifies
        // the actual reshape behavior when allowzero=1 and shape contains 0.
        // According to spec: with allowzero=1, a 0 in shape means "set dimension to 0",
        // not "copy from input". Add test: reshape_allowzero_explicit_zero

        // TODO: Missing test coverage for invalid shape values
        // Shape can contain negative values other than -1 (e.g., -2, -3). These should be rejected.
        // Add test: reshape_invalid_negative_value

        // TODO: Missing test coverage for more than one -1 in shape
        // Spec allows "at most one dimension" to be -1. Multiple -1s are invalid.
        // This is validated (line 338-342) but no test. Add test: reshape_multiple_infer_dim

        // TODO: Missing test coverage for incompatible total element count
        // When reshape shape specifies total elements != input total elements (and no -1 to infer),
        // this should fail. Add test: reshape_incompatible_size

        // Validate shape input type - must be Tensor or Shape
        match &node.inputs[1].ty {
            ArgType::Tensor(t) => {
                // Shape tensor must be 1D and int64 dtype
                if t.rank != 1 {
                    return Err(ProcessError::Custom(format!(
                        "Reshape: shape tensor must be 1D, got rank {}",
                        t.rank
                    )));
                }
                if t.dtype != crate::DType::I64 {
                    return Err(ProcessError::TypeMismatch {
                        expected: "Shape tensor with dtype I64".to_string(),
                        actual: format!("Shape tensor with dtype {:?}", t.dtype),
                    });
                }
            }
            ArgType::Shape(_) => {
                // Shape type is valid
            }
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor or Shape for shape input".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        }

        // Validate static shape values if available
        if let Some(shape_data) = node.inputs[1].value()
            && let Ok(shape_values) = shape_data.to_i64_vec()
        {
            // Count how many -1 values we have (at most one is allowed)
            let neg_one_count = shape_values.iter().filter(|&&v| v == -1).count();
            if neg_one_count > 1 {
                return Err(ProcessError::Custom(
                    "Reshape: shape can contain at most one -1 value".to_string(),
                ));
            }

            // If allowzero attribute is set, validate that we don't have both 0 and -1
            let mut allowzero = 0i64;
            for (key, value) in node.attrs.iter() {
                if key.as_str() == "allowzero" {
                    allowzero = value.clone().into_i64();
                    break;
                }
            }

            if allowzero == 1 {
                let has_zero = shape_values.contains(&0);
                let has_neg_one = shape_values.contains(&(-1));
                if has_zero && has_neg_one {
                    return Err(ProcessError::InvalidAttribute {
                        name: "allowzero".to_string(),
                        reason: "When allowzero=1, shape cannot contain both 0 and -1".to_string(),
                    });
                }
            }
        }

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
        node: &NodeBuilder,
        _opset: usize,
    ) -> Result<Self::Config, ProcessError> {
        // Extract shape input as either static or runtime
        let shape = match &node.inputs[1].ty {
            ArgType::Tensor(_tensor) => {
                // Extract shape from tensor input
                // Note: We don't validate rank here because extract_config runs before type inference
                // The rank might be 0 initially and will be updated during type inference
                match node.inputs[1].value() {
                    Some(tensor_data) => {
                        // Only validate when we have actual tensor data
                        assert_eq!(
                            tensor_data.shape.len(),
                            1,
                            "Reshape: shape tensor must be 1D"
                        );
                        ReshapeInput::Static(tensor_data.to_vec::<i64>().unwrap())
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
        Ok(config)
    }

    fn build_node(&self, builder: NodeBuilder, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Reshape {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DType;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_test_node(allowzero: i64, shape_vec: Vec<i64>) -> TestNodeBuilder {
        let mut builder = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_i64_data("shape", shape_vec.clone(), vec![shape_vec.len()])
            .output_tensor_f32("reshaped", 2, None);

        if allowzero != 0 {
            builder = builder.attr_int("allowzero", allowzero);
        }

        builder
    }

    fn create_runtime_reshape_node() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Reshape, "test_runtime_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("shape", 0, None) // No static value - runtime input
            .output_tensor_f32("reshaped", 2, None)
    }

    fn create_reshape_with_shape_input() -> TestNodeBuilder {
        TestNodeBuilder::new(NodeType::Reshape, "test_reshape_with_shape")
            .input_tensor_f32("data", 4, None)
            .add_input("shape", ArgType::Shape(2))
            .output_tensor_f32("reshaped", 2, None)
    }

    #[test]
    fn test_reshape_config_basic() {
        let node = create_test_node(0, vec![2, 3]).process(ReshapeProcessor, 16);

        let processor = ReshapeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
        match &config.shape {
            ReshapeInput::Static(shape) => assert_eq!(shape, &vec![2, 3]),
            _ => panic!("Expected static shape"),
        }
    }

    #[test]
    fn test_reshape_config_allowzero_supported() {
        let _node = create_test_node(1, vec![2, 3]).process(ReshapeProcessor, 16);
        // Test passes if no panic occurs during processing
    }

    #[test]
    #[ignore] // TODO: Test needs redesign - runtime reshape requires rank information from output or Shape type input
    fn test_reshape_config_runtime() {
        let node = create_runtime_reshape_node().process(ReshapeProcessor, 16);

        let processor = ReshapeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
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
        let spec = processor.spec();
        let result = crate::processor::validate_node_spec(&node, 16, &spec);
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
        let node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape")
            .input_tensor_f32("data", 4, None)
            .input_tensor_with_data(
                "shape",
                DType::I64,
                2,                                                     // 2D tensor (rank 2)
                crate::ir::TensorData::new(vec![2i64, 3], vec![2, 1]), // 2D shape - this should cause panic
            )
            .output_tensor_f32("reshaped", 2, None)
            .build_with_graph_data(16);
        let processor = ReshapeProcessor;
        // This should panic when validating the shape tensor is 1D
        let _ = processor.extract_config(&node, 16);
    }

    #[test]
    fn test_reshape_update_outputs_basic() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_update_outputs_int() {
        let mut node = create_test_node(0, vec![2, 3]).build_with_graph_data(16);
        node.inputs[0].ty = ArgType::Tensor(TensorType {
            dtype: DType::I32,
            rank: 4,
            static_shape: None,
        });

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.static_shape, None);
                assert_eq!(tensor.dtype, DType::I32);
                assert_eq!(tensor.rank, 2);
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_config_with_shape_type() {
        let node = create_reshape_with_shape_input().process(ReshapeProcessor, 16);

        let processor = ReshapeProcessor;
        let config = processor.extract_config(&node, 16).unwrap();
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
                assert_eq!(tensor.dtype, DType::F32);
                assert_eq!(tensor.rank, 2); // Should get rank from Shape(2) input
            }
            _ => panic!("Expected tensor output"),
        }
    }

    #[test]
    fn test_reshape_to_scalar() {
        // Test reshaping to a scalar (rank 0)
        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_reshape_scalar")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64_data("shape", vec![], vec![0]) // Empty shape = scalar
            .output_tensor_f32("reshaped", 0, None)
            .build_with_graph_data(16);

        reshape_update_outputs(&mut node);
        match &node.outputs[0].ty {
            ArgType::Scalar(elem_type) => {
                assert_eq!(*elem_type, DType::F32);
            }
            _ => panic!("Expected scalar output"),
        }
    }

    #[test]
    fn test_reshape_dynamic_shape_with_output_rank() {
        // Test dynamic reshape where shape input has no static_shape,
        // but output rank is known from ONNX model metadata.
        // This simulates real-world models where shape is computed by other nodes (e.g., Concat)
        // but the ONNX model's value_info already specifies the output rank.

        let mut node = TestNodeBuilder::new(NodeType::Reshape, "test_dynamic_reshape")
            .input_tensor_f32("data", 2, None)
            .input_tensor_i64("shape", 1, None) // Dynamic shape input - no static value
            .output_tensor_f32("reshaped", 4, None) // Output has rank 4 but no static_shape
            .build();

        // Verify the shape input has no static_shape (simulating dynamic case)
        match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 1);
                assert_eq!(tensor.static_shape, None); // No static shape
            }
            _ => panic!("Expected tensor shape input"),
        }

        // Verify output has rank but no static_shape
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.static_shape, None); // No static shape, just rank
            }
            _ => panic!("Expected tensor output"),
        }

        // This should succeed by using the output's rank field
        reshape_update_outputs(&mut node);

        // Verify the output still has the correct rank
        match &node.outputs[0].ty {
            ArgType::Tensor(tensor) => {
                assert_eq!(tensor.rank, 4);
                assert_eq!(tensor.dtype, DType::F32);
            }
            _ => panic!("Expected tensor output"),
        }
    }
}
