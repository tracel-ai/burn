//! # LSTM
//!
//! Long Short-Term Memory recurrent neural network.
//!
//! **ONNX Spec**: <https://onnx.ai/onnx/operators/onnx__LSTM.html>
//!
//! ## Opset Versions
//! - **Opset 1**: Initial version
//! - **Opset 7**: Added `layout` attribute
//! - **Opset 14**: Added `layout` attribute with batch-first option

use crate::ir::{ArgType, Argument, Node, RawNode, TensorType};
use crate::processor::{
    InputSpec, NodeProcessor, NodeSpec, OutputPreferences, OutputSpec, ProcessError,
};
use derive_new::new;
use onnx_ir_derive::NodeBuilder;

/// Direction of LSTM processing
#[derive(Debug, Clone, PartialEq, Default)]
pub enum LstmDirection {
    /// Process sequence from start to end
    #[default]
    Forward,
    /// Process sequence from end to start
    Reverse,
    /// Process in both directions
    Bidirectional,
}

impl std::str::FromStr for LstmDirection {
    type Err = ProcessError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "forward" => Ok(LstmDirection::Forward),
            "reverse" => Ok(LstmDirection::Reverse),
            "bidirectional" => Ok(LstmDirection::Bidirectional),
            _ => Err(ProcessError::InvalidAttribute {
                name: "direction".to_string(),
                reason: format!(
                    "Invalid direction '{}'. Must be 'forward', 'reverse', or 'bidirectional'",
                    s
                ),
            }),
        }
    }
}

impl LstmDirection {
    /// Returns the number of directions (1 for forward/reverse, 2 for bidirectional)
    pub fn num_directions(&self) -> usize {
        match self {
            LstmDirection::Forward | LstmDirection::Reverse => 1,
            LstmDirection::Bidirectional => 2,
        }
    }
}

/// Configuration for LSTM operations
#[derive(Debug, Clone, new)]
#[allow(clippy::too_many_arguments)]
pub struct LstmConfig {
    /// Size of the input features
    pub input_size: usize,
    /// Number of neurons in the hidden layer (required ONNX attribute)
    pub hidden_size: usize,
    /// Direction of LSTM processing
    pub direction: LstmDirection,
    /// Whether bias is present (input B is provided)
    pub has_bias: bool,
    /// Whether initial hidden state is provided
    pub has_initial_h: bool,
    /// Whether initial cell state is provided
    pub has_initial_c: bool,
    /// Whether peephole weights are provided (not yet supported in Burn)
    pub has_peephole: bool,
    /// Tensor layout: false = seq_length major (default), true = batch_size major
    pub batch_first: bool,
}

/// Node representation for LSTM operation
#[derive(Debug, Clone, NodeBuilder)]
pub struct LstmNode {
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub config: LstmConfig,
}

pub(crate) struct LstmProcessor;

impl NodeProcessor for LstmProcessor {
    type Config = LstmConfig;

    fn spec(&self) -> NodeSpec {
        NodeSpec {
            min_opset: 1,
            max_opset: None,
            // Inputs: X, W, R (required), B, sequence_lens, initial_h, initial_c, P (optional)
            inputs: InputSpec::Range(3, 8),
            // Outputs: Y, Y_h, Y_c (all optional, but at least one should be present)
            outputs: OutputSpec::Range(0, 3),
        }
    }

    fn lift_constants(&self, node: &mut RawNode, _opset: usize) -> Result<(), ProcessError> {
        // W (weights) and R (recurrence weights) are typically constants
        if node.inputs.len() > 1 && node.inputs[1].is_constant() {
            node.inputs[1].to_static()?;
        }
        if node.inputs.len() > 2 && node.inputs[2].is_constant() {
            node.inputs[2].to_static()?;
        }
        // B (bias) is optional but typically constant
        if node.inputs.len() > 3 && node.inputs[3].is_constant() {
            node.inputs[3].to_static()?;
        }
        // P (peephole) is optional but typically constant
        if node.inputs.len() > 7 && node.inputs[7].is_constant() {
            node.inputs[7].to_static()?;
        }
        Ok(())
    }

    fn infer_types(
        &self,
        node: &mut RawNode,
        opset: usize,
        _output_preferences: &OutputPreferences,
    ) -> Result<(), ProcessError> {
        // Validate input tensor (X)
        let input_tensor = match &node.inputs[0].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[0].ty),
                });
            }
        };

        // LSTM expects 3D input: [seq_length, batch_size, input_size] or [batch_size, seq_length, input_size]
        if input_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "LSTM expects input tensor of rank 3, got rank {}",
                input_tensor.rank
            )));
        }

        // Validate weight tensor (W)
        let weight_tensor = match &node.inputs[1].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[1].ty),
                });
            }
        };

        if weight_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "LSTM expects weight tensor (W) of rank 3, got rank {}",
                weight_tensor.rank
            )));
        }

        // Validate recurrence weight tensor (R)
        let recurrence_tensor = match &node.inputs[2].ty {
            ArgType::Tensor(tensor) => tensor.clone(),
            _ => {
                return Err(ProcessError::TypeMismatch {
                    expected: "Tensor".to_string(),
                    actual: format!("{:?}", node.inputs[2].ty),
                });
            }
        };

        if recurrence_tensor.rank != 3 {
            return Err(ProcessError::Custom(format!(
                "LSTM expects recurrence weight tensor (R) of rank 3, got rank {}",
                recurrence_tensor.rank
            )));
        }

        // Validate optional bias tensor (B) if present and not empty
        if node.inputs.len() > 3
            && !node.inputs[3].is_optional()
            && let ArgType::Tensor(tensor) = &node.inputs[3].ty
            && tensor.rank != 2
        {
            return Err(ProcessError::Custom(format!(
                "LSTM expects bias tensor (B) of rank 2, got rank {}",
                tensor.rank
            )));
        }

        // Validate optional initial_h if present and not empty
        if node.inputs.len() > 5
            && !node.inputs[5].is_optional()
            && let ArgType::Tensor(tensor) = &node.inputs[5].ty
            && tensor.rank != 3
        {
            return Err(ProcessError::Custom(format!(
                "LSTM expects initial_h tensor of rank 3, got rank {}",
                tensor.rank
            )));
        }

        // Validate optional initial_c if present and not empty
        if node.inputs.len() > 6
            && !node.inputs[6].is_optional()
            && let ArgType::Tensor(tensor) = &node.inputs[6].ty
            && tensor.rank != 3
        {
            return Err(ProcessError::Custom(format!(
                "LSTM expects initial_c tensor of rank 3, got rank {}",
                tensor.rank
            )));
        }

        // Extract config to get hidden_size and direction for output type inference
        let config = self.extract_config(node, opset)?;
        let _num_directions = config.direction.num_directions();

        // Infer output types based on which outputs are requested
        // Output 0: Y - all hidden states [seq_length, num_directions, batch_size, hidden_size]
        if !node.outputs.is_empty() {
            node.outputs[0].ty = ArgType::Tensor(TensorType {
                dtype: input_tensor.dtype,
                rank: 4, // [seq_length, num_directions, batch_size, hidden_size]
                static_shape: None,
            });
        }

        // Output 1: Y_h - final hidden state [num_directions, batch_size, hidden_size]
        if node.outputs.len() > 1 {
            node.outputs[1].ty = ArgType::Tensor(TensorType {
                dtype: input_tensor.dtype,
                rank: 3,
                static_shape: None,
            });
        }

        // Output 2: Y_c - final cell state [num_directions, batch_size, hidden_size]
        if node.outputs.len() > 2 {
            node.outputs[2].ty = ArgType::Tensor(TensorType {
                dtype: input_tensor.dtype,
                rank: 3,
                static_shape: None,
            });
        }

        // Validate peephole is not used (not supported in Burn)
        if config.has_peephole {
            return Err(ProcessError::Custom(
                "LSTM peephole connections (input P) are not yet supported".to_string(),
            ));
        }

        // Validate sequence_lens is not used (not supported in Burn)
        if node.inputs.len() > 4 && !node.inputs[4].is_optional() {
            return Err(ProcessError::Custom(
                "LSTM sequence_lens input is not yet supported. All sequences must have the same length.".to_string(),
            ));
        }

        // Log warning for unsupported attributes
        for (key, _value) in node.attrs.iter() {
            match key.as_str() {
                "hidden_size" | "direction" | "layout" => {}
                "activations" | "activation_alpha" | "activation_beta" => {
                    // We only support default activations (Sigmoid, Tanh, Tanh)
                    // Custom activations would require burn-nn changes
                }
                "clip" => {
                    // Cell clipping is not supported
                }
                "input_forget" => {
                    // Input-forget coupling is not supported
                }
                _ => {}
            }
        }

        Ok(())
    }

    fn extract_config(&self, node: &RawNode, _opset: usize) -> Result<Self::Config, ProcessError> {
        // Get input_size from weight tensor shape: [num_directions, 4*hidden_size, input_size]
        let weight_shape = node.inputs[1]
            .value()
            .ok_or_else(|| {
                ProcessError::Custom("LSTM: weight tensor (W) must be a constant".to_string())
            })?
            .shape
            .to_vec();

        if weight_shape.len() != 3 {
            return Err(ProcessError::Custom(format!(
                "LSTM: weight tensor (W) must have rank 3, got {}",
                weight_shape.len()
            )));
        }

        let input_size = weight_shape[2];

        // Extract hidden_size from attributes (required)
        let hidden_size = node
            .attrs
            .get("hidden_size")
            .ok_or_else(|| ProcessError::MissingAttribute("hidden_size".to_string()))?
            .clone()
            .into_i64() as usize;

        // Extract direction (default: "forward")
        let direction = node
            .attrs
            .get("direction")
            .map(|v| v.clone().into_string())
            .unwrap_or_else(|| "forward".to_string());
        let direction: LstmDirection = direction.parse()?;

        // Extract layout (default: 0 = seq_length major)
        // layout attribute was added in opset 14, but we support it for all versions
        let layout = node
            .attrs
            .get("layout")
            .map(|v| v.clone().into_i64())
            .unwrap_or(0);
        let batch_first = layout == 1;

        // Check optional inputs
        let has_bias = node.inputs.len() > 3 && !node.inputs[3].is_optional();
        let has_initial_h = node.inputs.len() > 5 && !node.inputs[5].is_optional();
        let has_initial_c = node.inputs.len() > 6 && !node.inputs[6].is_optional();
        let has_peephole = node.inputs.len() > 7 && !node.inputs[7].is_optional();

        // Validate unsupported attributes
        if let Some(clip) = node.attrs.get("clip") {
            let clip_val = clip.clone().into_f32();
            if clip_val > 0.0 {
                return Err(ProcessError::Custom(
                    "LSTM clip attribute is not yet supported".to_string(),
                ));
            }
        }

        if let Some(input_forget) = node.attrs.get("input_forget") {
            let input_forget_val = input_forget.clone().into_i64();
            if input_forget_val != 0 {
                return Err(ProcessError::Custom(
                    "LSTM input_forget coupling is not yet supported".to_string(),
                ));
            }
        }

        // Validate activations are default (Sigmoid, Tanh, Tanh)
        if let Some(activations) = node.attrs.get("activations") {
            let acts = activations.clone().into_strings();
            let expected = vec!["Sigmoid", "Tanh", "Tanh"];
            let expected_bidirectional = vec!["Sigmoid", "Tanh", "Tanh", "Sigmoid", "Tanh", "Tanh"];

            let is_default = match direction {
                LstmDirection::Bidirectional => acts.is_empty() || acts == expected_bidirectional,
                _ => acts.is_empty() || acts == expected,
            };

            if !is_default {
                return Err(ProcessError::Custom(format!(
                    "LSTM custom activations {:?} are not yet supported. Only default activations (Sigmoid, Tanh, Tanh) are supported.",
                    acts
                )));
            }
        }

        let config = LstmConfig::new(
            input_size,
            hidden_size,
            direction,
            has_bias,
            has_initial_h,
            has_initial_c,
            has_peephole,
            batch_first,
        );

        Ok(config)
    }

    fn build_node(&self, builder: RawNode, opset: usize) -> Node {
        let config = self
            .extract_config(&builder, opset)
            .expect("Config extraction failed");

        Node::Lstm(LstmNode {
            name: builder.name,
            inputs: builder.inputs,
            outputs: builder.outputs,
            config,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::NodeType;
    use crate::node::test_utils::TestNodeBuilder;

    fn create_lstm_node(
        hidden_size: i64,
        direction: Option<&str>,
        layout: Option<i64>,
        num_outputs: usize,
    ) -> RawNode {
        let num_directions = match direction {
            Some("bidirectional") => 2,
            _ => 1,
        };

        let mut builder = TestNodeBuilder::new(NodeType::Lstm, "test_lstm")
            // X: [seq_length=10, batch_size=2, input_size=4]
            .input_tensor_f32("X", 3, Some(vec![10, 2, 4]))
            // W: [num_directions, 4*hidden_size, input_size]
            .input_tensor_f32_data(
                "W",
                vec![0.0; num_directions * 4 * hidden_size as usize * 4],
                vec![num_directions, 4 * hidden_size as usize, 4],
            )
            // R: [num_directions, 4*hidden_size, hidden_size]
            .input_tensor_f32_data(
                "R",
                vec![0.0; num_directions * 4 * hidden_size as usize * hidden_size as usize],
                vec![
                    num_directions,
                    4 * hidden_size as usize,
                    hidden_size as usize,
                ],
            )
            .attr_int("hidden_size", hidden_size);

        if let Some(dir) = direction {
            builder = builder.attr_string("direction", dir);
        }

        if let Some(lay) = layout {
            builder = builder.attr_int("layout", lay);
        }

        // Add outputs
        for i in 0..num_outputs {
            let output_name = match i {
                0 => "Y",
                1 => "Y_h",
                2 => "Y_c",
                _ => unreachable!(),
            };
            let output_rank = if i == 0 { 4 } else { 3 };
            builder = builder.output_tensor_f32(output_name, output_rank, None);
        }

        builder.build_with_graph_data(14) // opset 14
    }

    #[test]
    fn test_lstm_config_basic() {
        let node = create_lstm_node(8, None, None, 3);
        let processor = LstmProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert_eq!(config.input_size, 4);
        assert_eq!(config.hidden_size, 8);
        assert_eq!(config.direction, LstmDirection::Forward);
        assert!(!config.has_bias);
        assert!(!config.has_initial_h);
        assert!(!config.has_initial_c);
        assert!(!config.has_peephole);
        assert!(!config.batch_first);
    }

    #[test]
    fn test_lstm_config_bidirectional() {
        let node = create_lstm_node(8, Some("bidirectional"), None, 3);
        let processor = LstmProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert_eq!(config.direction, LstmDirection::Bidirectional);
        assert_eq!(config.direction.num_directions(), 2);
    }

    #[test]
    fn test_lstm_config_batch_first() {
        let node = create_lstm_node(8, None, Some(1), 3);
        let processor = LstmProcessor;
        let config = processor.extract_config(&node, 14).unwrap();

        assert!(config.batch_first);
    }

    #[test]
    fn test_lstm_type_inference() {
        let mut node = create_lstm_node(8, None, None, 3);
        let processor = LstmProcessor;
        let prefs = OutputPreferences::new();

        processor.infer_types(&mut node, 14, &prefs).unwrap();

        // Y: [seq_length, num_directions, batch_size, hidden_size] -> rank 4
        assert!(matches!(&node.outputs[0].ty, ArgType::Tensor(t) if t.rank == 4));
        // Y_h: [num_directions, batch_size, hidden_size] -> rank 3
        assert!(matches!(&node.outputs[1].ty, ArgType::Tensor(t) if t.rank == 3));
        // Y_c: [num_directions, batch_size, hidden_size] -> rank 3
        assert!(matches!(&node.outputs[2].ty, ArgType::Tensor(t) if t.rank == 3));
    }

    #[test]
    fn test_lstm_direction_parsing() {
        assert_eq!(
            "forward".parse::<LstmDirection>().unwrap(),
            LstmDirection::Forward
        );
        assert_eq!(
            "reverse".parse::<LstmDirection>().unwrap(),
            LstmDirection::Reverse
        );
        assert_eq!(
            "bidirectional".parse::<LstmDirection>().unwrap(),
            LstmDirection::Bidirectional
        );
        assert_eq!(
            "FORWARD".parse::<LstmDirection>().unwrap(),
            LstmDirection::Forward
        );
        assert!("invalid".parse::<LstmDirection>().is_err());
    }

    #[test]
    fn test_lstm_spec() {
        let processor = LstmProcessor;
        let spec = processor.spec();

        assert_eq!(spec.min_opset, 1);
        assert!(spec.max_opset.is_none());
    }
}
