use crate::ir::{
    ArgType, Argument, AttributeValue, ElementType, Node, NodeType, TensorData, TensorType,
};
use std::collections::HashMap;

/// Builder for creating test node instances with convenient defaults and simple API.
pub struct NodeBuilder {
    node_type: NodeType,
    name: String,
    inputs: Vec<Argument>,
    outputs: Vec<Argument>,
    attrs: HashMap<String, AttributeValue>,
    /// Stores constant data for inputs that should be constants (input_name -> (data, shape))
    constant_data: HashMap<String, TensorData>,
}

impl NodeBuilder {
    /// Create a new builder with the specified node type and name
    pub fn new(node_type: NodeType, name: &str) -> Self {
        Self {
            node_type,
            name: name.to_string(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: HashMap::new(),
            constant_data: HashMap::new(),
        }
    }

    /// Add a generic input with the given name and type
    ///
    /// Note: Prefer using the specialized methods like `input_tensor_f32`,
    /// `input_scalar_f32`, etc. for better readability and type safety.
    #[doc(hidden)]
    pub fn add_input(mut self, name: &str, ty: ArgType) -> Self {
        // In ONNX protobuf, optional inputs are represented by empty names
        let value_source = if name.is_empty() {
            crate::ir::ValueSource::Optional
        } else {
            crate::ir::ValueSource::Dynamic
        };

        self.inputs.push(Argument {
            name: name.to_string(),
            ty,
            data_id: None,
            value_source,
            value_store: None,
        });
        self
    }

    /// Add a float32 tensor input with the given name and rank
    pub fn input_tensor_f32(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a float64 tensor input with the given name and rank
    pub fn input_tensor_f64(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float64,
                rank,
                static_shape,
            }),
        )
    }

    /// Add an int32 tensor input with the given name and rank
    pub fn input_tensor_i32(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Int32,
                rank,
                static_shape,
            }),
        )
    }

    /// Add an int64 tensor input with the given name and rank
    pub fn input_tensor_i64(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a bool tensor input with the given name and rank
    pub fn input_tensor_bool(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a float16 tensor input with the given name and rank
    pub fn input_tensor_f16(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float16,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a string tensor input with the given name and rank
    pub fn input_tensor_string(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_input(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::String,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a scalar input with the given name and element type
    pub fn input_scalar(self, name: &str, elem_type: ElementType) -> Self {
        self.add_input(name, ArgType::Scalar(elem_type))
    }

    /// Add a float32 scalar input with the given name
    pub fn input_scalar_f32(self, name: &str) -> Self {
        self.input_scalar(name, ElementType::Float32)
    }

    /// Add an int64 scalar input with the given name
    pub fn input_scalar_i64(self, name: &str) -> Self {
        self.input_scalar(name, ElementType::Int64)
    }

    /// Add a shape input with the given name and rank
    pub fn input_shape(self, name: &str, rank: usize) -> Self {
        self.add_input(name, ArgType::Shape(rank))
    }

    /// Add a tensor input with data value
    ///
    /// Note: In the new design, constant values are stored in GraphState, not in Arguments.
    /// This method creates the argument without the value field. If you need the value
    /// in GraphState for testing, you'll need to add it separately.
    pub fn input_tensor_with_data(
        mut self,
        name: &str,
        elem_type: ElementType,
        rank: usize,
        tensor_data: TensorData,
    ) -> Self {
        let arg = Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type,
                rank,
                static_shape: None,
            }),
            data_id: None,
            value_source: crate::ir::ValueSource::Constant,
            value_store: None,
        };
        self.inputs.push(arg);
        // Store the constant data for later registration in GraphState
        self.constant_data.insert(name.to_string(), tensor_data);
        self
    }

    /// Add a float32 tensor input with data values
    pub fn input_tensor_f32_data(self, name: &str, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let tensor_data = TensorData::new(data, shape.clone());
        self.input_tensor_with_data(name, ElementType::Float32, shape.len(), tensor_data)
    }

    /// Add an int64 tensor input with data values
    pub fn input_tensor_i64_data(self, name: &str, data: Vec<i64>, shape: Vec<usize>) -> Self {
        let tensor_data = TensorData::new(data, shape.clone());
        self.input_tensor_with_data(name, ElementType::Int64, shape.len(), tensor_data)
    }

    /// Add a float32 scalar tensor input (rank 0)
    ///
    /// Note: In the new design, constant values are stored in GraphState, not in Arguments.
    /// This method creates the argument without the value field.
    pub fn input_scalar_tensor_f32(mut self, name: &str, value: Option<f32>) -> Self {
        let value_source = if value.is_some() {
            crate::ir::ValueSource::Constant
        } else {
            crate::ir::ValueSource::Dynamic
        };
        let arg = Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0,
                static_shape: None,
            }),
            data_id: None,
            value_source,
            value_store: None,
        };
        self.inputs.push(arg);
        // If value is provided, store it as constant data
        if let Some(v) = value {
            self.constant_data
                .insert(name.to_string(), TensorData::new(vec![v], vec![]));
        }
        self
    }

    /// Add an int64 scalar tensor input (rank 0)
    ///
    /// Note: In the new design, constant values are stored in GraphState, not in Arguments.
    /// This method creates the argument without the value field.
    pub fn input_scalar_tensor_i64(mut self, name: &str, value: Option<i64>) -> Self {
        let value_source = if value.is_some() {
            crate::ir::ValueSource::Constant
        } else {
            crate::ir::ValueSource::Dynamic
        };
        let arg = Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 0,
                static_shape: None,
            }),
            data_id: None,
            value_source,
            value_store: None,
        };
        self.inputs.push(arg);
        // If value is provided, store it as constant data
        if let Some(v) = value {
            self.constant_data
                .insert(name.to_string(), TensorData::new(vec![v], vec![]));
        }
        self
    }

    /// Add multiple tensor inputs with the same type but different names
    pub fn input_tensors_f32<I>(
        mut self,
        name_prefix: &str,
        count: usize,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        for i in 0..count {
            self = self.input_tensor_f32(&format!("{name_prefix}_{i}"), rank, static_shape.clone());
        }
        self
    }

    /// Add a generic output with the given name and type
    ///
    /// Note: Prefer using the specialized methods like `output_tensor_f32`,
    /// `output_scalar_f32`, etc. for better readability and type safety.
    #[doc(hidden)]
    pub fn add_output(mut self, name: &str, ty: ArgType) -> Self {
        self.outputs.push(Argument {
            name: name.to_string(),
            ty,
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        self
    }

    /// Add a float32 tensor output with the given name and rank
    pub fn output_tensor_f32(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a float64 tensor output with the given name and rank
    pub fn output_tensor_f64(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float64,
                rank,
                static_shape,
            }),
        )
    }

    /// Add an int32 tensor output with the given name and rank
    pub fn output_tensor_i32(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Int32,
                rank,
                static_shape,
            }),
        )
    }

    /// Add an int64 tensor output with the given name and rank
    pub fn output_tensor_i64(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a bool tensor output with the given name and rank
    pub fn output_tensor_bool(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Bool,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a float16 tensor output with the given name and rank
    pub fn output_tensor_f16(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::Float16,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a string tensor output with the given name and rank
    pub fn output_tensor_string(
        self,
        name: &str,
        rank: usize,
        static_shape: Option<Vec<usize>>,
    ) -> Self {
        self.add_output(
            name,
            ArgType::Tensor(TensorType {
                elem_type: ElementType::String,
                rank,
                static_shape,
            }),
        )
    }

    /// Add a scalar output with the given name and element type
    pub fn output_scalar(self, name: &str, elem_type: ElementType) -> Self {
        self.add_output(name, ArgType::Scalar(elem_type))
    }

    /// Add a float32 scalar output with the given name
    pub fn output_scalar_f32(self, name: &str) -> Self {
        self.output_scalar(name, ElementType::Float32)
    }

    /// Add an int64 scalar output with the given name
    pub fn output_scalar_i64(self, name: &str) -> Self {
        self.output_scalar(name, ElementType::Int64)
    }

    /// Add a shape output with the given name and rank
    pub fn output_shape(self, name: &str, rank: usize) -> Self {
        self.add_output(name, ArgType::Shape(rank))
    }

    /// Add an integer attribute
    pub fn attr_int(mut self, name: &str, value: i64) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::Int64(value));
        self
    }

    /// Add a float attribute
    pub fn attr_float(mut self, name: &str, value: f32) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::Float32(value));
        self
    }

    /// Add a string attribute
    pub fn attr_string(mut self, name: &str, value: &str) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::String(value.to_string()));
        self
    }

    /// Add an integer array attribute
    pub fn attr_ints(mut self, name: &str, values: Vec<i64>) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::Int64s(values));
        self
    }

    /// Add a float array attribute
    pub fn attr_floats(mut self, name: &str, values: Vec<f32>) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::Float32s(values));
        self
    }

    /// Add a string array attribute
    pub fn attr_strings(mut self, name: &str, values: Vec<String>) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::Strings(values));
        self
    }

    /// Add a tensor attribute
    pub fn attr_tensor(mut self, name: &str, tensor: TensorData) -> Self {
        self.attrs
            .insert(name.to_string(), AttributeValue::Tensor(tensor));
        self
    }

    /// Add a default output with the given name
    pub fn output_default(mut self, name: &str) -> Self {
        self.outputs.push(Argument {
            name: name.to_string(),
            ty: ArgType::default(),
            data_id: None,
            value_source: crate::ir::ValueSource::Dynamic,
            value_store: None,
        });
        self
    }

    /// Build the node
    pub fn build(self) -> Node {
        Node {
            node_type: self.node_type,
            name: self.name,
            inputs: self.inputs,
            outputs: self.outputs,
            attrs: self.attrs,
            config: None,
        }
    }

    /// Build the node and register any constant inputs in GraphState.
    /// This is useful for tests that need constant values accessible via GraphState.
    ///
    /// Note: After calling this method, the GraphState will be wrapped in Rc<RefCell<>>
    /// and attached to the node's arguments.
    pub fn build_with_graph_data(self, _opset: usize) -> Node {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Create a new GraphState for this test
        let mut graph_data = crate::graph_state::GraphState::new(&[], &[], &[]);

        // Register constants in GraphState before building the node
        for (input_name, tensor_data) in &self.constant_data {
            graph_data.register_test_constant(input_name.clone(), tensor_data.clone());
        }

        // Build the node first
        let mut node = self.build();

        // Update input arguments to have data_id from registered constants
        for arg in &mut node.inputs {
            if let Some(data_id) = graph_data.get_constant_data_id(&arg.name) {
                arg.data_id = Some(data_id);
            }
        }

        // Wrap GraphState in Rc<RefCell<>> and attach to all arguments
        let graph_data_rc = Rc::new(RefCell::new(graph_data));

        for arg in &mut node.inputs {
            arg.value_store = Some(graph_data_rc.clone());
        }
        for arg in &mut node.outputs {
            arg.value_store = Some(graph_data_rc.clone());
        }

        node
    }
}
