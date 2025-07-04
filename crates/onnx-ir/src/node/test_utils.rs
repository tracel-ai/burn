use crate::ir::{
    ArgType, Argument, AttributeValue, Data, ElementType, Node, NodeType, TensorData, TensorType,
};
use std::collections::HashMap;

/// Builder for creating test node instances with convenient defaults and simple API.
pub struct NodeBuilder {
    node_type: NodeType,
    name: String,
    inputs: Vec<Argument>,
    outputs: Vec<Argument>,
    attrs: HashMap<String, AttributeValue>,
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
        }
    }

    /// Add a generic input with the given name and type
    ///
    /// Note: Prefer using the specialized methods like `input_tensor_f32`,
    /// `input_scalar_f32`, etc. for better readability and type safety.
    #[doc(hidden)]
    pub fn add_input(mut self, name: &str, ty: ArgType) -> Self {
        self.inputs.push(Argument {
            name: name.to_string(),
            ty,
            value: None,
            passed: true,
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
    pub fn input_tensor_with_data(
        mut self,
        name: &str,
        elem_type: ElementType,
        rank: usize,
        data: Data,
        shape: Vec<usize>,
    ) -> Self {
        let arg = Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type,
                rank,
                static_shape: None,
            }),
            value: Some(TensorData { data, shape }),
            passed: true,
        };
        self.inputs.push(arg);
        self
    }

    /// Add a float32 tensor input with data values
    pub fn input_tensor_f32_data(self, name: &str, data: Vec<f32>, shape: Vec<usize>) -> Self {
        self.input_tensor_with_data(
            name,
            ElementType::Float32,
            shape.len(),
            Data::Float32s(data),
            shape,
        )
    }

    /// Add an int64 tensor input with data values
    pub fn input_tensor_i64_data(self, name: &str, data: Vec<i64>, shape: Vec<usize>) -> Self {
        self.input_tensor_with_data(
            name,
            ElementType::Int64,
            shape.len(),
            Data::Int64s(data),
            shape,
        )
    }

    /// Add a float32 scalar tensor input (rank 0)
    pub fn input_scalar_tensor_f32(mut self, name: &str, value: Option<f32>) -> Self {
        let arg = Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Float32,
                rank: 0,
                static_shape: None,
            }),
            value: value.map(|val| TensorData {
                data: Data::Float32(val),
                shape: vec![],
            }),
            passed: true,
        };
        self.inputs.push(arg);
        self
    }

    /// Add an int64 scalar tensor input (rank 0)
    pub fn input_scalar_tensor_i64(mut self, name: &str, value: i64) -> Self {
        let arg = Argument {
            name: name.to_string(),
            ty: ArgType::Tensor(TensorType {
                elem_type: ElementType::Int64,
                rank: 0,
                static_shape: None,
            }),
            value: Some(TensorData {
                data: Data::Int64(value),
                shape: vec![],
            }),
            passed: true,
        };
        self.inputs.push(arg);
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
            value: None,
            passed: true,
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
            value: None,
            passed: true,
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
        }
    }
}
