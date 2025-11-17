//! ONNX argument types
//!
//! This module contains types for representing node inputs and outputs,
//! including their types, data sources, and metadata.

use core::fmt;
use std::{cell::RefCell, fmt::Formatter, rc::Rc};

use burn_tensor::DType;

use super::tensor_data_ext::TensorData;

pub type Rank = usize;
pub type Shape = Vec<usize>;

/// Unique identifier for tensor data in the central store
pub type DataId = usize;

/// Describes where an argument's value comes from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSource {
    /// Static constant value embedded in the argument (name="" with embedded data)
    Static(DataId),
    /// Points to a constant node output (name="constant1_out1")
    Constant,
    /// Points to a runtime node output (name="conv1_out1")
    Dynamic,
    /// Optional/not provided (name="")
    Optional,
}

/// A node input or output.
#[derive(Clone)]
pub struct Argument {
    /// The name of the node input.
    pub name: String,

    /// The type of the argument.
    pub ty: ArgType,

    /// Describes where this argument's value comes from
    /// For Static values, contains the tensor data ID directly
    pub value_source: ValueSource,

    /// Reference to the value store for lazy constant lookup and type expectations
    pub(crate) value_store: Option<Rc<RefCell<crate::graph_state::GraphState>>>,
}

impl fmt::Debug for Argument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Argument")
            .field("name", &self.name)
            .field("ty", &self.ty)
            .field("value_source", &self.value_source)
            .finish()
    }
}

impl Argument {
    /// Copy everything except the name from the other argument
    pub fn copy_value(&mut self, other_arg: &Argument) {
        self.ty = other_arg.ty.clone();
        self.value_source = other_arg.value_source;
    }
}

/// The type of an argument.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgType {
    Scalar(DType),
    Shape(Rank),
    Tensor(TensorType),
}

/// Represents the type of a tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorType {
    /// The data type of the tensor values (e.g. F32, F64, I64, etc.)
    pub dtype: DType,

    /// The number of dimensions in the tensor
    pub rank: Rank,

    /// Static shape if known (populated during shape inference)
    pub static_shape: Option<Vec<usize>>,
}

impl Default for TensorType {
    fn default() -> Self {
        Self {
            dtype: DType::F32,
            rank: 0,
            static_shape: None,
        }
    }
}

impl TensorType {
    pub fn new(dtype: DType, rank: Rank, static_shape: Option<Vec<usize>>) -> Self {
        Self {
            dtype,
            rank,
            static_shape,
        }
    }
}

impl Default for ArgType {
    fn default() -> Self {
        Self::Tensor(TensorType::default())
    }
}

impl ArgType {
    /// Check if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }

    /// Check if this is a tensor type
    pub fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor(_))
    }

    /// Check if this is a shape type
    pub fn is_shape(&self) -> bool {
        matches!(self, Self::Shape(_))
    }

    /// Get the rank (number of dimensions)
    pub fn rank(&self) -> usize {
        match self {
            ArgType::Scalar(_) => 0,
            ArgType::Shape(_) => 1,
            ArgType::Tensor(t) => t.rank,
        }
    }

    /// Get the data type
    pub fn elem_type(&self) -> DType {
        match self {
            ArgType::Scalar(s) => *s,
            ArgType::Shape(_) => panic!("ArgType::Shape has no DType"),
            ArgType::Tensor(t) => t.dtype,
        }
    }

    /// Get the static shape if available
    pub fn static_shape(&self) -> Option<&Vec<usize>> {
        match self {
            ArgType::Tensor(t) => t.static_shape.as_ref(),
            _ => None,
        }
    }
}

impl Argument {
    /// Create a new argument with a specific type
    pub fn new(name: impl Into<String>, ty: ArgType) -> Self {
        let name = name.into();
        // Default to Dynamic (points to a node output by name)
        let value_source = if name.is_empty() {
            ValueSource::Optional
        } else {
            ValueSource::Dynamic
        };

        Self {
            name,
            ty,
            value_source,
            value_store: None,
        }
    }

    /// Create a new argument with default type (F32 tensor rank 0)
    pub fn from_name(name: impl Into<String>) -> Self {
        Self::new(name, ArgType::default())
    }

    /// Get the constant value from the central tensor store
    pub fn value(&self) -> Option<TensorData> {
        let store = self.value_store.as_ref()?;

        match &self.value_source {
            // Static: data is embedded directly
            ValueSource::Static(data_id) => store.borrow().get_tensor_data(*data_id).cloned(),
            // Constant: look up the constant node by output name
            ValueSource::Constant => {
                let data_id = store.borrow().get_constant_data_id_by_output(&self.name)?;
                store.borrow().get_tensor_data(data_id).cloned()
            }
            // Dynamic/Optional: no constant data
            ValueSource::Dynamic | ValueSource::Optional => None,
        }
    }

    /// Check if this is a static constant (embedded value)
    pub fn is_static(&self) -> bool {
        matches!(self.value_source, ValueSource::Static(_))
    }

    /// Check if this argument points to a constant node output
    pub fn is_constant(&self) -> bool {
        self.value_source == ValueSource::Constant
    }

    /// Check if this argument points to a runtime node output
    pub fn is_dynamic(&self) -> bool {
        self.value_source == ValueSource::Dynamic
    }

    /// Check if this argument is optional/not provided
    pub fn is_optional(&self) -> bool {
        self.value_source == ValueSource::Optional
    }

    /// Convert a Constant argument to Static by embedding the constant's data
    ///
    /// This looks up the constant node by name, retrieves its data_id,
    /// and embeds it in this argument, clearing the name.
    ///
    /// Returns an error if this is not a Constant argument.
    pub fn to_static(&mut self) -> Result<(), crate::processor::ProcessError> {
        use crate::processor::ProcessError;

        if !self.is_constant() {
            return Err(ProcessError::Custom(format!(
                "Cannot convert {:?} argument to Static (only Constant can be converted)",
                self.value_source
            )));
        }

        // Look up the constant node by name
        let store = self.value_store.as_ref().ok_or_else(|| {
            ProcessError::Custom("No value store available to look up constant".to_string())
        })?;

        let data_id = {
            let graph_data = store.borrow();

            // Get the data_id from the constant node using the output name
            graph_data
                .get_constant_data_id_by_output(&self.name)
                .ok_or_else(|| {
                    ProcessError::Custom(format!(
                        "Constant node not found or has no data_id for output name: {}",
                        self.name
                    ))
                })?
        };

        // Embed the data_id in ValueSource::Static, clear the name
        // The name is cleared because Static values are accessed via data_id, not by name
        self.value_source = ValueSource::Static(data_id);
        self.name.clear();

        Ok(())
    }
}
