use core::fmt;
use std::{any::Any, cell::RefCell, collections::HashMap, fmt::Formatter, rc::Rc};
use strum::{Display, EnumString};

// burn-tensor integration
pub use burn_tensor::DType;

/// Reference to a runtime input by name and index.
/// Used in configs to point to node inputs instead of storing stale copies.
#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeInputRef {
    /// Name of the input argument
    pub name: String,
    /// Index in the node's inputs array
    pub input_index: usize,
}

impl RuntimeInputRef {
    pub fn new(name: String, input_index: usize) -> Self {
        Self { name, input_index }
    }
}

/// Trait for node-specific configuration
/// Each node type can have its own configuration struct that implements this trait
pub trait NodeConfig {
    /// Downcast to Any for type-safe retrieval
    fn as_any(&self) -> &dyn Any;

    /// Clone the config into a boxed trait object
    fn clone_box(&self) -> Box<dyn NodeConfig>;
}

pub type Rank = usize;
pub type Shape = Vec<usize>;

/// Unique identifier for tensor data in the central store
pub type TensorId = usize;

/// Describes where an argument's value comes from
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueSource {
    /// Static constant value embedded in the argument (name="" + data_id=Some)
    Static,
    /// Points to a constant node output (name="constant1_out1" + data_id=None)
    Constant,
    /// Points to a runtime node output (name="conv1_out1" + data_id=None)
    Dynamic,
    /// Optional/not provided (name="" + data_id=None)
    Optional,
}

/// A node input or output.
#[derive(Clone)]
pub struct Argument {
    /// The name of the node input.
    pub name: String,

    /// The type of the argument.
    pub ty: ArgType,

    /// Unique ID referencing tensor data in central store
    /// Some = this argument has constant/static data available
    /// None = runtime data only
    pub data_id: Option<TensorId>,

    /// Describes where this argument's value comes from
    pub value_source: ValueSource,

    /// Reference to the value store for lazy constant lookup and type expectations
    pub(crate) value_store: Option<Rc<RefCell<crate::graph_state::GraphState>>>,
}

impl fmt::Debug for Argument {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Argument")
            .field("name", &self.name)
            .field("ty", &self.ty)
            .field("data_id", &self.data_id)
            .field("value_source", &self.value_source)
            .field(
                "value_store",
                &self.value_store.as_ref().map(|_| "Rc<RefCell<GraphState>>"),
            )
            .finish()
    }
}

impl Argument {
    /// Copy everything except the name from the other argument
    pub fn copy_value(&mut self, other_arg: &Argument) {
        self.ty = other_arg.ty.clone();
        self.data_id = other_arg.data_id;
    }
}

/// The type of an argument.
#[derive(Debug, Clone, PartialEq)]
pub enum ArgType {
    Scalar(DType),
    Shape(Rank),
    Tensor(TensorType),
}

/// The type of an attribute.
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Float32(f32),
    Float32s(Vec<f32>),
    Int64(i64),
    Int64s(Vec<i64>),
    String(String),
    Strings(Vec<String>),
    Tensor(TensorData),
    Tensors(Vec<TensorData>),
}

pub type Attributes = HashMap<String, AttributeValue>;

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
    pub fn new(name: String) -> Self {
        // Default to Dynamic (points to a node output by name)
        let value_source = if name.is_empty() {
            ValueSource::Optional
        } else {
            ValueSource::Dynamic
        };

        Self {
            name,
            ty: ArgType::default(),
            data_id: None,
            value_source,
            value_store: None,
        }
    }

    /// Get the constant value from the central tensor store
    pub fn value(&self) -> Option<TensorData> {
        let store = self.value_store.as_ref()?;

        // If this argument has a direct data_id (Static or Dynamic with data), use it
        if let Some(data_id) = self.data_id {
            return store.borrow().get_tensor_data(data_id).cloned();
        }

        // If this is a Constant argument (points to constant node by output name),
        // look up the constant node and get the data from its input
        if self.is_constant() {
            let data_id = store.borrow().get_constant_data_id_by_output(&self.name)?;
            return store.borrow().get_tensor_data(data_id).cloned();
        }

        None
    }

    /// Check if this is a static constant (embedded value)
    pub fn is_static(&self) -> bool {
        self.value_source == ValueSource::Static
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

        // Embed the data_id, clear the name, and mark as Static
        // The name is cleared because Static values are accessed via data_id, not by name
        self.data_id = Some(data_id);
        self.name.clear();
        self.value_source = ValueSource::Static;

        Ok(())
    }
}

/// Re-export burn-tensor's TensorData for direct use
pub use burn_tensor::TensorData;

/// Extension trait providing ONNX-specific helper methods for TensorData
pub trait TensorDataExt {
    /// Alias for dtype for backward compatibility with ONNX naming
    fn elem_type(&self) -> DType;

    /// Extract the first element as f64, converting from any numeric type
    ///
    /// Useful for extracting scalar parameters from constant nodes.
    fn scalar_f64(&self) -> Result<f64, burn_tensor::DataError>;

    /// Extract the first element as f32, converting from any numeric type
    fn scalar_f32(&self) -> Result<f32, burn_tensor::DataError>;

    /// Extract the first element as i64, converting from any numeric type
    fn scalar_i64(&self) -> Result<i64, burn_tensor::DataError>;

    /// Convert to Vec<i64>, handling Int32 and Int64 types
    ///
    /// Useful for extracting indices, shapes, or other integer arrays that need to be i64.
    fn to_i64_vec(&self) -> Result<Vec<i64>, burn_tensor::DataError>;

    /// Convert to Vec<usize>, handling Int32 and Int64 types
    ///
    /// Useful for extracting shape or dimension values.
    fn to_usize_vec(&self) -> Result<Vec<usize>, burn_tensor::DataError>;

    /// Convert to Vec<f32>, handling all numeric types with automatic conversion
    ///
    /// Useful for extracting numeric arrays that need to be f32.
    fn to_f32_vec(&self) -> Result<Vec<f32>, burn_tensor::DataError>;
}

impl TensorDataExt for burn_tensor::TensorData {
    fn elem_type(&self) -> DType {
        self.dtype
    }

    fn scalar_f64(&self) -> Result<f64, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::F16 => {
                let val = self.as_slice::<half::f16>()?[0];
                Ok(f32::from(val) as f64)
            }
            DType::F32 => Ok(self.as_slice::<f32>()?[0] as f64),
            DType::F64 => Ok(self.as_slice::<f64>()?[0]),
            DType::I32 => Ok(self.as_slice::<i32>()?[0] as f64),
            DType::I64 => Ok(self.as_slice::<i64>()?[0] as f64),
            DType::I8 => Ok(self.as_slice::<i8>()?[0] as f64),
            DType::U8 => Ok(self.as_slice::<u8>()?[0] as f64),
            DType::U16 => Ok(self.as_slice::<u16>()?[0] as f64),
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to f64",
                other
            ))),
        }
    }

    fn scalar_f32(&self) -> Result<f32, burn_tensor::DataError> {
        self.scalar_f64().map(|v| v as f32)
    }

    fn scalar_i64(&self) -> Result<i64, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::I64 => Ok(self.as_slice::<i64>()?[0]),
            DType::I32 => Ok(self.as_slice::<i32>()?[0] as i64),
            DType::I8 => Ok(self.as_slice::<i8>()?[0] as i64),
            DType::U8 => Ok(self.as_slice::<u8>()?[0] as i64),
            DType::U16 => Ok(self.as_slice::<u16>()?[0] as i64),
            DType::F32 => Ok(self.as_slice::<f32>()?[0] as i64),
            DType::F64 => Ok(self.as_slice::<f64>()?[0] as i64),
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to i64",
                other
            ))),
        }
    }

    fn to_i64_vec(&self) -> Result<Vec<i64>, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::I64 => self.to_vec::<i64>(),
            DType::I32 => {
                let vec_i32 = self.to_vec::<i32>()?;
                Ok(vec_i32.into_iter().map(|v| v as i64).collect())
            }
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to Vec<i64>",
                other
            ))),
        }
    }

    fn to_usize_vec(&self) -> Result<Vec<usize>, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::I64 => {
                let vec_i64 = self.to_vec::<i64>()?;
                Ok(vec_i64.into_iter().map(|v| v as usize).collect())
            }
            DType::I32 => {
                let vec_i32 = self.to_vec::<i32>()?;
                Ok(vec_i32.into_iter().map(|v| v as usize).collect())
            }
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to Vec<usize>",
                other
            ))),
        }
    }

    fn to_f32_vec(&self) -> Result<Vec<f32>, burn_tensor::DataError> {
        use burn_tensor::DType;
        match self.dtype {
            DType::F32 => self.to_vec::<f32>(),
            DType::F64 => {
                let vec = self.to_vec::<f64>()?;
                Ok(vec.into_iter().map(|v| v as f32).collect())
            }
            DType::F16 => {
                let vec = self.to_vec::<half::f16>()?;
                Ok(vec.into_iter().map(f32::from).collect())
            }
            DType::I64 => {
                let vec = self.to_vec::<i64>()?;
                Ok(vec.into_iter().map(|v| v as f32).collect())
            }
            DType::I32 => {
                let vec = self.to_vec::<i32>()?;
                Ok(vec.into_iter().map(|v| v as f32).collect())
            }
            DType::I8 => {
                let vec = self.to_vec::<i8>()?;
                Ok(vec.into_iter().map(|v| v as f32).collect())
            }
            DType::U8 => {
                let vec = self.to_vec::<u8>()?;
                Ok(vec.into_iter().map(|v| v as f32).collect())
            }
            DType::U16 => {
                let vec = self.to_vec::<u16>()?;
                Ok(vec.into_iter().map(|v| v as f32).collect())
            }
            other => Err(burn_tensor::DataError::TypeMismatch(format!(
                "Cannot convert {:?} to Vec<f32>",
                other
            ))),
        }
    }
}

/// ONNX graph representation
#[derive(Debug, Clone)]
pub struct OnnxGraph {
    /// The nodes of the graph.
    pub nodes: Vec<Node>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// Reference to GraphState to keep tensor data alive for .value() access
    /// This ensures Arguments can access tensor data via their data_id
    pub(crate) _graph_data: Option<std::rc::Rc<std::cell::RefCell<crate::graph_state::GraphState>>>,
}

/// Nodes produced by the ONNX parser
pub struct Node {
    /// The type of the node.
    /// This should be a valid ONNX operator.
    pub node_type: NodeType,

    /// The name of the node.
    pub name: String,

    /// The inputs of the node.
    pub inputs: Vec<Argument>,

    /// The outputs of the node.
    pub outputs: Vec<Argument>,

    /// ONNX attributes (opset-specific parameters)
    pub attrs: Attributes,

    /// Node-specific configuration (populated during processing)
    pub(crate) config: Option<Box<dyn NodeConfig>>,
}

impl Node {
    /// Get a reference to the node's configuration with automatic downcasting.
    /// Returns None if the config is not set or cannot be downcast to type T.
    ///
    /// # Example
    /// ```ignore
    /// if let Some(config) = node.get_config::<ArgMaxConfig>() {
    ///     // use config
    /// }
    /// ```
    pub fn get_config<T: NodeConfig + 'static>(&self) -> Option<&T> {
        self.config.as_ref()?.as_any().downcast_ref::<T>()
    }

    /// Get a reference to the node's configuration with automatic downcasting.
    /// Panics if the config is not set or cannot be downcast to type T.
    ///
    /// # Example
    /// ```ignore
    /// let config = node.config::<ArgMaxConfig>();
    /// // use config
    /// ```
    ///
    /// # Panics
    /// Panics if the config is not set or is the wrong type.
    pub fn config<T: NodeConfig + 'static>(&self) -> &T {
        self.get_config::<T>().unwrap_or_else(|| {
            panic!(
                "Node '{}' ({:?}) config is not set or has wrong type. Expected {}",
                self.name,
                self.node_type,
                std::any::type_name::<T>()
            )
        })
    }
}

// Custom Clone implementation since Box<dyn NodeConfig> doesn't auto-derive Clone
impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            node_type: self.node_type.clone(),
            name: self.name.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            attrs: self.attrs.clone(),
            config: self.config.as_ref().map(|c| c.clone_box()),
        }
    }
}

// Custom Debug implementation since Box<dyn NodeConfig> doesn't auto-derive Debug
impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("node_type", &self.node_type)
            .field("name", &self.name)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .field("attrs", &self.attrs)
            .field("config", &self.config.as_ref().map(|_| "Some(<config>)"))
            .finish()
    }
}

/// Supported ONNX operators (plus Burn-specific extensions for dimensional mapping)
///
/// See: <https://github.com/onnx/onnx/blob/main/docs/Operators.md>
#[derive(Debug, Hash, Eq, PartialEq, EnumString, Clone, Display)]
pub enum NodeType {
    Abs,
    Acos,
    Acosh,
    Add,
    And,
    ArgMax,
    ArgMin,
    Asin,
    Asinh,
    Atan,
    Atanh,
    Attention,
    AveragePool,
    AveragePool1d,
    AveragePool2d,
    BatchNormalization,
    Bernoulli,
    BitShift,
    BitwiseAnd,
    BitwiseNot,
    BitwiseOr,
    BitwiseXor,
    BlackmanWindow,
    Cast,
    CastLike,
    Ceil,
    Celu,
    CenterCropPad,
    Clip,
    Col,
    Compress,
    Concat,
    ConcatFromSequence,
    Constant,
    ConstantOfShape,
    Conv,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvInteger,
    ConvTranspose,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    Cos,
    Cosh,
    CumSum,
    DepthToSpace,
    DequantizeLinear,
    Det,
    DFT,
    Div,
    Dropout,
    DynamicQuantizeLinear,
    Einsum,
    Elu,
    Equal,
    Erf,
    Exp,
    Expand,
    EyeLike,
    Flatten,
    Floor,
    Gather,
    GatherElements,
    GatherND,
    Gelu,
    Gemm,
    GlobalAveragePool,
    GlobalLpPool,
    GlobalMaxPool,
    Greater,
    GreaterOrEqual,
    GridSample,
    GroupNormalization,
    GRU,
    HammingWindow,
    HannWindow,
    Hardmax,
    HardSigmoid,
    HardSwish,
    Identity,
    If,
    Im,
    InstanceNormalization,
    IsInf,
    IsNaN,
    LayerNormalization,
    LeakyRelu,
    Less,
    LessOrEqual,
    Linear,
    Log,
    LogSoftmax,
    Loop,
    LpNormalization,
    LpPool,
    LRN,
    LSTM,
    MatMul,
    MatMulInteger,
    Max,
    MaxPool,
    MaxPool1d,
    MaxPool2d,
    MaxRoiPool,
    MaxUnpool,
    Mean,
    MeanVarianceNormalization,
    MelWeightMatrix,
    Min,
    Mish,
    Mod,
    Mul,
    Multinomial,
    Neg,
    NegativeLogLikelihoodLoss,
    NonMaxSuppression,
    NonZero,
    Not,
    OneHot,
    Optional,
    OptionalGetElement,
    OptionalHasElement,
    Or,
    Pad,
    Pow,
    PRelu,
    QLinearConv,
    QLinearMatMul,
    QuantizeLinear,
    RandomNormal,
    RandomNormalLike,
    RandomUniform,
    RandomUniformLike,
    Range,
    Reciprocal,
    ReduceL1,
    ReduceL2,
    ReduceLogSum,
    ReduceLogSumExp,
    ReduceMax,
    ReduceMean,
    ReduceMin,
    ReduceProd,
    ReduceSum,
    ReduceSumSquare,
    Relu,
    Reshape,
    Resize,
    ReverseSequence,
    RNN,
    RoiAlign,
    Round,
    Scan,
    Scatter,
    ScatterElements,
    ScatterND,
    Selu,
    SequenceAt,
    SequenceConstruct,
    SequenceEmpty,
    SequenceErase,
    SequenceInsert,
    SequenceLength,
    SequenceMap,
    Shape,
    Shrink,
    Sigmoid,
    Sign,
    Sin,
    Sinh,
    Size,
    Slice,
    Softmax,
    SoftmaxCrossEntropyLoss,
    Softplus,
    Softsign,
    SpaceToDepth,
    Split,
    SplitToSequence,
    Sqrt,
    Squeeze,
    STFT,
    StringNormalizer,
    Sub,
    Sum,
    Tan,
    Tanh,
    TfIdfVectorizer,
    ThresholdedRelu,
    Tile,
    TopK,
    Transpose,
    Trilu,
    Unique,
    Unsqueeze,
    Upsample,
    Where,
    Xor,
}

impl AttributeValue {
    pub fn into_f32(self) -> f32 {
        if let AttributeValue::Float32(elem) = self {
            elem
        } else {
            panic!("Expected Float32, got {self:?}");
        }
    }

    pub fn into_i32(self) -> i32 {
        if let AttributeValue::Int64(elem) = self {
            elem as i32
        } else {
            panic!("Expected Int32, got {self:?}");
        }
    }

    pub fn into_i64(self) -> i64 {
        if let AttributeValue::Int64(elem) = self {
            elem
        } else {
            panic!("Expected Int64, got {self:?}");
        }
    }

    pub fn into_string(self) -> String {
        if let AttributeValue::String(elem) = self {
            elem
        } else {
            panic!("Expected String, got {self:?}");
        }
    }

    pub fn into_tensor(self) -> TensorData {
        if let AttributeValue::Tensor(elem) = self {
            elem
        } else {
            panic!("Expected Tensor, got {self:?}");
        }
    }

    pub fn into_f32s(self) -> Vec<f32> {
        if let AttributeValue::Float32s(elem) = self {
            elem
        } else {
            panic!("Expected Float32s, got {self:?}");
        }
    }

    pub fn into_i64s(self) -> Vec<i64> {
        if let AttributeValue::Int64s(elem) = self {
            elem
        } else {
            panic!("Expected Int64s, got {self:?}");
        }
    }

    pub fn into_strings(self) -> Vec<String> {
        if let AttributeValue::Strings(elem) = self {
            elem
        } else {
            panic!("Expected Strings, got {self:?}");
        }
    }

    pub fn into_tensors(self) -> Vec<TensorData> {
        if let AttributeValue::Tensors(elem) = self {
            elem
        } else {
            panic!("Expected Tensors, got {self:?}");
        }
    }
}

/// Convert AttributeValue to an Argument
impl From<AttributeValue> for Argument {
    fn from(attr: AttributeValue) -> Argument {
        let name = "".to_string();

        match attr {
            AttributeValue::Float32(_value) => Argument {
                ty: ArgType::Scalar(DType::F32),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Float32s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    dtype: DType::F32,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Int64(_value) => Argument {
                ty: ArgType::Scalar(DType::I64),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Int64s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    dtype: DType::I64,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::String(_value) => {
                panic!(
                    "String type not supported in DType - use AttributeValue directly for strings"
                )
            }
            AttributeValue::Strings(_values) => {
                panic!(
                    "String type not supported in DType - use AttributeValue directly for strings"
                )
            }
            AttributeValue::Tensor(tensor) => {
                if tensor.shape.is_empty() {
                    // Handle scalar tensors by converting them to scalar arguments
                    Argument {
                        ty: ArgType::Scalar(tensor.elem_type()),
                        name,
                        data_id: None,
                        value_source: ValueSource::Optional,
                        value_store: None,
                    }
                } else {
                    // Convert tensor to argument
                    Argument {
                        ty: ArgType::Tensor(TensorType {
                            rank: tensor.shape.len(),
                            dtype: tensor.elem_type(),
                            static_shape: Some(tensor.shape.to_vec()),
                        }),
                        name,
                        data_id: None,
                        value_source: ValueSource::Optional,
                        value_store: None,
                    }
                }
            }
            _ => panic!("Unsupported attribute type"),
        }
    }
}
