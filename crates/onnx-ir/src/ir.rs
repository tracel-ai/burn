use core::fmt;
use half::f16;
use std::{any::Any, cell::RefCell, collections::HashMap, fmt::Formatter, rc::Rc};
use strum::{Display, EnumString};

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
    Scalar(ElementType),
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

/// The type of an element.
#[derive(Debug, Clone, PartialEq)]
pub enum ElementType {
    Float32,
    Float64,
    Int32,
    Int64,
    String,
    Float16,
    Bool,
    Uint16,
    Uint8,
    Int8,
}

/// Represents the type of a tensor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorType {
    /// The element type of the tensor values (e.g. Float32, Int64, etc.)
    pub elem_type: ElementType,

    /// The number of dimensions in the tensor
    pub rank: Rank,

    /// The static shape information of the tensor determined during shape inference
    pub static_shape: Option<Vec<usize>>, // TODO fill in with inferred shape information
}

impl Default for ElementType {
    fn default() -> Self {
        Self::Float32
    }
}

impl Default for ArgType {
    fn default() -> Self {
        Self::Tensor(TensorType::default())
    }
}

impl ArgType {
    /// Check if this argument is a scalar
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
    /// Check if this argument is a tensor
    pub fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor(_))
    }
    /// Check if this argument is a shape
    pub fn is_shape(&self) -> bool {
        matches!(self, Self::Shape(_))
    }

    /// returns the rank (dimension) of the Arg
    pub fn rank(&self) -> usize {
        match self {
            ArgType::Scalar(_) => 0,
            ArgType::Shape(_) => 1,
            ArgType::Tensor(t) => t.rank,
        }
    }

    /// returns the element type of the Arg
    pub fn elem_type(&self) -> &ElementType {
        match self {
            ArgType::Scalar(s) => s,
            ArgType::Shape(_) => panic!("ArgType::Shape has no ElementType"),
            ArgType::Tensor(t) => &t.elem_type,
        }
    }

    /// returns the static shape if available
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

    /// Get the constant value for this argument by ID from central store
    /// Returns None if this argument has no data_id or data not found
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

    /// Check if this argument is a static constant (embedded value)
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

/// Representation of a tensor with data and shape information.
///
/// A tensor is a multi-dimensional array of data with a specific shape.
/// This struct stores both the raw data values and the dimensional information
/// that defines the tensor's structure.
#[derive(Debug, Clone)]
pub struct TensorData {
    /// The data values of the tensor.
    pub data: Data,

    /// The dimensional shape of the tensor.
    pub shape: Shape,
}

impl TensorData {
    /// The element type of the tensor inferred from the data.
    pub fn elem_type(&self) -> ElementType {
        match &self.data {
            Data::Bool(_) | Data::Bools(_) => ElementType::Bool,
            Data::Float16(_) | Data::Float16s(_) => ElementType::Float16,
            Data::Float32(_) | Data::Float32s(_) => ElementType::Float32,
            Data::Float64(_) | Data::Float64s(_) => ElementType::Float64,
            Data::Uint16(_) | Data::Uint16s(_) => ElementType::Uint16,
            Data::Uint8(_) | Data::Uint8s(_) => ElementType::Uint8,
            Data::Int8(_) | Data::Int8s(_) => ElementType::Int8,
            Data::Int32(_) | Data::Int32s(_) => ElementType::Int32,
            Data::Int64(_) | Data::Int64s(_) => ElementType::Int64,
            Data::String(_) | Data::Strings(_) => ElementType::String,
        }
    }
}

/// Container to hold data for tensors and arguments
#[derive(Clone)]
pub enum Data {
    Bool(bool),
    Bools(Vec<bool>),
    Float16(f16),
    Float16s(Vec<f16>),
    Float32(f32),
    Float32s(Vec<f32>),
    Float64(f64),
    Float64s(Vec<f64>),
    Uint16(u16),
    Uint16s(Vec<u16>),
    Uint8(u8),
    Uint8s(Vec<u8>),
    Int8(i8),
    Int8s(Vec<i8>),
    Int32(i32),
    Int32s(Vec<i32>),
    Int64(i64),
    Int64s(Vec<i64>),
    String(String),
    Strings(Vec<String>),
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

    /// The attributes of the node.
    pub attrs: Attributes, // TODO make crate level

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

// Required by topological sort
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.node_type == other.node_type
    }
}

// Required by topological sort
impl Eq for Node {}

// Required by topological sort
impl core::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
        self.node_type.hash(state);
        self.inputs.hash(state);
        self.outputs.hash(state);
    }
}

// Required by topological sort
impl core::hash::Hash for Argument {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl Eq for Argument {}

// Required by HashSet
impl PartialEq for Argument {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

/// The list of supported node types (ONNX operators and some extra ones to map easily to Burn's ops)
/// Refer: <https://github.com/onnx/onnx/blob/main/docs/Operators.md>
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

/// Truncate the vector display for debug display
fn trunc<T: fmt::Display>(v: &[T]) -> String {
    const BEGIN_INDEX: usize = 0;
    const MAX_LEN: usize = 5;
    let mut s = String::new();
    s.push('[');
    for (i, item) in v.iter().enumerate() {
        if i > BEGIN_INDEX {
            s.push_str(", ");
        }
        s.push_str(&format!("{item}"));
        if i > MAX_LEN {
            s.push_str(", ...");
            break;
        }
    }
    s.push(']');
    s
}

/// Shorten the tensor data for debug display
impl fmt::Debug for Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Data::Float16s(v) => write!(f, "Float16s({})", trunc(v)),
            Data::Float32s(v) => write!(f, "Float32s({})", trunc(v)),
            Data::Float64s(v) => write!(f, "Float64s({})", trunc(v)),
            Data::Int32s(v) => write!(f, "Int32s({})", trunc(v)),
            Data::Int64s(v) => write!(f, "Int64s({})", trunc(v)),
            Data::Strings(v) => write!(f, "Strings({})", trunc(v)),
            Data::Bools(v) => write!(f, "Bools({})", trunc(v)),
            Data::Float16(v) => write!(f, "Float16({v})"),
            Data::Float32(v) => write!(f, "Float32({v})"),
            Data::Float64(v) => write!(f, "Float64({v})"),
            Data::Uint16(v) => write!(f, "Uint16({v})"),
            Data::Uint16s(v) => write!(f, "Uint16s({})", trunc(v)),
            Data::Uint8s(v) => write!(f, "Uint8s({})", trunc(v)),
            Data::Int8s(v) => write!(f, "Int8s({})", trunc(v)),
            Data::Uint8(v) => write!(f, "Uint8({v})"),
            Data::Int8(v) => write!(f, "Int8({v})"),
            Data::Int32(v) => write!(f, "Int32({v})"),
            Data::Int64(v) => write!(f, "Int64({v})"),
            Data::String(v) => write!(f, "String({v})"),
            Data::Bool(v) => write!(f, "Bool({v})"),
        }
    }
}

impl Data {
    pub fn into_scalar(self) -> Self {
        match self {
            Data::Float16s(data) => {
                assert_eq!(data.len(), 1);
                Data::Float16(data[0])
            }
            Data::Float32s(data) => {
                assert_eq!(data.len(), 1);
                Data::Float32(data[0])
            }
            Data::Float64s(data) => {
                assert_eq!(data.len(), 1);
                Data::Float64(data[0])
            }
            Data::Int32s(data) => {
                assert_eq!(data.len(), 1);
                Data::Int32(data[0])
            }
            Data::Int64s(data) => {
                assert_eq!(data.len(), 1);
                Data::Int64(data[0])
            }
            Data::Bools(data) => {
                assert_eq!(data.len(), 1);
                Data::Bool(data[0])
            }
            Data::Strings(data) => {
                assert_eq!(data.len(), 1);
                Data::String(data[0].clone())
            }
            _ => self,
        }
    }
    pub fn into_f16(self) -> f16 {
        match self {
            Data::Float16(elem) => elem,
            Data::Float32(elem) => f16::from_f32(elem),
            Data::Float64(elem) => f16::from_f64(elem),
            _ => panic!("Cannot convert {self:?} to f16"),
        }
    }

    pub fn into_f32(self) -> f32 {
        match self {
            Data::Float16(elem) => elem.to_f32(),
            Data::Float32(elem) => elem,
            Data::Float64(elem) => elem as f32,
            Data::Int32(elem) => elem as f32,
            Data::Int64(elem) => elem as f32,
            Data::Float32s(elem) if elem.len() == 1 => elem[0],
            _ => panic!("Cannot convert {self:?} to f32"),
        }
    }

    pub fn into_f64(self) -> f64 {
        match self {
            Data::Float16(elem) => elem.to_f64(),
            Data::Float32(elem) => elem as f64,
            Data::Float64(elem) => elem,
            Data::Int32(elem) => elem as f64,
            Data::Int64(elem) => elem as f64,
            Data::Float64s(elem) if elem.len() == 1 => elem[0],
            _ => panic!("Cannot convert {self:?} to f64"),
        }
    }

    pub fn into_i32(self) -> i32 {
        match self {
            Data::Int32(elem) => elem,
            Data::Int64(elem) => elem as i32,
            Data::Float32(elem) => elem as i32,
            Data::Float64(elem) => elem as i32,
            Data::Float32s(elem) if elem.len() == 1 => elem[0] as i32,
            Data::Int32s(elem) if elem.len() == 1 => elem[0],
            Data::Uint8(v) => v as i32,
            Data::Int8(v) => v as i32,
            _ => panic!("Cannot convert {self:?} to i32"),
        }
    }

    pub fn into_i64(self) -> i64 {
        match self {
            Data::Int32(elem) => elem as i64,
            Data::Int64(elem) => elem,
            Data::Float32(elem) => elem as i64,
            Data::Float64(elem) => elem as i64,
            Data::Int64s(elem) if elem.len() == 1 => elem[0],
            _ => panic!("Cannot convert {self:?} to i64"),
        }
    }

    pub fn into_bool(self) -> bool {
        match self {
            Data::Bool(elem) => elem,
            Data::Bools(elem) if elem.len() == 1 => elem[0],
            _ => panic!("Expected Bool, got {self:?}"),
        }
    }

    pub fn into_string(self) -> String {
        if let Data::String(elem) = self {
            elem
        } else {
            panic!("Expected String, got {self:?}");
        }
    }

    pub fn into_f16s(self) -> Vec<f16> {
        match self {
            Data::Float16s(elem) => elem,
            Data::Float32s(elem) => elem.into_iter().map(f16::from_f32).collect(),
            Data::Float64s(elem) => elem.into_iter().map(f16::from_f64).collect(),
            _ => panic!("Cannot convert {self:?} to Vec<f16>"),
        }
    }

    pub fn into_f32s(self) -> Vec<f32> {
        match self {
            Data::Float16s(elem) => elem.into_iter().map(|x| x.to_f32()).collect(),
            Data::Float32s(elem) => elem,
            Data::Float64s(elem) => elem.into_iter().map(|x| x as f32).collect(),
            Data::Int32s(elem) => elem.into_iter().map(|x| x as f32).collect(),
            Data::Int64s(elem) => elem.into_iter().map(|x| x as f32).collect(),
            Data::Uint8s(v) => v.into_iter().map(|x| x as f32).collect(),
            Data::Int8s(v) => v.into_iter().map(|x| x as f32).collect(),
            _ => panic!("Cannot convert {self:?} to Vec<f32>"),
        }
    }

    pub fn into_f64s(self) -> Vec<f64> {
        match self {
            Data::Float16s(elem) => elem.into_iter().map(|x| x.to_f64()).collect(),
            Data::Float32s(elem) => elem.into_iter().map(|x| x as f64).collect(),
            Data::Float64s(elem) => elem,
            Data::Int32s(elem) => elem.into_iter().map(|x| x as f64).collect(),
            Data::Int64s(elem) => elem.into_iter().map(|x| x as f64).collect(),
            _ => panic!("Cannot convert {self:?} to Vec<f64>"),
        }
    }

    pub fn into_i32s(self) -> Vec<i32> {
        match self {
            Data::Int32s(elem) => elem,
            Data::Int64s(elem) => elem.into_iter().map(|x| x as i32).collect(),
            Data::Float32s(elem) => elem.into_iter().map(|x| x as i32).collect(),
            Data::Float64s(elem) => elem.into_iter().map(|x| x as i32).collect(),
            Data::Uint8s(v) => v.into_iter().map(|x| x as i32).collect(),
            Data::Int8s(v) => v.into_iter().map(|x| x as i32).collect(),
            _ => panic!("Cannot convert {self:?} to Vec<i32>"),
        }
    }

    pub fn into_i64s(self) -> Vec<i64> {
        match self {
            Data::Int32s(elem) => elem.into_iter().map(|x| x as i64).collect(),
            Data::Int64s(elem) => elem,
            Data::Float32s(elem) => elem.into_iter().map(|x| x as i64).collect(),
            Data::Float64s(elem) => elem.into_iter().map(|x| x as i64).collect(),
            _ => panic!("Cannot convert {self:?} to Vec<i64>"),
        }
    }

    pub fn into_usizes(self) -> Vec<usize> {
        match self {
            Data::Int32s(elem) => elem.into_iter().map(|x| x as usize).collect(),
            Data::Int64s(elem) => elem.into_iter().map(|x| x as usize).collect(),
            Data::Float32s(elem) => elem.into_iter().map(|x| x as usize).collect(),
            Data::Float64s(elem) => elem.into_iter().map(|x| x as usize).collect(),
            _ => panic!("Cannot convert {self:?} to Vec<usize>"),
        }
    }

    pub fn into_bools(self) -> Vec<bool> {
        if let Data::Bools(elem) = self {
            elem
        } else {
            panic!("Expected Bools, got {self:?}");
        }
    }

    pub fn into_strings(self) -> Vec<String> {
        if let Data::Strings(elem) = self {
            elem
        } else {
            panic!("Expected Strings, got {self:?}");
        }
    }
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
        // "" is used as a placeholder for the name
        // TODO dt review this empty string placeholder; it came up a few times in the issues
        let name = "".to_string();

        match attr {
            AttributeValue::Float32(_value) => Argument {
                ty: ArgType::Scalar(ElementType::Float32),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Float32s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    elem_type: ElementType::Float32,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Int64(_value) => Argument {
                ty: ArgType::Scalar(ElementType::Int64),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Int64s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    elem_type: ElementType::Int64,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::String(_value) => Argument {
                ty: ArgType::Scalar(ElementType::String),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
            AttributeValue::Strings(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    elem_type: ElementType::String,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                data_id: None,
                value_source: ValueSource::Optional,
                value_store: None,
            },
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
                            elem_type: tensor.elem_type(),
                            static_shape: Some(tensor.shape.clone()),
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
