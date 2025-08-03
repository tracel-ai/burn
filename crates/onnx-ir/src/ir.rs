use core::fmt;
use half::f16;
use std::{collections::HashMap, fmt::Formatter};
use strum::{Display, EnumString};

use crate::protos::TensorProto;

pub type Rank = usize;
pub type Shape = Vec<usize>;

/// A node input or output.
#[derive(Debug, Clone)]
pub struct Argument {
    /// The name of the node input.
    pub name: String,

    /// The type of the argument.
    pub ty: ArgType,

    /// The data of the argument.
    pub value: Option<TensorData>,

    /// True if the argument is passed to node, false otherwise. We use it mainly for informational purposes.
    /// The argument should contain a value if passed is false.
    pub passed: bool,
}

impl Argument {
    /// Copy everything except the name from the other argument
    pub fn copy_value(&mut self, other_arg: &Argument) {
        self.ty = other_arg.ty.clone();
        self.value.clone_from(&other_arg.value);
    }

    pub fn from_initializer(initializer: &TensorProto) -> Argument {
        let name = initializer.name.clone();
        let tensor_data = TensorData::try_from(initializer.clone())
            .unwrap_or_else(|_| panic!("invalid tensor {}", &initializer.name));

        if tensor_data.shape.is_empty() {
            // Convert zero dim tensor to scalar
            Self {
                name,
                ty: ArgType::Scalar(tensor_data.elem_type()),
                value: Some(tensor_data),
                passed: false,
            }
        } else {
            Self {
                name,
                ty: ArgType::Tensor(TensorType {
                    elem_type: tensor_data.elem_type(),
                    rank: tensor_data.shape.len(),
                    static_shape: Some(tensor_data.shape.clone()),
                }),
                value: Some(tensor_data),
                passed: false,
            }
        }
    }
}

/// The type of an argument.
#[derive(Debug, Clone)]
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
}

/// Represents the type of a tensor.
#[derive(Debug, Clone, Default)]
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
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
    pub fn is_tensor(&self) -> bool {
        matches!(self, Self::Tensor(_))
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
}

impl Argument {
    pub fn new(name: String) -> Self {
        Self {
            name,
            ty: ArgType::default(),
            value: None,
            passed: false,
        }
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
}

/// Nodes produced by the ONNX parser
#[derive(Debug, Clone)]
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
    pub attrs: Attributes,
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
        if let Data::Bool(elem) = self {
            elem
        } else {
            panic!("Expected Bool, got {self:?}");
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
            AttributeValue::Float32(value) => Argument {
                ty: ArgType::Scalar(ElementType::Float32),
                name,
                value: Some(TensorData {
                    shape: vec![],
                    data: Data::Float32(value),
                }),
                passed: false,
            },
            AttributeValue::Float32s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    elem_type: ElementType::Float32,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                value: Some(TensorData {
                    shape: vec![values.len()],
                    data: Data::Float32s(values),
                }),
                passed: false,
            },
            AttributeValue::Int64(value) => Argument {
                ty: ArgType::Scalar(ElementType::Int64),
                name,
                value: Some(TensorData {
                    shape: vec![],
                    data: Data::Int64(value),
                }),
                passed: false,
            },
            AttributeValue::Int64s(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    elem_type: ElementType::Int64,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                value: Some(TensorData {
                    shape: vec![values.len()],
                    data: Data::Int64s(values),
                }),
                passed: false,
            },
            AttributeValue::String(value) => Argument {
                ty: ArgType::Scalar(ElementType::String),
                name,
                value: Some(TensorData {
                    shape: vec![],
                    data: Data::String(value),
                }),
                passed: false,
            },
            AttributeValue::Strings(values) => Argument {
                ty: ArgType::Tensor(TensorType {
                    rank: 1,
                    elem_type: ElementType::String,
                    static_shape: Some(vec![values.len()]),
                }),
                name,
                value: Some(TensorData {
                    shape: vec![values.len()],
                    data: Data::Strings(values),
                }),
                passed: false,
            },
            AttributeValue::Tensor(tensor) => {
                if tensor.shape.is_empty() {
                    // Handle scalar tensors by converting them to scalar arguments
                    Argument {
                        ty: ArgType::Scalar(tensor.elem_type()),
                        name,
                        value: Some(TensorData {
                            shape: vec![],
                            data: tensor.data,
                        }),
                        passed: false,
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
                        value: Some(TensorData {
                            shape: tensor.shape,
                            data: tensor.data,
                        }),
                        passed: false,
                    }
                }
            }
            _ => panic!("Unsupported attribute type"),
        }
    }
}

impl Argument {
    pub fn into_tensor(self) -> Option<TensorData> {
        if let ArgType::Tensor(_) = self.ty {
            self.value
        } else {
            None
        }
    }
}
