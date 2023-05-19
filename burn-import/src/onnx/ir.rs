use core::fmt;
use half::f16;
use std::{collections::HashMap, fmt::Formatter};
use strum_macros::{Display, EnumString};

pub type Shape = Vec<usize>;

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: String,
    pub arg_type: Option<ArgType>,
}

#[derive(Debug, Clone)]
pub enum ArgType {
    Tensor(Tensor),
}

#[derive(Debug, Clone)]
pub struct SparseTensor(Tensor, Tensor, Shape);

#[derive(Debug, Clone)]
pub enum AttributeValue {
    Float32(f32),
    Int64(i64),
    String(String),
    Tensor(Tensor),
    SparseTensor(SparseTensor),
    Float32s(Vec<f32>),
    Int64s(Vec<i64>),
    Strings(Vec<String>),
    Tensors(Vec<Tensor>),
    SparseTensors(Vec<SparseTensor>),
}
pub type Attributes = HashMap<String, AttributeValue>;

#[derive(Debug, Clone)]
pub enum ElementType {
    Float32,
    Float64,
    Int32,
    Int64,
    String,
    Float16,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: Option<String>,
    pub elem_type: ElementType,
    pub shape: Shape,
    pub data: Option<TensorData>,
}

#[derive(Clone)]
pub enum TensorData {
    Float16(Vec<f16>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    String(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct ONNXGraph {
    pub nodes: Vec<Node>,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub initializers: Vec<Argument>,
    pub old_node_names: HashMap<String, String>,
    pub old_input_names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub initializers: Vec<Argument>,
    pub attrs: Attributes,
    pub is_stateful: bool,
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

/// The list of supported node types (ONNX operators and some extra ones to map easily to Burn's ops)
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
    AveragePool,
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
    ConvInteger,
    ConvTranspose,
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
    ReduceL,
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
fn trunc<T: fmt::Display>(v: &Vec<T>) -> String {
    const BEGIN_INDEX: usize = 0;
    const MAX_LEN: usize = 5;
    let mut s = String::new();
    s.push('[');
    for (i, item) in v.iter().enumerate() {
        if i > BEGIN_INDEX {
            s.push_str(", ");
        }
        s.push_str(&format!("{}", item));
        if i > MAX_LEN {
            s.push_str(", ...");
            break;
        }
    }
    s.push(']');
    s
}

/// Shorten the tensor data for debug display
impl fmt::Debug for TensorData {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            TensorData::Float16(v) => write!(f, "Float16({})", trunc(v)),
            TensorData::Float32(v) => write!(f, "Float32({})", trunc(v)),
            TensorData::Float64(v) => write!(f, "Float64({})", trunc(v)),
            TensorData::Int32(v) => write!(f, "Int32({})", trunc(v)),
            TensorData::Int64(v) => write!(f, "Int64({})", trunc(v)),
            TensorData::String(v) => write!(f, "String({})", trunc(v)),
        }
    }
}
