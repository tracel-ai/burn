use burn_ndarray::NdArrayBackend;
use core::fmt;
use half::f16;
use std::{collections::HashMap, fmt::Formatter};
use strum_macros::{Display, EnumString};

pub type Shape = Vec<usize>;

#[derive(Debug, Clone)]
pub struct Argument {
    pub name: String,
    pub ty: ArgType,
}

#[derive(Debug, Clone)]
pub enum ArgType {
    Scalar(ElementType),
    Shape(usize),
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
    Bool,
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub elem_type: ElementType,
    pub dim: usize,
    pub data: Option<TensorData>,
    pub shape: Option<Shape>,
}

impl Default for Tensor {
    fn default() -> Self {
        Self {
            elem_type: ElementType::Float32,
            dim: 0,
            data: None,
            shape: None,
        }
    }
}

#[derive(Clone)]
pub enum TensorData {
    Float16(Vec<f16>),
    Float32(Vec<f32>),
    Float64(Vec<f64>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    String(Vec<String>),
    Bool(Vec<bool>),
}

/// ONNX graph representation
#[derive(Debug, Clone)]
pub struct ONNXGraph {
    /// The nodes of the graph.
    pub nodes: Vec<Node>,

    /// The inputs of the graph.
    pub inputs: Vec<Argument>,

    /// The outputs of the graph.
    pub outputs: Vec<Argument>,

    /// The original node names.
    pub old_node_names: HashMap<String, String>,

    /// The original input names.
    pub old_input_names: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct State {
    pub name: String,
    pub ty: StateType,
}

#[derive(Debug, Clone)]
pub enum StateType {
    Tensor(Tensor),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub node_type: NodeType,
    pub name: String,
    pub inputs: Vec<Argument>,
    pub outputs: Vec<Argument>,
    pub states: Vec<State>,
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
            TensorData::Bool(v) => write!(f, "Bool({})", trunc(v)),
        }
    }
}

/// Convert itermediate representation of tensor into a burn tensor
impl<const D: usize> TryFrom<&Tensor> for burn::tensor::Tensor<NdArrayBackend<f32>, D> {
    type Error = ();

    fn try_from(value: &Tensor) -> Result<Self, Self::Error> {
        let shape: [usize; D] = value.shape.clone().unwrap().try_into().unwrap();
        let TensorData::Float32(floats) = value.data.clone().unwrap() else {
            todo!("Tensor data must be float32s");
        };

        Ok(burn::tensor::Tensor::from_data(floats.as_slice()).reshape(shape))
    }
}
