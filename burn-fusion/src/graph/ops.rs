use crate::FusionBackend;
use crate::{HandleContainer, TensorDescription};
use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    Distribution, Element,
};
use std::ops::Range;

/// General trait to abstract how a single operation is executed.
pub trait Ops<B: FusionBackend>: Send + Sync {
    /// Execute the operation.
    fn execute(self: Box<Self>, handles: &mut HandleContainer<B>);
}

/// Describe all tensor operations possible.
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum TensorOpsDescription {
    /// Basic operation on a float tensor.
    BaseOpsFloat(BaseOpsDescription),
    /// Basic operation on an int tensor.
    BaseOpsInt(BaseOpsDescription),
    /// Basic operation on a bool tensor.
    BaseOpsBool(BaseOpsDescription),
    /// Numeric operation on a float tensor.
    NumericOpsFloat(NumericOpsDescription<f32>),
    /// Numeric operation on an int tensor.
    NumericOpsInt(NumericOpsDescription<i32>),
    /// Operation specific to a bool tensor.
    BoolOps(BoolOpsDescription),
    /// Operation specific to an int tensor.
    IntOps(IntOpsDescription),
    /// Operation specific to a float tensor.
    FloatOps(FloatOpsDescription),
    /// Module operation.
    ModuleOps(ModuleOpsDescription),
}

/// Operation description specific to a float tensor.
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum FloatOpsDescription {
    /// Operation corresponding to [exp](burn_tensor::ops::TensorOps::exp).
    Exp(UnaryOpsDescription),
    /// Operation corresponding to [log](burn_tensor::ops::TensorOps::log).
    Log(UnaryOpsDescription),
    /// Operation corresponding to [log1p](burn_tensor::ops::TensorOps::log1p).
    Log1p(UnaryOpsDescription),
    /// Operation corresponding to [erf](burn_tensor::ops::TensorOps::erf).
    Erf(UnaryOpsDescription),
    /// Operation corresponding to [powf](burn_tensor::ops::TensorOps::powf).
    Powf(ScalarOpsDescription<f32>),
    /// Operation corresponding to [sqrt](burn_tensor::ops::TensorOps::sqrt).
    Sqrt(UnaryOpsDescription),
    /// Operation corresponding to [cos](burn_tensor::ops::TensorOps::cos).
    Cos(UnaryOpsDescription),
    /// Operation corresponding to [sin](burn_tensor::ops::TensorOps::sin).
    Sin(UnaryOpsDescription),
    /// Operation corresponding to [tanh](burn_tensor::ops::TensorOps::tanh).
    Tanh(UnaryOpsDescription),
    /// Operation corresponding to [into_int](burn_tensor::ops::TensorOps::into_int).
    IntoInt(UnaryOpsDescription),
    /// Operation corresponding to [matmul](burn_tensor::ops::TensorOps::matmul).
    Matmul(BinaryOpsDescription),
    /// Operation corresponding to [random](burn_tensor::ops::TensorOps::random).
    Random(RandomOpsDescription),
    /// Operation corresponding to [recip](burn_tensor::ops::TensorOps::recip).
    Recip(UnaryOpsDescription),
}

/// Operation description specific to module.
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum ModuleOpsDescription {
    /// Operation corresponding to [embedding](burn_tensor::ops::ModuleOps::embedding).
    Embedding(EmbeddingDescription),
    /// Operation corresponding to [embedding_backward](burn_tensor::ops::ModuleOps::embedding_backward).
    EmbeddingBackward(EmbeddingBackwardDescription),
    /// Operation corresponding to [conv1d](burn_tensor::ops::ModuleOps::conv1d).
    Conv1d(Conv1dDescription),
    /// Operation corresponding to [conv2d](burn_tensor::ops::ModuleOps::conv2d).
    Conv2d(Conv2dDescription),
    /// Operation corresponding to [conv transpose 1d](burn_tensor::ops::ModuleOps::conv_transpose1d).
    ConvTranspose1d(ConvTranspose1dDescription),
    /// Operation corresponding to [conv transpose 2d](burn_tensor::ops::ModuleOps::conv_transpose2d).
    ConvTranspose2d(ConvTranspose2dDescription),
    /// Operation corresponding to [avg pool 1d](burn_tensor::ops::ModuleOps::avg_pool1d).
    AvgPool1d(AvgPool1dDescription),
    /// Operation corresponding to [avg pool 2d](burn_tensor::ops::ModuleOps::avg_pool2d).
    AvgPool2d(AvgPool2dDescription),
    /// Operation corresponding to
    /// [avg pool 1d backward](burn_tensor::ops::ModuleOps::avg_pool1d_backward).
    AvgPool1dBackward(AvgPool1dBackwardDescription),
    /// Operation corresponding to
    /// [avg pool 2d backward](burn_tensor::ops::ModuleOps::avg_pool2d_backward).
    AvgPool2dBackward(AvgPool2dBackwardDescription),
    /// Operation corresponding to
    /// [adaptive avg pool 1d](burn_tensor::ops::ModuleOps::adaptive_avg_pool1d).
    AdaptiveAvgPool1d(AdaptiveAvgPool1dDescription),
    /// Operation corresponding to
    /// [adaptive avg pool 2d](burn_tensor::ops::ModuleOps::adaptive_avg_pool2d).
    AdaptiveAvgPool2d(AdaptiveAvgPool2dDescription),
    /// Operation corresponding to
    /// [adaptive avg pool 1d backward](burn_tensor::ops::ModuleOps::adaptive_avg_pool1d_backward).
    AdaptiveAvgPool1dBackward(AdaptiveAvgPool1dBackwardDescription),
    /// Operation corresponding to
    /// [adaptive avg pool 2d backward](burn_tensor::ops::ModuleOps::adaptive_avg_pool2d_backward).
    AdaptiveAvgPool2dBackward(AdaptiveAvgPool2dBackwardDescription),
    /// Operation corresponding to
    /// [max pool 1d](burn_tensor::ops::ModuleOps::max_pool1d).
    MaxPool1d(MaxPool1dDescription),
    /// Operation corresponding to
    /// [max pool 1d with indices](burn_tensor::ops::ModuleOps::max_pool1d_with_indices).
    MaxPool1dWithIndices(MaxPool1dWithIndicesDescription),
    /// Operation corresponding to
    /// [max pool 1d with indices backward](burn_tensor::ops::ModuleOps::max_pool1d_with_indices_backward).
    MaxPool1dWithIndicesBackward(MaxPool1dWithIndicesBackwardDescription),
    /// Operation corresponding to
    /// [max pool 2d](burn_tensor::ops::ModuleOps::max_pool1d).
    MaxPool2d(MaxPool2dDescription),
    /// Operation corresponding to
    /// [max pool 2d with indices](burn_tensor::ops::ModuleOps::max_pool2d_with_indices).
    MaxPool2dWithIndices(MaxPool2dWithIndicesDescription),
    /// Operation corresponding to
    /// [max pool 2d with indices backward](burn_tensor::ops::ModuleOps::max_pool2d_with_indices_backward).
    MaxPool2dWithIndicesBackward(MaxPool2dWithIndicesBackwardDescription),
}

/// Basic operations that can be done on any tensor type.
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum BaseOpsDescription {
    /// Operation corresponding to:
    ///
    /// Float => [to device](burn_tensor::ops::TensorOps::to_device).
    /// Int => [to device](burn_tensor::ops::IntTensorOps::int_to_device).
    /// Bool => [to device](burn_tensor::ops::BoolTensorOps::bool_to_device).
    ToDevice(TensorDescription),
    /// Operation corresponding to:
    ///
    /// Float => [reshape](burn_tensor::ops::TensorOps::reshape).
    /// Int => [reshape](burn_tensor::ops::IntTensorOps::int_reshape).
    /// Bool => [reshape](burn_tensor::ops::BoolTensorOps::bool_reshape).
    Reshape(ReshapeDescription),
    /// Operation corresponding to:
    ///
    /// Float => [swap_dims](burn_tensor::ops::TensorOps::swap_dims).
    /// Int => [swap_dims](burn_tensor::ops::IntTensorOps::int_swap_dims).
    /// Bool => [swap_dims](burn_tensor::ops::BoolTensorOps::bool_swap_dims).
    SwapDims(SwapDimsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [slice](burn_tensor::ops::TensorOps::slice).
    /// Int => [slice](burn_tensor::ops::IntTensorOps::int_slice).
    /// Bool => [slice](burn_tensor::ops::BoolTensorOps::bool_slice).
    Slice(SliceOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [slice assign](burn_tensor::ops::TensorOps::slice_assign).
    /// Int => [slice assign](burn_tensor::ops::IntTensorOps::int_slice_assign).
    /// Bool => [slice assign](burn_tensor::ops::BoolTensorOps::bool_slice_assign).
    SliceAssign(SliceAssignOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [equal](burn_tensor::ops::TensorOps::equal).
    /// Int => [equal](burn_tensor::ops::IntTensorOps::int_equal).
    /// Bool => [equal](burn_tensor::ops::BoolTensorOps::bool_equal).
    Equal(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [repeat](burn_tensor::ops::TensorOps::repeat).
    /// Int => [repeat](burn_tensor::ops::IntTensorOps::int_repeat).
    /// Bool => [repeat](burn_tensor::ops::BoolTensorOps::bool_repeat).
    Repeat(RepeatOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [cat](burn_tensor::ops::TensorOps::cat).
    /// Int => [cat](burn_tensor::ops::IntTensorOps::int_cat).
    /// Bool => [cat](burn_tensor::ops::BoolTensorOps::bool_cat).
    Cat(CatOpsDescription),
}

/// Numeric operations on int and float tensors.
#[derive(Clone, Debug, PartialEq)]
pub enum NumericOpsDescription<E> {
    /// Operation corresponding to:
    ///
    /// Float => [add](burn_tensor::ops::TensorOps::add).
    /// Int => [add](burn_tensor::ops::IntTensorOps::int_add).
    Add(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [add scalar](burn_tensor::ops::TensorOps::add_scalar).
    /// Int => [add scalar](burn_tensor::ops::IntTensorOps::int_add_scalar).
    AddScalar(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [sub](burn_tensor::ops::TensorOps::sub).
    /// Int => [sub](burn_tensor::ops::IntTensorOps::int_sub).
    Sub(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [sub scalar](burn_tensor::ops::TensorOps::sub_scalar).
    /// Int => [sub scalar](burn_tensor::ops::IntTensorOps::int_sub_scalar).
    SubScalar(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [div](burn_tensor::ops::TensorOps::div).
    /// Int => [div](burn_tensor::ops::IntTensorOps::int_div).
    Div(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [div scalar](burn_tensor::ops::TensorOps::div_scalar).
    /// Int => [div scalar](burn_tensor::ops::IntTensorOps::int_div_scalar).
    DivScalar(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mul](burn_tensor::ops::TensorOps::mul).
    /// Int => [mul](burn_tensor::ops::IntTensorOps::int_mul).
    Mul(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [mul scalar](burn_tensor::ops::TensorOps::mul_scalar).
    /// Int => [mul scalar](burn_tensor::ops::IntTensorOps::int_mul_scalar).
    MulScalar(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [abs](burn_tensor::ops::TensorOps::abs).
    /// Int => [abs](burn_tensor::ops::IntTensorOps::int_abs).
    Abs(UnaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [ones](burn_tensor::ops::TensorOps::ones).
    /// Int => [ones](burn_tensor::ops::IntTensorOps::int_ones).
    Ones(TensorDescription),
    /// Operation corresponding to:
    ///
    /// Float => [zeros](burn_tensor::ops::TensorOps::zeros).
    /// Int => [zeros](burn_tensor::ops::IntTensorOps::int_zeros).
    Zeros(TensorDescription),
    /// Operation corresponding to:
    ///
    /// Float => [full](burn_tensor::ops::TensorOps::full).
    /// Int => [full](burn_tensor::ops::IntTensorOps::int_full).
    Full((TensorDescription, E)),
    /// Operation corresponding to:
    ///
    /// Float => [gather](burn_tensor::ops::TensorOps::gather).
    /// Int => [gather](burn_tensor::ops::IntTensorOps::int_gather).
    Gather(GatherOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [scatter](burn_tensor::ops::TensorOps::scatter).
    /// Int => [scatter](burn_tensor::ops::IntTensorOps::int_scatter).
    Scatter(ScatterOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [select](burn_tensor::ops::TensorOps::select).
    /// Int => [select](burn_tensor::ops::IntTensorOps::int_select).
    Select(SelectOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [select assign](burn_tensor::ops::TensorOps::select_assign).
    /// Int => [select assign](burn_tensor::ops::IntTensorOps::int_select_assign).
    SelectAssign(SelectAssignOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [mask where](burn_tensor::ops::TensorOps::mask_where).
    /// Int => [mask where](burn_tensor::ops::IntTensorOps::int_mask_where).
    MaskWhere(MaskWhereOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [mask fill](burn_tensor::ops::TensorOps::mask_fill).
    /// Int => [mask fill](burn_tensor::ops::IntTensorOps::int_mask_fill).
    MaskFill(MaskFillOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mean dim](burn_tensor::ops::TensorOps::mean_dim).
    /// Int => [mean dim](burn_tensor::ops::IntTensorOps::int_mean_dim).
    MeanDim(ScalarOpsDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [mean](burn_tensor::ops::TensorOps::mean).
    /// Int => [mean](burn_tensor::ops::IntTensorOps::int_mean).
    Mean(UnaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [sum](burn_tensor::ops::TensorOps::sum).
    /// Int => [sum](burn_tensor::ops::IntTensorOps::int_sum).
    Sum(UnaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [sum dim](burn_tensor::ops::TensorOps::sum_dim).
    /// Int => [sum dim](burn_tensor::ops::IntTensorOps::int_sum_dim).
    SumDim(ScalarOpsDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [equal elem](burn_tensor::ops::TensorOps::equal_elem).
    /// Int => [equal elem](burn_tensor::ops::IntTensorOps::int_equal_elem).
    EqualElem(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater](burn_tensor::ops::TensorOps::greater).
    /// Int => [greater](burn_tensor::ops::IntTensorOps::int_greater).
    Greater(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [greater elem](burn_tensor::ops::TensorOps::greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterElem(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal](burn_tensor::ops::TensorOps::greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterEqual(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal elem](burn_tensor::ops::TensorOps::greater_equal_elem).
    /// Int => [greater equal elem](burn_tensor::ops::IntTensorOps::int_greater_equal_elem).
    GreaterEqualElem(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower](burn_tensor::ops::TensorOps::lower).
    /// Int => [lower](burn_tensor::ops::IntTensorOps::int_lower).
    Lower(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [lower elem](burn_tensor::ops::TensorOps::lower_elem).
    /// Int => [lower elem](burn_tensor::ops::IntTensorOps::int_lower_elem).
    LowerElem(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal](burn_tensor::ops::TensorOps::lower_equal).
    /// Int => [lower equal](burn_tensor::ops::IntTensorOps::int_lower_equal).
    LowerEqual(BinaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal elem](burn_tensor::ops::TensorOps::lower_equal_elem).
    /// Int => [lower equal elem](burn_tensor::ops::IntTensorOps::int_lower_equal_elem).
    LowerEqualElem(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [argmax](burn_tensor::ops::TensorOps::argmax).
    /// Int => [argmax](burn_tensor::ops::IntTensorOps::int_argmax).
    ArgMax(ScalarOpsDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [argmin](burn_tensor::ops::TensorOps::argmin).
    /// Int => [argmin](burn_tensor::ops::IntTensorOps::int_argmin).
    ArgMin(ScalarOpsDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [max](burn_tensor::ops::TensorOps::max).
    /// Int => [max](burn_tensor::ops::IntTensorOps::int_max).
    Max(UnaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [max dim with indices](burn_tensor::ops::TensorOps::max_dim_with_indices).
    /// Int => [max dim with indices](burn_tensor::ops::IntTensorOps::int_max_dim_with_indices).
    MaxDimWithIndices(ReduceDimWithIndicesDescription),
    /// Operation corresponding to:
    ///
    /// Float => [min dim with indices](burn_tensor::ops::TensorOps::min_dim_with_indices).
    /// Int => [min dim with indices](burn_tensor::ops::IntTensorOps::int_min_dim_with_indices).
    MinDimWithIndices(ReduceDimWithIndicesDescription),
    /// Operation corresponding to:
    ///
    /// Float => [min](burn_tensor::ops::TensorOps::min).
    /// Int => [min](burn_tensor::ops::IntTensorOps::int_min).
    Min(UnaryOpsDescription),
    /// Operation corresponding to:
    ///
    /// Float => [max dim](burn_tensor::ops::TensorOps::max_dim).
    /// Int => [max dim](burn_tensor::ops::IntTensorOps::int_max_dim).
    MaxDim(ScalarOpsDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [min dim](burn_tensor::ops::TensorOps::min_dim).
    /// Int => [min dim](burn_tensor::ops::IntTensorOps::int_min_dim).
    MinDim(ScalarOpsDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [clamp](burn_tensor::ops::TensorOps::clamp).
    /// Int => [clamp](burn_tensor::ops::IntTensorOps::int_clamp).
    Clamp(ClampOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [clamp max](burn_tensor::ops::TensorOps::clamp_max).
    /// Int => [clamp max](burn_tensor::ops::IntTensorOps::int_clamp_max).
    ClampMax(ScalarOpsDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [clamp min](burn_tensor::ops::TensorOps::clamp_min).
    /// Int => [cleamp min](burn_tensor::ops::IntTensorOps::int_clamp_min).
    ClampMin(ScalarOpsDescription<E>),
}

/// Operation description specific to an int tensor.
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum IntOpsDescription {
    /// Operation corresponding to [into float](burn_tensor::ops::IntTensorOps::int_into_float).
    IntoFloat(UnaryOpsDescription),
}

/// Operation description specific to a bool tensor.
#[derive(Clone, Debug, Hash, PartialEq)]
pub enum BoolOpsDescription {
    /// Operation corresponding to [into float](burn_tensor::ops::BoolTensorOps::bool_into_float).
    IntoFloat(UnaryOpsDescription),
    /// Operation corresponding to [into int](burn_tensor::ops::BoolTensorOps::bool_into_int).
    IntoInt(UnaryOpsDescription),
    /// Operation corresponding to [not](burn_tensor::ops::BoolTensorOps::bool_not).
    Not(UnaryOpsDescription),
}

#[derive(Clone, Debug, Hash, PartialEq)]
/// Swap dim operation description.
pub struct SwapDimsDescription {
    /// Input tensor description.
    pub input: TensorDescription,
    /// output tensor description.
    pub out: TensorDescription,
    /// The first dim to swap.
    pub dim1: usize,
    /// The second dim to swap.
    pub dim2: usize,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct RandomOpsDescription {
    pub out: TensorDescription,
    pub distribution: Distribution,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct ReshapeDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct BinaryOpsDescription {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct UnaryOpsDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct ScalarOpsDescription<E> {
    pub lhs: TensorDescription,
    pub rhs: E,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct GatherOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct ScatterOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct SelectOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct SelectAssignOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct SliceOpsDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct SliceAssignOpsDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct MaskWhereOpsDescription {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct MaskFillOpsDescription<E> {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: E,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct ClampOpsDescription<E> {
    pub tensor: TensorDescription,
    pub min: E,
    pub max: E,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct RepeatOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub times: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct CatOpsDescription {
    pub tensors: Vec<TensorDescription>,
    pub dim: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct ReduceDimWithIndicesDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct EmbeddingDescription {
    pub weights: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct EmbeddingBackwardDescription {
    pub weights: TensorDescription,
    pub out_grad: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct Conv1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvOptions<1>,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct Conv2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvOptions<2>,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct ConvTranspose1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTransposeOptions<1>,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct ConvTranspose2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTransposeOptions<2>,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AvgPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AvgPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AvgPool1dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AvgPool2dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dDescription {
    pub x: TensorDescription,
    pub output_size: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dDescription {
    pub x: TensorDescription,
    pub output_size: [usize; 2],
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct MaxPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub indices: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct MaxPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
}

#[allow(missing_docs)]
#[derive(Clone, Debug, Hash, PartialEq)]
pub struct MaxPool2dWithIndicesDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq)]
#[allow(missing_docs)]
pub struct MaxPool2dWithIndicesBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub indices: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
}

impl TensorOpsDescription {
    /// Cleanup the remaining tensor handles that have not been used.
    pub(crate) fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::BaseOpsInt(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::BaseOpsBool(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::NumericOpsFloat(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::NumericOpsInt(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::BoolOps(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::IntOps(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::FloatOps(ops) => ops.cleanup_tensor(handles),
            TensorOpsDescription::ModuleOps(ops) => ops.cleanup_tensor(handles),
        }

        // Cleanup tensor handles that were outputted, but ignored.
        handles.cleanup_orphans();
    }
}

impl BaseOpsDescription {
    fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            BaseOpsDescription::ToDevice(_) => (),
            BaseOpsDescription::Reshape(desc) => {
                handles.cleanup(&desc.input);
            }
            BaseOpsDescription::SwapDims(desc) => {
                handles.cleanup(&desc.input);
            }
            BaseOpsDescription::Slice(desc) => {
                handles.cleanup(&desc.tensor);
            }
            BaseOpsDescription::SliceAssign(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.value);
            }
            BaseOpsDescription::Equal(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            BaseOpsDescription::Repeat(desc) => {
                handles.cleanup(&desc.tensor);
            }
            BaseOpsDescription::Cat(desc) => {
                for t in desc.tensors.iter() {
                    handles.cleanup(t);
                }
            }
        }
    }
}

impl<E: Element> NumericOpsDescription<E> {
    fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOpsDescription::Add(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::AddScalar(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Sub(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::SubScalar(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Mul(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::MulScalar(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Div(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::DivScalar(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Ones(_) => {}
            NumericOpsDescription::Gather(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
            }
            NumericOpsDescription::Scatter(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
                handles.cleanup(&desc.value);
            }
            NumericOpsDescription::Select(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
            }
            NumericOpsDescription::SelectAssign(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
                handles.cleanup(&desc.value);
            }
            NumericOpsDescription::MaskWhere(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.value);
                handles.cleanup(&desc.mask);
            }
            NumericOpsDescription::MaskFill(desc) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.mask);
            }
            NumericOpsDescription::EqualElem(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::GreaterElem(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::GreaterEqualElem(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::LowerElem(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::LowerEqualElem(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Greater(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::GreaterEqual(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::Lower(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::LowerEqual(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::ArgMax(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::ArgMin(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Clamp(desc) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::ClampMin(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::ClampMax(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Abs(desc) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::Zeros(_) => {}
            NumericOpsDescription::Full(_) => {}
            NumericOpsDescription::MeanDim(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Mean(desc) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::Sum(desc) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::SumDim(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Max(desc) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::MaxDimWithIndices(desc) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::MinDimWithIndices(desc) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::Min(desc) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::MaxDim(desc) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::MinDim(desc) => {
                handles.cleanup(&desc.lhs);
            }
        }
    }
}

impl FloatOpsDescription {
    fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            FloatOpsDescription::Matmul(desc) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            FloatOpsDescription::Random(_) => {}
            FloatOpsDescription::Exp(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Log(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Log1p(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Erf(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Recip(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Powf(desc) => handles.cleanup(&desc.lhs),
            FloatOpsDescription::Sqrt(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Cos(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Sin(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::Tanh(desc) => handles.cleanup(&desc.input),
            FloatOpsDescription::IntoInt(desc) => handles.cleanup(&desc.input),
        }
    }
}

impl IntOpsDescription {
    fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            IntOpsDescription::IntoFloat(desc) => {
                handles.cleanup(&desc.input);
            }
        }
    }
}

impl BoolOpsDescription {
    fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            BoolOpsDescription::IntoFloat(desc) => {
                handles.cleanup(&desc.input);
            }
            BoolOpsDescription::IntoInt(desc) => {
                handles.cleanup(&desc.input);
            }
            BoolOpsDescription::Not(desc) => {
                handles.cleanup(&desc.input);
            }
        }
    }
}

impl ModuleOpsDescription {
    fn cleanup_tensor<B: FusionBackend>(&self, handles: &mut HandleContainer<B>) {
        match self {
            ModuleOpsDescription::Embedding(desc) => {
                handles.cleanup(&desc.weights);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::EmbeddingBackward(desc) => {
                handles.cleanup(&desc.weights);
                handles.cleanup(&desc.out_grad);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::Conv1d(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::Conv2d(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::ConvTranspose1d(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::ConvTranspose2d(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::AvgPool1d(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AvgPool2d(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AvgPool1dBackward(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AvgPool2dBackward(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AdaptiveAvgPool1d(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AdaptiveAvgPool2d(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::MaxPool1d(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool1dWithIndices(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::MaxPool2d(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool2dWithIndices(desc) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
                handles.cleanup(&desc.indices);
            }
        }
    }
}
impl core::hash::Hash for RandomOpsDescription {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.out.hash(state);

        match self.distribution {
            Distribution::Default => 1u8.hash(state),
            Distribution::Bernoulli(_) => 2u8.hash(state),
            Distribution::Uniform(_, _) => 3u8.hash(state),
            Distribution::Normal(_, _) => 4u8.hash(state),
        }
    }
}
impl<E> core::hash::Hash for ScalarOpsDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lhs.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for MaskFillOpsDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.mask.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for ClampOpsDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for NumericOpsDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            NumericOpsDescription::Add(desc) => desc.hash(state),
            NumericOpsDescription::AddScalar(desc) => desc.hash(state),
            NumericOpsDescription::Sub(desc) => desc.hash(state),
            NumericOpsDescription::SubScalar(desc) => desc.hash(state),
            NumericOpsDescription::Div(desc) => desc.hash(state),
            NumericOpsDescription::DivScalar(desc) => desc.hash(state),
            NumericOpsDescription::Mul(desc) => desc.hash(state),
            NumericOpsDescription::MulScalar(desc) => desc.hash(state),
            NumericOpsDescription::Abs(desc) => desc.hash(state),
            NumericOpsDescription::Ones(desc) => desc.hash(state),
            NumericOpsDescription::Zeros(desc) => desc.hash(state),
            NumericOpsDescription::Full(desc) => desc.0.hash(state),
            NumericOpsDescription::Gather(desc) => desc.hash(state),
            NumericOpsDescription::Scatter(desc) => desc.hash(state),
            NumericOpsDescription::Select(desc) => desc.hash(state),
            NumericOpsDescription::SelectAssign(desc) => desc.hash(state),
            NumericOpsDescription::MaskWhere(desc) => desc.hash(state),
            NumericOpsDescription::MaskFill(desc) => desc.hash(state),
            NumericOpsDescription::MeanDim(desc) => desc.hash(state),
            NumericOpsDescription::Mean(desc) => desc.hash(state),
            NumericOpsDescription::Sum(desc) => desc.hash(state),
            NumericOpsDescription::SumDim(desc) => desc.hash(state),
            NumericOpsDescription::EqualElem(desc) => desc.hash(state),
            NumericOpsDescription::Greater(desc) => desc.hash(state),
            NumericOpsDescription::GreaterElem(desc) => desc.hash(state),
            NumericOpsDescription::GreaterEqual(desc) => desc.hash(state),
            NumericOpsDescription::GreaterEqualElem(desc) => desc.hash(state),
            NumericOpsDescription::Lower(desc) => desc.hash(state),
            NumericOpsDescription::LowerElem(desc) => desc.hash(state),
            NumericOpsDescription::LowerEqual(desc) => desc.hash(state),
            NumericOpsDescription::LowerEqualElem(desc) => desc.hash(state),
            NumericOpsDescription::ArgMax(desc) => desc.hash(state),
            NumericOpsDescription::ArgMin(desc) => desc.hash(state),
            NumericOpsDescription::Max(desc) => desc.hash(state),
            NumericOpsDescription::MaxDimWithIndices(desc) => desc.hash(state),
            NumericOpsDescription::MinDimWithIndices(desc) => desc.hash(state),
            NumericOpsDescription::Min(desc) => desc.hash(state),
            NumericOpsDescription::MaxDim(desc) => desc.hash(state),
            NumericOpsDescription::MinDim(desc) => desc.hash(state),
            NumericOpsDescription::Clamp(desc) => desc.hash(state),
            NumericOpsDescription::ClampMax(desc) => desc.hash(state),
            NumericOpsDescription::ClampMin(desc) => desc.hash(state),
        }
    }
}
