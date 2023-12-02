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
    pub(crate) fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => ops.nodes(),
            TensorOpsDescription::BaseOpsInt(ops) => ops.nodes(),
            TensorOpsDescription::BaseOpsBool(ops) => ops.nodes(),
            TensorOpsDescription::NumericOpsFloat(ops) => ops.nodes(),
            TensorOpsDescription::NumericOpsInt(ops) => ops.nodes(),
            TensorOpsDescription::BoolOps(ops) => ops.nodes(),
            TensorOpsDescription::IntOps(ops) => ops.nodes(),
            TensorOpsDescription::FloatOps(ops) => ops.nodes(),
            TensorOpsDescription::ModuleOps(ops) => ops.nodes(),
        }
    }
}

impl BaseOpsDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            BaseOpsDescription::ToDevice(desc) => vec![desc],
            BaseOpsDescription::Reshape(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOpsDescription::SwapDims(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOpsDescription::Slice(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            BaseOpsDescription::SliceAssign(desc) => {
                vec![&desc.tensor, &desc.value, &desc.out]
            }
            BaseOpsDescription::Equal(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            BaseOpsDescription::Repeat(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            BaseOpsDescription::Cat(desc) => desc.tensors.iter().collect(),
        }
    }
}

impl<E: Element> NumericOpsDescription<E> {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            NumericOpsDescription::Add(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::AddScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Sub(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::SubScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Mul(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::MulScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Div(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::DivScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Ones(desc) => vec![desc],
            NumericOpsDescription::Gather(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.out]
            }
            NumericOpsDescription::Scatter(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.value, &desc.out]
            }
            NumericOpsDescription::Select(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.out]
            }
            NumericOpsDescription::SelectAssign(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.value, &desc.out]
            }
            NumericOpsDescription::MaskWhere(desc) => {
                vec![&desc.tensor, &desc.mask, &desc.value, &desc.out]
            }
            NumericOpsDescription::MaskFill(desc) => {
                vec![&desc.tensor, &desc.mask, &desc.out]
            }
            NumericOpsDescription::EqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::GreaterElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::GreaterEqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::LowerElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::LowerEqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Greater(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::GreaterEqual(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::Lower(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::LowerEqual(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOpsDescription::ArgMax(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::ArgMin(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Clamp(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            NumericOpsDescription::ClampMin(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::ClampMax(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Abs(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOpsDescription::Zeros(desc) => vec![desc],
            NumericOpsDescription::Full(desc) => vec![&desc.0],
            NumericOpsDescription::MeanDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Mean(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOpsDescription::Sum(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOpsDescription::SumDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::Max(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOpsDescription::MaxDimWithIndices(desc) => {
                vec![&desc.tensor, &desc.out_indices, &desc.out]
            }
            NumericOpsDescription::MinDimWithIndices(desc) => {
                vec![&desc.tensor, &desc.out_indices, &desc.out]
            }
            NumericOpsDescription::Min(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOpsDescription::MaxDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOpsDescription::MinDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
        }
    }
}

impl FloatOpsDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            FloatOpsDescription::Matmul(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            FloatOpsDescription::Random(desc) => vec![&desc.out],
            FloatOpsDescription::Exp(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Log(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Log1p(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Erf(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Recip(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Powf(desc) => vec![&desc.lhs, &desc.out],
            FloatOpsDescription::Sqrt(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Cos(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Sin(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::Tanh(desc) => vec![&desc.input, &desc.out],
            FloatOpsDescription::IntoInt(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl IntOpsDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            IntOpsDescription::IntoFloat(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl BoolOpsDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            BoolOpsDescription::IntoFloat(desc) => vec![&desc.input, &desc.out],
            BoolOpsDescription::IntoInt(desc) => vec![&desc.input, &desc.out],
            BoolOpsDescription::Not(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl ModuleOpsDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            ModuleOpsDescription::Embedding(desc) => {
                vec![&desc.weights, &desc.indices, &desc.out]
            }
            ModuleOpsDescription::EmbeddingBackward(desc) => {
                vec![&desc.weights, &desc.out_grad, &desc.indices, &desc.out]
            }
            ModuleOpsDescription::Conv1d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOpsDescription::Conv2d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOpsDescription::ConvTranspose1d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOpsDescription::ConvTranspose2d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOpsDescription::AvgPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOpsDescription::AvgPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOpsDescription::AvgPool1dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOpsDescription::AvgPool2dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOpsDescription::AdaptiveAvgPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOpsDescription::AdaptiveAvgPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOpsDescription::MaxPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOpsDescription::MaxPool1dWithIndices(desc) => {
                vec![&desc.x, &desc.out, &desc.out_indices]
            }
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.indices, &desc.grad]
            }
            ModuleOpsDescription::MaxPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOpsDescription::MaxPool2dWithIndices(desc) => {
                vec![&desc.x, &desc.out, &desc.out_indices]
            }
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.indices, &desc.grad]
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
