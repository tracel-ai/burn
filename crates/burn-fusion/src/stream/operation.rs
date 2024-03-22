use crate::FusionBackend;
use crate::{HandleContainer, TensorDescription};
use burn_tensor::ops::{ConvOptions, ConvTransposeOptions, InterpolateMode, InterpolateOptions};
use burn_tensor::{Distribution, Element};
use serde::{Deserialize, Serialize};
use std::ops::Range;

/// General trait to abstract how a single operation is executed.
pub trait Operation<B: FusionBackend>: Send + Sync {
    /// Execute the operation.
    fn execute(self: Box<Self>, handles: &mut HandleContainer<B>);
}

/// Describe all tensor operations possible.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum OperationDescription {
    /// Basic operation on a float tensor.
    BaseFloat(BaseOperationDescription),
    /// Basic operation on an int tensor.
    BaseInt(BaseOperationDescription),
    /// Basic operation on a bool tensor.
    BaseBool(BaseOperationDescription),
    /// Numeric operation on a float tensor.
    NumericFloat(NumericOperationDescription<f32>),
    /// Numeric operation on an int tensor.
    NumericInt(NumericOperationDescription<i32>),
    /// Operation specific to a bool tensor.
    Bool(BoolOperationDescription),
    /// Operation specific to an int tensor.
    Int(IntOperationDescription),
    /// Operation specific to a float tensor.
    Float(FloatOperationDescription),
    /// Module operation.
    Module(ModuleOperationDescription),
}

/// Operation description specific to a float tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum FloatOperationDescription {
    /// Operation corresponding to [exp](burn_tensor::ops::FloatTensorOps::float_exp).
    Exp(UnaryOperationDescription),
    /// Operation corresponding to [log](burn_tensor::ops::FloatTensorOps::float_log).
    Log(UnaryOperationDescription),
    /// Operation corresponding to [log1p](burn_tensor::ops::FloatTensorOps::float_log1p).
    Log1p(UnaryOperationDescription),
    /// Operation corresponding to [erf](burn_tensor::ops::FloatTensorOps::float_erf).
    Erf(UnaryOperationDescription),
    /// Operation corresponding to [powf_scalar](burn_tensor::ops::FloatTensorOps::float_powf_scalar).
    PowfScalar(ScalarOperationDescription<f32>),
    /// Operation corresponding to [sqrt](burn_tensor::ops::FloatTensorOps::float_sqrt).
    Sqrt(UnaryOperationDescription),
    /// Operation corresponding to [cos](burn_tensor::ops::FloatTensorOps::float_cos).
    Cos(UnaryOperationDescription),
    /// Operation corresponding to [sin](burn_tensor::ops::FloatTensorOps::float_sin).
    Sin(UnaryOperationDescription),
    /// Operation corresponding to [tanh](burn_tensor::ops::FloatTensorOps::float_tanh).
    Tanh(UnaryOperationDescription),
    /// Operation corresponding to [into_int](burn_tensor::ops::FloatTensorOps::float_into_int).
    IntoInt(UnaryOperationDescription),
    /// Operation corresponding to [matmul](burn_tensor::ops::FloatTensorOps::float_matmul).
    Matmul(BinaryOperationDescription),
    /// Operation corresponding to [random](burn_tensor::ops::FloatTensorOps::float_random).
    Random(RandomOperationDescription),
    /// Operation corresponding to [recip](burn_tensor::ops::FloatTensorOps::float_recip).
    Recip(UnaryOperationDescription),
}

/// Operation description specific to module.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum ModuleOperationDescription {
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
    /// Operation corresponding to [interpolate](burn_tensor::ops::ModuleOps::interpolate).
    Interpolate(InterpolateDescription),
    /// Operation corresponding to [interpolate backward](burn_tensor::ops::ModuleOps::interpolate_backward).
    InterpolateBackward(InterpolateBackwardDescription),
}

/// Basic operations that can be done on any tensor type.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BaseOperationDescription {
    /// Operation corresponding to:
    ///
    /// Float => [to device](burn_tensor::ops::FloatTensorOps::float_to_device).
    /// Int => [to device](burn_tensor::ops::IntTensorOps::int_to_device).
    /// Bool => [to device](burn_tensor::ops::BoolTensorOps::bool_to_device).
    ToDevice(TensorDescription),
    /// Operation corresponding to:
    ///
    /// Float => [reshape](burn_tensor::ops::FloatTensorOps::float_reshape).
    /// Int => [reshape](burn_tensor::ops::IntTensorOps::int_reshape).
    /// Bool => [reshape](burn_tensor::ops::BoolTensorOps::bool_reshape).
    Reshape(ReshapeDescription),

    /// Operation corresponding to:
    ///
    /// Float => [swap_dims](burn_tensor::ops::FloatTensorOps::float_swap_dims).
    /// Int => [swap_dims](burn_tensor::ops::IntTensorOps::int_swap_dims).
    /// Bool => [swap_dims](burn_tensor::ops::BoolTensorOps::bool_swap_dims).
    SwapDims(SwapDimsDescription),

    /// Operation corresponding to:
    ///
    /// Float => [permute](burn_tensor::ops::FloatTensorOps::float_permute).
    /// Int => [permute](burn_tensor::ops::IntTensorOps::int_permute).
    /// Bool => [permute](burn_tensor::ops::BoolTensorOps::bool_permute).
    Permute(PermuteOperationDescription),

    /// Operation corresponding to:
    /// Float => [flip](burn_tensor::ops::FloatTensorOps::float_flip).
    /// Int => [flip](burn_tensor::ops::IntTensorOps::int_flip).
    /// Bool => [flip](burn_tensor::ops::BoolTensorOps::bool_flip).
    Flip(FlipOperationDescription),

    /// Operation corresponding to:
    ///
    /// Float => [expand](burn_tensor::ops::FloatTensorOps::float_expand).
    /// Int => [expand](burn_tensor::ops::IntTensorOps::int_expand).
    /// Bool => [expand](burn_tensor::ops::BoolTensorOps::bool_expand).
    Expand(ExpandOperationDescription),

    /// Operation corresponding to:
    ///
    /// Float => [slice](burn_tensor::ops::FloatTensorOps::float_slice).
    /// Int => [slice](burn_tensor::ops::IntTensorOps::int_slice).
    /// Bool => [slice](burn_tensor::ops::BoolTensorOps::bool_slice).
    Slice(SliceOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [slice assign](burn_tensor::ops::FloatTensorOps::float_slice_assign).
    /// Int => [slice assign](burn_tensor::ops::IntTensorOps::int_slice_assign).
    /// Bool => [slice assign](burn_tensor::ops::BoolTensorOps::bool_slice_assign).
    SliceAssign(SliceAssignOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [equal](burn_tensor::ops::FloatTensorOps::float_equal).
    /// Int => [equal](burn_tensor::ops::IntTensorOps::int_equal).
    /// Bool => [equal](burn_tensor::ops::BoolTensorOps::bool_equal).
    Equal(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [repeat](burn_tensor::ops::FloatTensorOps::float_repeat).
    /// Int => [repeat](burn_tensor::ops::IntTensorOps::int_repeat).
    /// Bool => [repeat](burn_tensor::ops::BoolTensorOps::bool_repeat).
    Repeat(RepeatOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [cat](burn_tensor::ops::FloatTensorOps::float_cat).
    /// Int => [cat](burn_tensor::ops::IntTensorOps::int_cat).
    /// Bool => [cat](burn_tensor::ops::BoolTensorOps::bool_cat).
    Cat(CatOperationDescription),
}

/// Numeric operations on int and float tensors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NumericOperationDescription<E> {
    /// Operation corresponding to:
    ///
    /// Float => [add](burn_tensor::ops::FloatTensorOps::float_add).
    /// Int => [add](burn_tensor::ops::IntTensorOps::int_add).
    Add(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [add scalar](burn_tensor::ops::FloatTensorOps::float_add_scalar).
    /// Int => [add scalar](burn_tensor::ops::IntTensorOps::int_add_scalar).
    AddScalar(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [sub](burn_tensor::ops::FloatTensorOps::float_sub).
    /// Int => [sub](burn_tensor::ops::IntTensorOps::int_sub).
    Sub(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [sub scalar](burn_tensor::ops::FloatTensorOps::float_sub_scalar).
    /// Int => [sub scalar](burn_tensor::ops::IntTensorOps::int_sub_scalar).
    SubScalar(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [div](burn_tensor::ops::FloatTensorOps::float_div).
    /// Int => [div](burn_tensor::ops::IntTensorOps::int_div).
    Div(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [div scalar](burn_tensor::ops::FloatTensorOps::float_div_scalar).
    /// Int => [div scalar](burn_tensor::ops::IntTensorOps::int_div_scalar).
    DivScalar(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mul](burn_tensor::ops::FloatTensorOps::float_mul).
    /// Int => [mul](burn_tensor::ops::IntTensorOps::int_mul).
    Mul(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [mul scalar](burn_tensor::ops::FloatTensorOps::float_mul_scalar).
    /// Int => [mul scalar](burn_tensor::ops::IntTensorOps::int_mul_scalar).
    MulScalar(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [abs](burn_tensor::ops::FloatTensorOps::float_abs).
    /// Int => [abs](burn_tensor::ops::IntTensorOps::int_abs).
    Abs(UnaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [ones](burn_tensor::ops::FloatTensorOps::float_ones).
    /// Int => [ones](burn_tensor::ops::IntTensorOps::int_ones).
    Ones(TensorDescription),
    /// Operation corresponding to:
    ///
    /// Float => [zeros](burn_tensor::ops::FloatTensorOps::float_zeros).
    /// Int => [zeros](burn_tensor::ops::IntTensorOps::int_zeros).
    Zeros(TensorDescription),
    /// Operation corresponding to:
    ///
    /// Float => [full](burn_tensor::ops::FloatTensorOps::float_full).
    /// Int => [full](burn_tensor::ops::IntTensorOps::int_full).
    Full((TensorDescription, E)),
    /// Operation corresponding to:
    ///
    /// Float => [gather](burn_tensor::ops::FloatTensorOps::float_gather).
    /// Int => [gather](burn_tensor::ops::IntTensorOps::int_gather).
    Gather(GatherOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [scatter](burn_tensor::ops::FloatTensorOps::float_scatter).
    /// Int => [scatter](burn_tensor::ops::IntTensorOps::int_scatter).
    Scatter(ScatterOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [select](burn_tensor::ops::FloatTensorOps::float_select).
    /// Int => [select](burn_tensor::ops::IntTensorOps::int_select).
    Select(SelectOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [select assign](burn_tensor::ops::FloatTensorOps::float_select_assign).
    /// Int => [select assign](burn_tensor::ops::IntTensorOps::int_select_assign).
    SelectAssign(SelectAssignOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [mask where](burn_tensor::ops::FloatTensorOps::float_mask_where).
    /// Int => [mask where](burn_tensor::ops::IntTensorOps::int_mask_where).
    MaskWhere(MaskWhereOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [mask fill](burn_tensor::ops::FloatTensorOps::float_mask_fill).
    /// Int => [mask fill](burn_tensor::ops::IntTensorOps::int_mask_fill).
    MaskFill(MaskFillOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mean dim](burn_tensor::ops::FloatTensorOps::float_mean_dim).
    /// Int => [mean dim](burn_tensor::ops::IntTensorOps::int_mean_dim).
    MeanDim(ScalarOperationDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [mean](burn_tensor::ops::FloatTensorOps::float_mean).
    /// Int => [mean](burn_tensor::ops::IntTensorOps::int_mean).
    Mean(UnaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [sum](burn_tensor::ops::FloatTensorOps::float_sum).
    /// Int => [sum](burn_tensor::ops::IntTensorOps::int_sum).
    Sum(UnaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [sum dim](burn_tensor::ops::FloatTensorOps::float_sum_dim).
    /// Int => [sum dim](burn_tensor::ops::IntTensorOps::int_sum_dim).
    SumDim(ScalarOperationDescription<usize>),

    /// Operation corresponding to:
    ///
    /// Float => [prod](burn_tensor::ops::FloatTensorOps::float_prod).
    /// Int => [prod](burn_tensor::ops::IntTensorOps::int_prod).
    Prod(UnaryOperationDescription),

    /// Operation corresponding to:
    ///
    /// Float => [prod dim](burn_tensor::ops::FloatTensorOps::float_prod_dim).
    /// Int => [prod dim](burn_tensor::ops::IntTensorOps::int_prod_dim).
    ProdDim(ScalarOperationDescription<usize>),

    /// Operation corresponding to:
    ///
    /// Float => [equal elem](burn_tensor::ops::FloatTensorOps::float_equal_elem).
    /// Int => [equal elem](burn_tensor::ops::IntTensorOps::int_equal_elem).
    EqualElem(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater](burn_tensor::ops::FloatTensorOps::float_greater).
    /// Int => [greater](burn_tensor::ops::IntTensorOps::int_greater).
    Greater(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [greater elem](burn_tensor::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterElem(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal](burn_tensor::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterEqual(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal elem](burn_tensor::ops::FloatTensorOps::float_greater_equal_elem).
    /// Int => [greater equal elem](burn_tensor::ops::IntTensorOps::int_greater_equal_elem).
    GreaterEqualElem(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower](burn_tensor::ops::FloatTensorOps::float_lower).
    /// Int => [lower](burn_tensor::ops::IntTensorOps::int_lower).
    Lower(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [lower elem](burn_tensor::ops::FloatTensorOps::float_lower_elem).
    /// Int => [lower elem](burn_tensor::ops::IntTensorOps::int_lower_elem).
    LowerElem(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal](burn_tensor::ops::FloatTensorOps::float_lower_equal).
    /// Int => [lower equal](burn_tensor::ops::IntTensorOps::int_lower_equal).
    LowerEqual(BinaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal elem](burn_tensor::ops::FloatTensorOps::float_lower_equal_elem).
    /// Int => [lower equal elem](burn_tensor::ops::IntTensorOps::int_lower_equal_elem).
    LowerEqualElem(ScalarOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Float => [argmax](burn_tensor::ops::FloatTensorOps::float_argmax).
    /// Int => [argmax](burn_tensor::ops::IntTensorOps::int_argmax).
    ArgMax(ScalarOperationDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [argmin](burn_tensor::ops::FloatTensorOps::float_argmin).
    /// Int => [argmin](burn_tensor::ops::IntTensorOps::int_argmin).
    ArgMin(ScalarOperationDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [max](burn_tensor::ops::FloatTensorOps::float_max).
    /// Int => [max](burn_tensor::ops::IntTensorOps::int_max).
    Max(UnaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [max dim with indices](burn_tensor::ops::FloatTensorOps::float_max_dim_with_indices).
    /// Int => [max dim with indices](burn_tensor::ops::IntTensorOps::int_max_dim_with_indices).
    MaxDimWithIndices(ReduceDimWithIndicesDescription),
    /// Operation corresponding to:
    ///
    /// Float => [min dim with indices](burn_tensor::ops::FloatTensorOps::float_min_dim_with_indices).
    /// Int => [min dim with indices](burn_tensor::ops::IntTensorOps::int_min_dim_with_indices).
    MinDimWithIndices(ReduceDimWithIndicesDescription),
    /// Operation corresponding to:
    ///
    /// Float => [min](burn_tensor::ops::FloatTensorOps::float_min).
    /// Int => [min](burn_tensor::ops::IntTensorOps::int_min).
    Min(UnaryOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [max dim](burn_tensor::ops::FloatTensorOps::float_max_dim).
    /// Int => [max dim](burn_tensor::ops::IntTensorOps::int_max_dim).
    MaxDim(ScalarOperationDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [min dim](burn_tensor::ops::FloatTensorOps::float_min_dim).
    /// Int => [min dim](burn_tensor::ops::IntTensorOps::int_min_dim).
    MinDim(ScalarOperationDescription<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [clamp](burn_tensor::ops::FloatTensorOps::float_clamp).
    /// Int => [clamp](burn_tensor::ops::IntTensorOps::int_clamp).
    Clamp(ClampOperationDescription<E>),
    /// Operation corresponding to:
    ///
    /// Int => [random](burn_tensor::ops::IntTensorOps::int_random).
    IntRandom(RandomOperationDescription),
    /// Operation corresponding to:
    ///
    /// Float => [powf](burn_tensor::ops::FloatTensorOps::float_powf).
    /// Int => [powf](burn_tensor::ops::IntTensorOps::int_powf).
    Powf(BinaryOperationDescription),
}

/// Operation description specific to an int tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum IntOperationDescription {
    /// Operation corresponding to [into float](burn_tensor::ops::IntTensorOps::int_into_float).
    IntoFloat(UnaryOperationDescription),
}

/// Operation description specific to a bool tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BoolOperationDescription {
    /// Operation corresponding to [into float](burn_tensor::ops::BoolTensorOps::bool_into_float).
    IntoFloat(UnaryOperationDescription),
    /// Operation corresponding to [into int](burn_tensor::ops::BoolTensorOps::bool_into_int).
    IntoInt(UnaryOperationDescription),
    /// Operation corresponding to [not](burn_tensor::ops::BoolTensorOps::bool_not).
    Not(UnaryOperationDescription),
}

/// Swap dim operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct SwapDimsDescription {
    /// Input tensor description.
    pub input: TensorDescription,
    /// Output tensor description.
    pub out: TensorDescription,
    /// The first dim to swap.
    pub dim1: usize,
    /// The second dim to swap.
    pub dim2: usize,
}

/// Permute operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct PermuteOperationDescription {
    /// Input tensor description.
    pub input: TensorDescription,
    /// Output tensor description.
    pub out: TensorDescription,
    /// The new order of the dimensions.
    pub axes: Vec<usize>,
}

/// Expand operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExpandOperationDescription {
    /// Input tensor description.
    pub input: TensorDescription,
    /// Output tensor description.
    pub out: TensorDescription,
    /// The new shape.
    pub shape: Vec<usize>,
}

/// Flip operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct FlipOperationDescription {
    /// Input tensor description.
    pub input: TensorDescription,
    /// Output tensor description.
    pub out: TensorDescription,
    /// The dimensions to flip.
    pub axes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RandomOperationDescription {
    pub out: TensorDescription,
    pub distribution: Distribution,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ReshapeDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ExpandDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct BinaryOperationDescription {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct UnaryOperationDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarOperationDescription<E> {
    pub lhs: TensorDescription,
    pub rhs: E,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GatherOperationDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScatterOperationDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectOperationDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectAssignOperationDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceOperationDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceAssignOperationDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskWhereOperationDescription {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskFillOperationDescription<E> {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: E,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ClampOperationDescription<E> {
    pub tensor: TensorDescription,
    pub min: E,
    pub max: E,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RepeatOperationDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub times: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CatOperationDescription {
    pub tensors: Vec<TensorDescription>,
    pub dim: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ReduceDimWithIndicesDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingDescription {
    pub weights: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingBackwardDescription {
    pub weights: TensorDescription,
    pub out_grad: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: Conv1dOptionsDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: Conv2dOptionsDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTranspose1dOptionsDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTranspose2dOptionsDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOptionsDescription {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOptionsDescription {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOptionsDescription {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub padding_out: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOptionsDescription {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

impl From<ConvOptions<1>> for Conv1dOptionsDescription {
    fn from(value: ConvOptions<1>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<2>> for Conv2dOptionsDescription {
    fn from(value: ConvOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvTransposeOptions<1>> for ConvTranspose1dOptionsDescription {
    fn from(value: ConvTransposeOptions<1>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            padding_out: value.padding_out,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvTransposeOptions<2>> for ConvTranspose2dOptionsDescription {
    fn from(value: ConvTransposeOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            padding_out: value.padding_out,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<Conv1dOptionsDescription> for ConvOptions<1> {
    fn from(val: Conv1dOptionsDescription) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv2dOptionsDescription> for ConvOptions<2> {
    fn from(val: Conv2dOptionsDescription) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose1dOptionsDescription> for ConvTransposeOptions<1> {
    fn from(val: ConvTranspose1dOptionsDescription) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose2dOptionsDescription> for ConvTransposeOptions<2> {
    fn from(val: ConvTranspose2dOptionsDescription) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dDescription {
    pub x: TensorDescription,
    pub output_size: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dDescription {
    pub x: TensorDescription,
    pub output_size: [usize; 2],
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
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
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct MaxPool2dWithIndicesDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum InterpolateModeDescription {
    Nearest,
    Bilinear,
    Bicubic,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOptionsDescription {
    pub mode: InterpolateModeDescription,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateDescription {
    pub x: TensorDescription,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsDescription,
    pub out: TensorDescription,
}

impl From<InterpolateModeDescription> for InterpolateMode {
    fn from(val: InterpolateModeDescription) -> Self {
        match val {
            InterpolateModeDescription::Nearest => Self::Nearest,
            InterpolateModeDescription::Bilinear => Self::Bilinear,
            InterpolateModeDescription::Bicubic => Self::Bicubic,
        }
    }
}

impl From<InterpolateOptionsDescription> for InterpolateOptions {
    fn from(val: InterpolateOptionsDescription) -> Self {
        Self {
            mode: val.mode.into(),
        }
    }
}

impl From<InterpolateMode> for InterpolateModeDescription {
    fn from(val: InterpolateMode) -> Self {
        match val {
            InterpolateMode::Nearest => Self::Nearest,
            InterpolateMode::Bilinear => Self::Bilinear,
            InterpolateMode::Bicubic => Self::Bicubic,
        }
    }
}

impl From<InterpolateOptions> for InterpolateOptionsDescription {
    fn from(val: InterpolateOptions) -> Self {
        Self {
            mode: val.mode.into(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsDescription,
    pub out: TensorDescription,
}

impl OperationDescription {
    /// Cleanup the remaining tensor handles that have not been used.
    pub(crate) fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            OperationDescription::BaseFloat(ops) => ops.nodes(),
            OperationDescription::BaseInt(ops) => ops.nodes(),
            OperationDescription::BaseBool(ops) => ops.nodes(),
            OperationDescription::NumericFloat(ops) => ops.nodes(),
            OperationDescription::NumericInt(ops) => ops.nodes(),
            OperationDescription::Bool(ops) => ops.nodes(),
            OperationDescription::Int(ops) => ops.nodes(),
            OperationDescription::Float(ops) => ops.nodes(),
            OperationDescription::Module(ops) => ops.nodes(),
        }
    }
}

impl BaseOperationDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            BaseOperationDescription::ToDevice(desc) => vec![desc],
            BaseOperationDescription::Reshape(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOperationDescription::SwapDims(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOperationDescription::Permute(desc) => {
                vec![&desc.input, &desc.out]
            }

            BaseOperationDescription::Expand(desc) => {
                vec![&desc.input, &desc.out]
            }

            BaseOperationDescription::Flip(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOperationDescription::Slice(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            BaseOperationDescription::SliceAssign(desc) => {
                vec![&desc.tensor, &desc.value, &desc.out]
            }
            BaseOperationDescription::Equal(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            BaseOperationDescription::Repeat(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            BaseOperationDescription::Cat(desc) => desc.tensors.iter().collect(),
        }
    }
}

impl<E: Element> NumericOperationDescription<E> {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            NumericOperationDescription::Add(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::AddScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Sub(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::SubScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Mul(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::MulScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Div(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::DivScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Ones(desc) => vec![desc],
            NumericOperationDescription::Gather(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.out]
            }
            NumericOperationDescription::Scatter(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.value, &desc.out]
            }
            NumericOperationDescription::Select(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.out]
            }
            NumericOperationDescription::SelectAssign(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.value, &desc.out]
            }
            NumericOperationDescription::MaskWhere(desc) => {
                vec![&desc.tensor, &desc.mask, &desc.value, &desc.out]
            }
            NumericOperationDescription::MaskFill(desc) => {
                vec![&desc.tensor, &desc.mask, &desc.out]
            }
            NumericOperationDescription::EqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::GreaterElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::GreaterEqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::LowerElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::LowerEqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Greater(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::GreaterEqual(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::Lower(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::LowerEqual(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationDescription::ArgMax(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::ArgMin(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Clamp(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            NumericOperationDescription::Abs(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationDescription::Zeros(desc) => vec![desc],
            NumericOperationDescription::Full(desc) => vec![&desc.0],
            NumericOperationDescription::MeanDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Mean(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationDescription::Sum(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationDescription::SumDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Prod(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationDescription::ProdDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::Max(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationDescription::MaxDimWithIndices(desc) => {
                vec![&desc.tensor, &desc.out_indices, &desc.out]
            }
            NumericOperationDescription::MinDimWithIndices(desc) => {
                vec![&desc.tensor, &desc.out_indices, &desc.out]
            }
            NumericOperationDescription::Min(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationDescription::MaxDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::MinDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationDescription::IntRandom(desc) => {
                vec![&desc.out]
            }
            NumericOperationDescription::Powf(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
        }
    }
}

impl FloatOperationDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            FloatOperationDescription::Matmul(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            FloatOperationDescription::Random(desc) => vec![&desc.out],
            FloatOperationDescription::Exp(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Log(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Log1p(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Erf(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Recip(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::PowfScalar(desc) => vec![&desc.lhs, &desc.out],
            FloatOperationDescription::Sqrt(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Cos(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Sin(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::Tanh(desc) => vec![&desc.input, &desc.out],
            FloatOperationDescription::IntoInt(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl IntOperationDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            IntOperationDescription::IntoFloat(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl BoolOperationDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            BoolOperationDescription::IntoFloat(desc) => vec![&desc.input, &desc.out],
            BoolOperationDescription::IntoInt(desc) => vec![&desc.input, &desc.out],
            BoolOperationDescription::Not(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl ModuleOperationDescription {
    fn nodes(&self) -> Vec<&TensorDescription> {
        match self {
            ModuleOperationDescription::Embedding(desc) => {
                vec![&desc.weights, &desc.indices, &desc.out]
            }
            ModuleOperationDescription::EmbeddingBackward(desc) => {
                vec![&desc.weights, &desc.out_grad, &desc.indices, &desc.out]
            }
            ModuleOperationDescription::Conv1d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationDescription::Conv2d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationDescription::ConvTranspose1d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationDescription::ConvTranspose2d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationDescription::AvgPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::AvgPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::AvgPool1dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationDescription::AvgPool2dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationDescription::AdaptiveAvgPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::AdaptiveAvgPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::AdaptiveAvgPool1dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationDescription::AdaptiveAvgPool2dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationDescription::MaxPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::MaxPool1dWithIndices(desc) => {
                vec![&desc.x, &desc.out, &desc.out_indices]
            }
            ModuleOperationDescription::MaxPool1dWithIndicesBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.indices, &desc.grad]
            }
            ModuleOperationDescription::MaxPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::MaxPool2dWithIndices(desc) => {
                vec![&desc.x, &desc.out, &desc.out_indices]
            }
            ModuleOperationDescription::MaxPool2dWithIndicesBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.indices, &desc.grad]
            }
            ModuleOperationDescription::Interpolate(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationDescription::InterpolateBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
        }
    }
}

impl core::hash::Hash for RandomOperationDescription {
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

impl<E> core::hash::Hash for ScalarOperationDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.lhs.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for MaskFillOperationDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.mask.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for ClampOperationDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for NumericOperationDescription<E> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            NumericOperationDescription::Add(desc) => desc.hash(state),
            NumericOperationDescription::AddScalar(desc) => desc.hash(state),
            NumericOperationDescription::Sub(desc) => desc.hash(state),
            NumericOperationDescription::SubScalar(desc) => desc.hash(state),
            NumericOperationDescription::Div(desc) => desc.hash(state),
            NumericOperationDescription::DivScalar(desc) => desc.hash(state),
            NumericOperationDescription::Mul(desc) => desc.hash(state),
            NumericOperationDescription::MulScalar(desc) => desc.hash(state),
            NumericOperationDescription::Abs(desc) => desc.hash(state),
            NumericOperationDescription::Ones(desc) => desc.hash(state),
            NumericOperationDescription::Zeros(desc) => desc.hash(state),
            NumericOperationDescription::Full(desc) => desc.0.hash(state),
            NumericOperationDescription::Gather(desc) => desc.hash(state),
            NumericOperationDescription::Scatter(desc) => desc.hash(state),
            NumericOperationDescription::Select(desc) => desc.hash(state),
            NumericOperationDescription::SelectAssign(desc) => desc.hash(state),
            NumericOperationDescription::MaskWhere(desc) => desc.hash(state),
            NumericOperationDescription::MaskFill(desc) => desc.hash(state),
            NumericOperationDescription::MeanDim(desc) => desc.hash(state),
            NumericOperationDescription::Mean(desc) => desc.hash(state),
            NumericOperationDescription::Sum(desc) => desc.hash(state),
            NumericOperationDescription::SumDim(desc) => desc.hash(state),
            NumericOperationDescription::Prod(desc) => desc.hash(state),
            NumericOperationDescription::ProdDim(desc) => desc.hash(state),
            NumericOperationDescription::EqualElem(desc) => desc.hash(state),
            NumericOperationDescription::Greater(desc) => desc.hash(state),
            NumericOperationDescription::GreaterElem(desc) => desc.hash(state),
            NumericOperationDescription::GreaterEqual(desc) => desc.hash(state),
            NumericOperationDescription::GreaterEqualElem(desc) => desc.hash(state),
            NumericOperationDescription::Lower(desc) => desc.hash(state),
            NumericOperationDescription::LowerElem(desc) => desc.hash(state),
            NumericOperationDescription::LowerEqual(desc) => desc.hash(state),
            NumericOperationDescription::LowerEqualElem(desc) => desc.hash(state),
            NumericOperationDescription::ArgMax(desc) => desc.hash(state),
            NumericOperationDescription::ArgMin(desc) => desc.hash(state),
            NumericOperationDescription::Max(desc) => desc.hash(state),
            NumericOperationDescription::MaxDimWithIndices(desc) => desc.hash(state),
            NumericOperationDescription::MinDimWithIndices(desc) => desc.hash(state),
            NumericOperationDescription::Min(desc) => desc.hash(state),
            NumericOperationDescription::MaxDim(desc) => desc.hash(state),
            NumericOperationDescription::MinDim(desc) => desc.hash(state),
            NumericOperationDescription::Clamp(desc) => desc.hash(state),
            NumericOperationDescription::IntRandom(desc) => desc.hash(state),
            NumericOperationDescription::Powf(desc) => desc.hash(state),
        }
    }
}
