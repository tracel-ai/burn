use crate::FusionBackend;
use crate::{HandleContainer, TensorDescription};
use burn_tensor::ops::FloatElem;
use burn_tensor::{
    ops::{ConvOptions, ConvTransposeOptions},
    Distribution, Element,
};
use core::hash::Hash;
use std::ops::Range;

/// General trait to abstract how a single operation is executed.
pub trait Ops<B: FusionBackend>: Send + Sync {
    /// The argument necessary for the execution to happen.
    type Args: Send + Sync;

    /// Execute the operation.
    fn execute(&self, args: &Self::Args, handles: &mut HandleContainer<B>);
}

/// Describe all tensor operations possible.
pub enum TensorOpsDescription<B: FusionBackend> {
    /// Basic operation on a float tensor.
    BaseOpsFloat(BaseOpsDescription<B>),
    /// Basic operation on an int tensor.
    BaseOpsInt(BaseOpsDescription<B>),
    /// Basic operation on a bool tensor.
    BaseOpsBool(BaseOpsDescription<B>),
    /// Numeric operation on a float tensor.
    NumericOpsFloat(NumericOpsDescription<B, B::FloatElem>),
    /// Numeric operation on an int tensor.
    NumericOpsInt(NumericOpsDescription<B, B::IntElem>),
    /// Operation specific to a bool tensor.
    BoolOps(BoolOpsDescription<B>),
    /// Operation specific to an int tensor.
    IntOps(IntOpsDescription<B>),
    /// Operation specific to a float tensor.
    FloatOps(FloatOpsDescription<B>),
    /// Module operation.
    ModuleOps(ModuleOpsDescription<B>),
}

/// Operation description specific to a float tensor.
pub enum FloatOpsDescription<B: FusionBackend> {
    /// Operation corresponding to [exp](burn_tensor::ops::TensorOps::exp).
    Exp(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [log](burn_tensor::ops::TensorOps::log).
    Log(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [log1p](burn_tensor::ops::TensorOps::log1p).
    Log1p(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [erf](burn_tensor::ops::TensorOps::erf).
    Erf(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [powf](burn_tensor::ops::TensorOps::powf).
    Powf(
        ScalarOpsDescription<f32>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<f32>>>,
    ),
    /// Operation corresponding to [sqrt](burn_tensor::ops::TensorOps::sqrt).
    Sqrt(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [cos](burn_tensor::ops::TensorOps::cos).
    Cos(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [sin](burn_tensor::ops::TensorOps::sin).
    Sin(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [tanh](burn_tensor::ops::TensorOps::tanh).
    Tanh(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [into_int](burn_tensor::ops::TensorOps::into_int).
    IntoInt(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [matmul](burn_tensor::ops::TensorOps::matmul).
    Matmul(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to [random](burn_tensor::ops::TensorOps::random).
    Random(
        (TensorDescription, Distribution<FloatElem<B>>),
        Box<dyn Ops<B, Args = (TensorDescription, Distribution<FloatElem<B>>)>>,
    ),
    /// Operation corresponding to [recip](burn_tensor::ops::TensorOps::recip).
    Recip(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
}

/// Operation description specific to module.
pub enum ModuleOpsDescription<B: FusionBackend> {
    /// Operation corresponding to [embedding](burn_tensor::ops::ModuleOps::embedding).
    Embedding(
        EmbeddingDescription,
        Box<dyn Ops<B, Args = EmbeddingDescription>>,
    ),
    /// Operation corresponding to [embedding_backward](burn_tensor::ops::ModuleOps::embedding_backward).
    EmbeddingBackward(
        EmbeddingBackwardDescription,
        Box<dyn Ops<B, Args = EmbeddingBackwardDescription>>,
    ),
    /// Operation corresponding to [conv1d](burn_tensor::ops::ModuleOps::conv1d).
    Conv1d(Conv1dDescription, Box<dyn Ops<B, Args = Conv1dDescription>>),
    /// Operation corresponding to [conv2d](burn_tensor::ops::ModuleOps::conv2d).
    Conv2d(Conv2dDescription, Box<dyn Ops<B, Args = Conv2dDescription>>),
    /// Operation corresponding to [conv transpose 1d](burn_tensor::ops::ModuleOps::conv_transpose1d).
    ConvTranspose1d(
        ConvTranspose1dDescription,
        Box<dyn Ops<B, Args = ConvTranspose1dDescription>>,
    ),
    /// Operation corresponding to [conv transpose 2d](burn_tensor::ops::ModuleOps::conv_transpose2d).
    ConvTranspose2d(
        ConvTranspose2dDescription,
        Box<dyn Ops<B, Args = ConvTranspose2dDescription>>,
    ),
    /// Operation corresponding to [avg pool 1d](burn_tensor::ops::ModuleOps::avg_pool1d).
    AvgPool1d(
        AvgPool1dDescription,
        Box<dyn Ops<B, Args = AvgPool1dDescription>>,
    ),
    /// Operation corresponding to [avg pool 2d](burn_tensor::ops::ModuleOps::avg_pool2d).
    AvgPool2d(
        AvgPool2dDescription,
        Box<dyn Ops<B, Args = AvgPool2dDescription>>,
    ),
    /// Operation corresponding to
    /// [avg pool 1d backward](burn_tensor::ops::ModuleOps::avg_pool1d_backward).
    AvgPool1dBackward(
        AvgPool1dBackwardDescription,
        Box<dyn Ops<B, Args = AvgPool1dBackwardDescription>>,
    ),
    /// Operation corresponding to
    /// [avg pool 2d backward](burn_tensor::ops::ModuleOps::avg_pool2d_backward).
    AvgPool2dBackward(
        AvgPool2dBackwardDescription,
        Box<dyn Ops<B, Args = AvgPool2dBackwardDescription>>,
    ),
    /// Operation corresponding to
    /// [adaptive avg pool 1d](burn_tensor::ops::ModuleOps::adaptive_avg_pool1d).
    AdaptiveAvgPool1d(
        AdaptiveAvgPool1dDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool1dDescription>>,
    ),
    /// Operation corresponding to
    /// [adaptive avg pool 2d](burn_tensor::ops::ModuleOps::adaptive_avg_pool2d).
    AdaptiveAvgPool2d(
        AdaptiveAvgPool2dDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool2dDescription>>,
    ),
    /// Operation corresponding to
    /// [adaptive avg pool 1d backward](burn_tensor::ops::ModuleOps::adaptive_avg_pool1d_backward).
    AdaptiveAvgPool1dBackward(
        AdaptiveAvgPool1dBackwardDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool1dBackwardDescription>>,
    ),
    /// Operation corresponding to
    /// [adaptive avg pool 2d backward](burn_tensor::ops::ModuleOps::adaptive_avg_pool2d_backward).
    AdaptiveAvgPool2dBackward(
        AdaptiveAvgPool2dBackwardDescription,
        Box<dyn Ops<B, Args = AdaptiveAvgPool2dBackwardDescription>>,
    ),
    /// Operation corresponding to
    /// [max pool 1d](burn_tensor::ops::ModuleOps::max_pool1d).
    MaxPool1d(
        MaxPool1dDescription,
        Box<dyn Ops<B, Args = MaxPool1dDescription>>,
    ),
    /// Operation corresponding to
    /// [max pool 1d with indices](burn_tensor::ops::ModuleOps::max_pool1d_with_indices).
    MaxPool1dWithIndices(
        MaxPool1dWithIndicesDescription,
        Box<dyn Ops<B, Args = MaxPool1dWithIndicesDescription>>,
    ),
    /// Operation corresponding to
    /// [max pool 1d with indices backward](burn_tensor::ops::ModuleOps::max_pool1d_with_indices_backward).
    MaxPool1dWithIndicesBackward(
        MaxPool1dWithIndicesBackwardDescription,
        Box<dyn Ops<B, Args = MaxPool1dWithIndicesBackwardDescription>>,
    ),
    /// Operation corresponding to
    /// [max pool 2d](burn_tensor::ops::ModuleOps::max_pool1d).
    MaxPool2d(
        MaxPool2dDescription,
        Box<dyn Ops<B, Args = MaxPool2dDescription>>,
    ),
    /// Operation corresponding to
    /// [max pool 2d with indices](burn_tensor::ops::ModuleOps::max_pool2d_with_indices).
    MaxPool2dWithIndices(
        MaxPool2dWithIndicesDescription,
        Box<dyn Ops<B, Args = MaxPool2dWithIndicesDescription>>,
    ),
    /// Operation corresponding to
    /// [max pool 2d with indices backward](burn_tensor::ops::ModuleOps::max_pool2d_with_indices_backward).
    MaxPool2dWithIndicesBackward(
        MaxPool2dWithIndicesBackwardDescription,
        Box<dyn Ops<B, Args = MaxPool2dWithIndicesBackwardDescription>>,
    ),
}

/// Basic operations that can be done on any tensor type.
pub enum BaseOpsDescription<B: FusionBackend> {
    /// Operation corresponding to:
    ///
    /// Float => [to device](burn_tensor::ops::TensorOps::to_device).
    /// Int => [to device](burn_tensor::ops::IntTensorOps::int_to_device).
    /// Bool => [to device](burn_tensor::ops::BoolTensorOps::bool_to_device).
    ToDevice(
        (TensorDescription, B::Device),
        Box<dyn Ops<B, Args = (TensorDescription, B::Device)>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [reshape](burn_tensor::ops::TensorOps::reshape).
    /// Int => [reshape](burn_tensor::ops::IntTensorOps::int_reshape).
    /// Bool => [reshape](burn_tensor::ops::BoolTensorOps::bool_reshape).
    Reshape(
        ReshapeDescription,
        Box<dyn Ops<B, Args = ReshapeDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [swap_dims](burn_tensor::ops::TensorOps::swap_dims).
    /// Int => [swap_dims](burn_tensor::ops::IntTensorOps::int_swap_dims).
    /// Bool => [swap_dims](burn_tensor::ops::BoolTensorOps::bool_swap_dims).
    SwapDims(
        SwapDimsDescription,
        Box<dyn Ops<B, Args = SwapDimsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [slice](burn_tensor::ops::TensorOps::slice).
    /// Int => [slice](burn_tensor::ops::IntTensorOps::int_slice).
    /// Bool => [slice](burn_tensor::ops::BoolTensorOps::bool_slice).
    Slice(
        SliceOpsDescription,
        Box<dyn Ops<B, Args = SliceOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [slice assign](burn_tensor::ops::TensorOps::slice_assign).
    /// Int => [slice assign](burn_tensor::ops::IntTensorOps::int_slice_assign).
    /// Bool => [slice assign](burn_tensor::ops::BoolTensorOps::bool_slice_assign).
    SliceAssign(
        SliceAssignOpsDescription,
        Box<dyn Ops<B, Args = SliceAssignOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [equal](burn_tensor::ops::TensorOps::equal).
    /// Int => [equal](burn_tensor::ops::IntTensorOps::int_equal).
    /// Bool => [equal](burn_tensor::ops::BoolTensorOps::bool_equal).
    Equal(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [repeat](burn_tensor::ops::TensorOps::repeat).
    /// Int => [repeat](burn_tensor::ops::IntTensorOps::int_repeat).
    /// Bool => [repeat](burn_tensor::ops::BoolTensorOps::bool_repeat).
    Repeat(
        RepeatOpsDescription,
        Box<dyn Ops<B, Args = RepeatOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [cat](burn_tensor::ops::TensorOps::cat).
    /// Int => [cat](burn_tensor::ops::IntTensorOps::int_cat).
    /// Bool => [cat](burn_tensor::ops::BoolTensorOps::bool_cat).
    Cat(CatOpsDescription, Box<dyn Ops<B, Args = CatOpsDescription>>),
}

/// Numeric operations on int and float tensors.
pub enum NumericOpsDescription<B: FusionBackend, E: Element> {
    /// Operation corresponding to:
    ///
    /// Float => [add](burn_tensor::ops::TensorOps::add).
    /// Int => [add](burn_tensor::ops::IntTensorOps::int_add).
    Add(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [add scalar](burn_tensor::ops::TensorOps::add_scalar).
    /// Int => [add scalar](burn_tensor::ops::IntTensorOps::int_add_scalar).
    AddScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [sub](burn_tensor::ops::TensorOps::sub).
    /// Int => [sub](burn_tensor::ops::IntTensorOps::int_sub).
    Sub(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [sub scalar](burn_tensor::ops::TensorOps::sub_scalar).
    /// Int => [sub scalar](burn_tensor::ops::IntTensorOps::int_sub_scalar).
    SubScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [div](burn_tensor::ops::TensorOps::div).
    /// Int => [div](burn_tensor::ops::IntTensorOps::int_div).
    Div(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [div scalar](burn_tensor::ops::TensorOps::div_scalar).
    /// Int => [div scalar](burn_tensor::ops::IntTensorOps::int_div_scalar).
    DivScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [mul](burn_tensor::ops::TensorOps::mul).
    /// Int => [mul](burn_tensor::ops::IntTensorOps::int_mul).
    Mul(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [mul scalar](burn_tensor::ops::TensorOps::mul_scalar).
    /// Int => [mul scalar](burn_tensor::ops::IntTensorOps::int_mul_scalar).
    MulScalar(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [abs](burn_tensor::ops::TensorOps::abs).
    /// Int => [abs](burn_tensor::ops::IntTensorOps::int_abs).
    Abs(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [ones](burn_tensor::ops::TensorOps::ones).
    /// Int => [ones](burn_tensor::ops::IntTensorOps::int_ones).
    Ones(TensorDescription, Box<dyn Ops<B, Args = TensorDescription>>),
    /// Operation corresponding to:
    ///
    /// Float => [zeros](burn_tensor::ops::TensorOps::zeros).
    /// Int => [zeros](burn_tensor::ops::IntTensorOps::int_zeros).
    Zeros(TensorDescription, Box<dyn Ops<B, Args = TensorDescription>>),
    /// Operation corresponding to:
    ///
    /// Float => [full](burn_tensor::ops::TensorOps::full).
    /// Int => [full](burn_tensor::ops::IntTensorOps::int_full).
    Full(
        (TensorDescription, E),
        Box<dyn Ops<B, Args = (TensorDescription, E)>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [gather](burn_tensor::ops::TensorOps::gather).
    /// Int => [gather](burn_tensor::ops::IntTensorOps::int_gather).
    Gather(
        GatherOpsDescription,
        Box<dyn Ops<B, Args = GatherOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [scatter](burn_tensor::ops::TensorOps::scatter).
    /// Int => [scatter](burn_tensor::ops::IntTensorOps::int_scatter).
    Scatter(
        ScatterOpsDescription,
        Box<dyn Ops<B, Args = ScatterOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [select](burn_tensor::ops::TensorOps::select).
    /// Int => [select](burn_tensor::ops::IntTensorOps::int_select).
    Select(
        SelectOpsDescription,
        Box<dyn Ops<B, Args = SelectOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [select assign](burn_tensor::ops::TensorOps::select_assign).
    /// Int => [select assign](burn_tensor::ops::IntTensorOps::int_select_assign).
    SelectAssign(
        SelectAssignOpsDescription,
        Box<dyn Ops<B, Args = SelectAssignOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [mask where](burn_tensor::ops::TensorOps::mask_where).
    /// Int => [mask where](burn_tensor::ops::IntTensorOps::int_mask_where).
    MaskWhere(
        MaskWhereOpsDescription,
        Box<dyn Ops<B, Args = MaskWhereOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [mask fill](burn_tensor::ops::TensorOps::mask_fill).
    /// Int => [mask fill](burn_tensor::ops::IntTensorOps::int_mask_fill).
    MaskFill(
        MaskFillOpsDescription<E>,
        Box<dyn Ops<B, Args = MaskFillOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [mean dim](burn_tensor::ops::TensorOps::mean_dim).
    /// Int => [mean dim](burn_tensor::ops::IntTensorOps::int_mean_dim).
    MeanDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [mean](burn_tensor::ops::TensorOps::mean).
    /// Int => [mean](burn_tensor::ops::IntTensorOps::int_mean).
    Mean(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [sum](burn_tensor::ops::TensorOps::sum).
    /// Int => [sum](burn_tensor::ops::IntTensorOps::int_sum).
    Sum(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [sum dim](burn_tensor::ops::TensorOps::sum_dim).
    /// Int => [sum dim](burn_tensor::ops::IntTensorOps::int_sum_dim).
    SumDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [equal elem](burn_tensor::ops::TensorOps::equal_elem).
    /// Int => [equal elem](burn_tensor::ops::IntTensorOps::int_equal_elem).
    EqualElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [greater](burn_tensor::ops::TensorOps::greater).
    /// Int => [greater](burn_tensor::ops::IntTensorOps::int_greater).
    Greater(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [greater elem](burn_tensor::ops::TensorOps::greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal](burn_tensor::ops::TensorOps::greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterEqual(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal elem](burn_tensor::ops::TensorOps::greater_equal_elem).
    /// Int => [greater equal elem](burn_tensor::ops::IntTensorOps::int_greater_equal_elem).
    GreaterEqualElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [lower](burn_tensor::ops::TensorOps::lower).
    /// Int => [lower](burn_tensor::ops::IntTensorOps::int_lower).
    Lower(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [lower elem](burn_tensor::ops::TensorOps::lower_elem).
    /// Int => [lower elem](burn_tensor::ops::IntTensorOps::int_lower_elem).
    LowerElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal](burn_tensor::ops::TensorOps::lower_equal).
    /// Int => [lower equal](burn_tensor::ops::IntTensorOps::int_lower_equal).
    LowerEqual(
        BinaryOpsDescription,
        Box<dyn Ops<B, Args = BinaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal elem](burn_tensor::ops::TensorOps::lower_equal_elem).
    /// Int => [lower equal elem](burn_tensor::ops::IntTensorOps::int_lower_equal_elem).
    LowerEqualElem(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [argmax](burn_tensor::ops::TensorOps::argmax).
    /// Int => [argmax](burn_tensor::ops::IntTensorOps::int_argmax).
    ArgMax(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [argmin](burn_tensor::ops::TensorOps::argmin).
    /// Int => [argmin](burn_tensor::ops::IntTensorOps::int_argmin).
    ArgMin(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [max](burn_tensor::ops::TensorOps::max).
    /// Int => [max](burn_tensor::ops::IntTensorOps::int_max).
    Max(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [max dim with indices](burn_tensor::ops::TensorOps::max_dim_with_indices).
    /// Int => [max dim with indices](burn_tensor::ops::IntTensorOps::int_max_dim_with_indices).
    MaxDimWithIndices(
        ReduceDimWithIndicesDescription,
        Box<dyn Ops<B, Args = ReduceDimWithIndicesDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [min dim with indices](burn_tensor::ops::TensorOps::min_dim_with_indices).
    /// Int => [min dim with indices](burn_tensor::ops::IntTensorOps::int_min_dim_with_indices).
    MinDimWithIndices(
        ReduceDimWithIndicesDescription,
        Box<dyn Ops<B, Args = ReduceDimWithIndicesDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [min](burn_tensor::ops::TensorOps::min).
    /// Int => [min](burn_tensor::ops::IntTensorOps::int_min).
    Min(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [max dim](burn_tensor::ops::TensorOps::max_dim).
    /// Int => [max dim](burn_tensor::ops::IntTensorOps::int_max_dim).
    MaxDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [min dim](burn_tensor::ops::TensorOps::min_dim).
    /// Int => [min dim](burn_tensor::ops::IntTensorOps::int_min_dim).
    MinDim(
        ScalarOpsDescription<usize>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<usize>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [clamp](burn_tensor::ops::TensorOps::clamp).
    /// Int => [clamp](burn_tensor::ops::IntTensorOps::int_clamp).
    Clamp(
        ClampOpsDescription<E>,
        Box<dyn Ops<B, Args = ClampOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [clamp max](burn_tensor::ops::TensorOps::clamp_max).
    /// Int => [clamp max](burn_tensor::ops::IntTensorOps::int_clamp_max).
    ClampMax(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
    /// Operation corresponding to:
    ///
    /// Float => [clamp min](burn_tensor::ops::TensorOps::clamp_min).
    /// Int => [cleamp min](burn_tensor::ops::IntTensorOps::int_clamp_min).
    ClampMin(
        ScalarOpsDescription<E>,
        Box<dyn Ops<B, Args = ScalarOpsDescription<E>>>,
    ),
}

/// Operation description specific to an int tensor.
pub enum IntOpsDescription<B: FusionBackend> {
    /// Operation corresponding to [into float](burn_tensor::ops::IntTensorOps::int_into_float).
    IntoFloat(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
}

/// Operation description specific to a bool tensor.
pub enum BoolOpsDescription<B: FusionBackend> {
    /// Operation corresponding to [into float](burn_tensor::ops::BoolTensorOps::bool_into_float).
    IntoFloat(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [into int](burn_tensor::ops::BoolTensorOps::bool_into_int).
    IntoInt(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
    /// Operation corresponding to [not](burn_tensor::ops::BoolTensorOps::bool_not).
    Not(
        UnaryOpsDescription,
        Box<dyn Ops<B, Args = UnaryOpsDescription>>,
    ),
}

#[derive(Hash)]
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

#[derive(Hash)]
#[allow(missing_docs)]
pub struct ReshapeDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
    pub shape: Vec<usize>,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct BinaryOpsDescription {
    pub lhs: TensorDescription,
    pub rhs: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct UnaryOpsDescription {
    pub input: TensorDescription,
    pub out: TensorDescription,
}

#[allow(missing_docs)]
pub struct ScalarOpsDescription<E> {
    pub lhs: TensorDescription,
    pub rhs: E,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct GatherOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct ScatterOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct SelectOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct SelectAssignOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub indices: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct SliceOpsDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct SliceAssignOpsDescription {
    pub tensor: TensorDescription,
    pub ranges: Vec<Range<usize>>,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct MaskWhereOpsDescription {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: TensorDescription,
    pub out: TensorDescription,
}

#[allow(missing_docs)]
pub struct MaskFillOpsDescription<E> {
    pub tensor: TensorDescription,
    pub mask: TensorDescription,
    pub value: E,
    pub out: TensorDescription,
}

#[allow(missing_docs)]
pub struct ClampOpsDescription<E> {
    pub tensor: TensorDescription,
    pub min: E,
    pub max: E,
    pub out: TensorDescription,
}

#[allow(missing_docs)]
pub struct RepeatOpsDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub times: usize,
    pub shape: Vec<usize>,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct CatOpsDescription {
    pub tensors: Vec<TensorDescription>,
    pub dim: usize,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct ReduceDimWithIndicesDescription {
    pub tensor: TensorDescription,
    pub dim: usize,
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct EmbeddingDescription {
    pub weights: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct EmbeddingBackwardDescription {
    pub weights: TensorDescription,
    pub out_grad: TensorDescription,
    pub indices: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct Conv1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvOptions<1>,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct Conv2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvOptions<2>,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct ConvTranspose1dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTransposeOptions<1>,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct ConvTranspose2dDescription {
    pub x: TensorDescription,
    pub weight: TensorDescription,
    pub bias: Option<TensorDescription>,
    pub options: ConvTransposeOptions<2>,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct AvgPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct AvgPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorDescription,
}

#[derive(Hash)]
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

#[derive(Hash)]
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

#[derive(Hash)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dDescription {
    pub x: TensorDescription,
    pub output_size: usize,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dDescription {
    pub x: TensorDescription,
    pub output_size: [usize; 2],
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dBackwardDescription {
    pub x: TensorDescription,
    pub grad: TensorDescription,
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct MaxPool1dDescription {
    pub x: TensorDescription,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorDescription,
}

#[derive(Hash)]
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

#[derive(Hash)]
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

#[derive(Hash)]
#[allow(missing_docs)]
pub struct MaxPool2dDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
}

#[derive(Hash)]
#[allow(missing_docs)]
pub struct MaxPool2dWithIndicesDescription {
    pub x: TensorDescription,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorDescription,
    pub out_indices: TensorDescription,
}

#[derive(Hash)]
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

impl<B: FusionBackend> TensorOpsDescription<B> {
    /// Cleanup the remaining tensor handles that have not been used.
    pub(crate) fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
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
    /// Execute the operation.
    pub(crate) fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            TensorOpsDescription::BaseOpsFloat(ops) => ops.execute(handles),
            TensorOpsDescription::BaseOpsInt(ops) => ops.execute(handles),
            TensorOpsDescription::BaseOpsBool(ops) => ops.execute(handles),
            TensorOpsDescription::NumericOpsFloat(ops) => ops.execute(handles),
            TensorOpsDescription::NumericOpsInt(ops) => ops.execute(handles),
            TensorOpsDescription::BoolOps(ops) => ops.execute(handles),
            TensorOpsDescription::IntOps(ops) => ops.execute(handles),
            TensorOpsDescription::FloatOps(ops) => ops.execute(handles),
            TensorOpsDescription::ModuleOps(ops) => ops.execute(handles),
        }
    }
}

impl<B: FusionBackend> BaseOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            BaseOpsDescription::ToDevice(_, _) => (),
            BaseOpsDescription::Reshape(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BaseOpsDescription::SwapDims(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BaseOpsDescription::Slice(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            BaseOpsDescription::SliceAssign(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.value);
            }
            BaseOpsDescription::Equal(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            BaseOpsDescription::Repeat(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            BaseOpsDescription::Cat(desc, _) => {
                for t in desc.tensors.iter() {
                    handles.cleanup(t);
                }
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            BaseOpsDescription::ToDevice(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Reshape(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::SwapDims(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Slice(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::SliceAssign(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Equal(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Repeat(desc, ops) => ops.execute(desc, handles),
            BaseOpsDescription::Cat(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusionBackend, E: Element> NumericOpsDescription<B, E> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOpsDescription::Add(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::AddScalar(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Sub(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::SubScalar(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Mul(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::MulScalar(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Div(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::DivScalar(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Ones(_, _) => {}
            NumericOpsDescription::Gather(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
            }
            NumericOpsDescription::Scatter(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
                handles.cleanup(&desc.value);
            }
            NumericOpsDescription::Select(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
            }
            NumericOpsDescription::SelectAssign(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.indices);
                handles.cleanup(&desc.value);
            }
            NumericOpsDescription::MaskWhere(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.value);
                handles.cleanup(&desc.mask);
            }
            NumericOpsDescription::MaskFill(desc, _) => {
                handles.cleanup(&desc.tensor);
                handles.cleanup(&desc.mask);
            }
            NumericOpsDescription::EqualElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::GreaterElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::GreaterEqualElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::LowerElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::LowerEqualElem(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Greater(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::GreaterEqual(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::Lower(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::LowerEqual(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            NumericOpsDescription::ArgMax(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::ArgMin(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Clamp(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::ClampMin(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::ClampMax(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Abs(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::Zeros(_, _) => {}
            NumericOpsDescription::Full(_, _) => {}
            NumericOpsDescription::MeanDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Mean(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::Sum(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::SumDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::Max(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::MaxDimWithIndices(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::MinDimWithIndices(desc, _) => {
                handles.cleanup(&desc.tensor);
            }
            NumericOpsDescription::Min(desc, _) => {
                handles.cleanup(&desc.input);
            }
            NumericOpsDescription::MaxDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
            NumericOpsDescription::MinDim(desc, _) => {
                handles.cleanup(&desc.lhs);
            }
        }
    }

    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            NumericOpsDescription::Add(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::AddScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Sub(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::SubScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Div(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::DivScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Mul(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MulScalar(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Ones(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Gather(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Scatter(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Select(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::SelectAssign(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaskWhere(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaskFill(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::EqualElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Greater(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::GreaterElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::GreaterEqual(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::GreaterEqualElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Lower(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::LowerElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::LowerEqual(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::LowerEqualElem(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ArgMax(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ArgMin(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Clamp(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ClampMin(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::ClampMax(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Abs(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Zeros(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Full(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MeanDim(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Mean(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Sum(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::SumDim(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Max(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaxDimWithIndices(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MinDimWithIndices(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::Min(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MaxDim(desc, ops) => ops.execute(desc, handles),
            NumericOpsDescription::MinDim(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusionBackend> FloatOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            FloatOpsDescription::Matmul(desc, _) => {
                handles.cleanup(&desc.lhs);
                handles.cleanup(&desc.rhs);
            }
            FloatOpsDescription::Random(_, _) => {}
            FloatOpsDescription::Exp(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Log(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Log1p(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Erf(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Recip(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Powf(desc, _) => handles.cleanup(&desc.lhs),
            FloatOpsDescription::Sqrt(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Cos(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Sin(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::Tanh(desc, _) => handles.cleanup(&desc.input),
            FloatOpsDescription::IntoInt(desc, _) => handles.cleanup(&desc.input),
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            FloatOpsDescription::Matmul(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Random(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Exp(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Log(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Log1p(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Erf(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Recip(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Powf(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Sqrt(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Cos(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Sin(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::Tanh(desc, ops) => ops.execute(desc, handles),
            FloatOpsDescription::IntoInt(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusionBackend> IntOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            IntOpsDescription::IntoFloat(desc, _) => {
                handles.cleanup(&desc.input);
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            IntOpsDescription::IntoFloat(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusionBackend> BoolOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            BoolOpsDescription::IntoFloat(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BoolOpsDescription::IntoInt(desc, _) => {
                handles.cleanup(&desc.input);
            }
            BoolOpsDescription::Not(desc, _) => {
                handles.cleanup(&desc.input);
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            BoolOpsDescription::IntoFloat(desc, ops) => ops.execute(desc, handles),
            BoolOpsDescription::IntoInt(desc, ops) => ops.execute(desc, handles),
            BoolOpsDescription::Not(desc, ops) => ops.execute(desc, handles),
        }
    }
}

impl<B: FusionBackend> ModuleOpsDescription<B> {
    fn cleanup_tensor(&self, handles: &mut HandleContainer<B>) {
        match self {
            ModuleOpsDescription::Embedding(desc, _) => {
                handles.cleanup(&desc.weights);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::EmbeddingBackward(desc, _) => {
                handles.cleanup(&desc.weights);
                handles.cleanup(&desc.out_grad);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::Conv1d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::Conv2d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::ConvTranspose1d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::ConvTranspose2d(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.weight);

                if let Some(bias) = &desc.bias {
                    handles.cleanup(bias);
                }
            }
            ModuleOpsDescription::AvgPool1d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AvgPool2d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AvgPool1dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AvgPool2dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AdaptiveAvgPool1d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AdaptiveAvgPool2d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
            }
            ModuleOpsDescription::MaxPool1d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool1dWithIndices(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
                handles.cleanup(&desc.indices);
            }
            ModuleOpsDescription::MaxPool2d(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool2dWithIndices(desc, _) => {
                handles.cleanup(&desc.x);
            }
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc, _) => {
                handles.cleanup(&desc.x);
                handles.cleanup(&desc.grad);
                handles.cleanup(&desc.indices);
            }
        }
    }
    fn execute(&self, handles: &mut HandleContainer<B>) {
        match self {
            ModuleOpsDescription::Embedding(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::EmbeddingBackward(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::Conv1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::Conv2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::ConvTranspose1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::ConvTranspose2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool1dBackward(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AvgPool2dBackward(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AdaptiveAvgPool1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AdaptiveAvgPool2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::AdaptiveAvgPool1dBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
            ModuleOpsDescription::AdaptiveAvgPool2dBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
            ModuleOpsDescription::MaxPool1d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool1dWithIndices(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool1dWithIndicesBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
            ModuleOpsDescription::MaxPool2d(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool2dWithIndices(desc, ops) => ops.execute(desc, handles),
            ModuleOpsDescription::MaxPool2dWithIndicesBackward(desc, ops) => {
                ops.execute(desc, handles)
            }
        }
    }
}
