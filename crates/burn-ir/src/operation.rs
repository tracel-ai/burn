use core::hash::Hash;
use core::ops::Range;
use serde::{Deserialize, Serialize};

use alloc::borrow::ToOwned;
use alloc::boxed::Box;
use alloc::{string::String, vec, vec::Vec};

use burn_tensor::{
    ops::{
        ConvOptions, ConvTransposeOptions, DeformConvOptions, InterpolateMode, InterpolateOptions,
    },
    quantization::QuantizationScheme,
    DType, Distribution, Element,
};

use crate::TensorIr;

/// Custom operation in fusion stream, declaring it's inputs and outputs.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct CustomOpIr {
    /// Unique identifier of the operation.
    pub id: String,
    /// Input tensors used in this the custom operation.
    pub inputs: Vec<TensorIr>,
    /// Output tensors used in this the custom operation.
    pub outputs: Vec<TensorIr>,
}

impl CustomOpIr {
    /// Create a new custom operation intermediate representation.
    pub fn new(id: &'static str, inputs: &[TensorIr], outputs: &[TensorIr]) -> Self {
        Self {
            id: id.to_owned(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        }
    }

    /// Consume the intermediate representation, and get the in and output tensors.
    pub fn consume<const N_IN: usize, const N_OUT: usize>(
        self,
    ) -> ([TensorIr; N_IN], [TensorIr; N_OUT]) {
        (
            self.inputs.try_into().expect(
                "Wrong number of inputs expected (expected {D}, is {}), check your implementation",
            ),
            self.outputs.try_into().expect(
                "Wrong number of outputs expected (expected {D}, is {}), check your implementation",
            ),
        )
    }

    fn nodes(&self) -> Vec<&TensorIr> {
        self.inputs.iter().chain(self.outputs.iter()).collect()
    }
}

/// irribe all tensor operations possible.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum OperationIr {
    /// Basic operation on a float tensor.
    BaseFloat(BaseOperationIr),
    /// Basic operation on an int tensor.
    BaseInt(BaseOperationIr),
    /// Basic operation on a bool tensor.
    BaseBool(BaseOperationIr),
    /// Numeric operation on a float tensor.
    NumericFloat(DType, NumericOperationIr<f32>),
    /// Numeric operation on an int tensor.
    NumericInt(DType, NumericOperationIr<i32>),
    /// Operation specific to a bool tensor.
    Bool(BoolOperationIr),
    /// Operation specific to an int tensor.
    Int(IntOperationIr),
    /// Operation specific to a float tensor.
    Float(DType, FloatOperationIr),
    /// Module operation.
    Module(ModuleOperationIr),
    /// Initialize operation.
    Init(InitOperationIr),
    /// A custom operation.
    Custom(CustomOpIr),
}

/// Operation intermediate representation specific to a float tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum FloatOperationIr {
    /// Operation corresponding to [exp](burn_tensor::ops::FloatTensorOps::float_exp).
    Exp(UnaryOpIr),
    /// Operation corresponding to [log](burn_tensor::ops::FloatTensorOps::float_log).
    Log(UnaryOpIr),
    /// Operation corresponding to [log1p](burn_tensor::ops::FloatTensorOps::float_log1p).
    Log1p(UnaryOpIr),
    /// Operation corresponding to [erf](burn_tensor::ops::FloatTensorOps::float_erf).
    Erf(UnaryOpIr),
    /// Operation corresponding to [powf_scalar](burn_tensor::ops::FloatTensorOps::float_powf_scalar).
    PowfScalar(ScalarOpIr<f32>),
    /// Operation corresponding to [sqrt](burn_tensor::ops::FloatTensorOps::float_sqrt).
    Sqrt(UnaryOpIr),
    /// Operation corresponding to [cos](burn_tensor::ops::FloatTensorOps::float_cos).
    Cos(UnaryOpIr),
    /// Operation corresponding to [sin](burn_tensor::ops::FloatTensorOps::float_sin).
    Sin(UnaryOpIr),
    /// Operation corresponding to [tanh](burn_tensor::ops::FloatTensorOps::float_tanh).
    Tanh(UnaryOpIr),
    /// Operation corresponding to [round](burn_tensor::ops::FloatTensorOps::float_round).
    Round(UnaryOpIr),
    /// Operation corresponding to [floor](burn_tensor::ops::FloatTensorOps::float_floor).
    Floor(UnaryOpIr),
    /// Operation corresponding to [ceil](burn_tensor::ops::FloatTensorOps::float_ceil).
    Ceil(UnaryOpIr),
    /// Operation corresponding to [into_int](burn_tensor::ops::FloatTensorOps::float_into_int).
    IntoInt(UnaryOpIr),
    /// Operation corresponding to [matmul](burn_tensor::ops::FloatTensorOps::float_matmul).
    Matmul(BinaryOpIr),
    /// Operation corresponding to [random](burn_tensor::ops::FloatTensorOps::float_random).
    Random(RandomOpIr),
    /// Operation corresponding to [recip](burn_tensor::ops::FloatTensorOps::float_recip).
    Recip(UnaryOpIr),
    /// Operation corresponding to [quantize](burn_tensor::ops::QTensorOps::quantize).
    Quantize(QuantizeOpIr),
    /// Operation corresponding to [dequantize](burn_tensor::ops::QTensorOps::dequantize).
    Dequantize(DequantizeOpIr),
}

/// Operation intermediate representation specific to module.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum ModuleOperationIr {
    /// Operation corresponding to [embedding](burn_tensor::ops::ModuleOps::embedding).
    Embedding(EmbeddingOpIr),
    /// Operation corresponding to [embedding_backward](burn_tensor::ops::ModuleOps::embedding_backward).
    EmbeddingBackward(EmbeddingBackwardOpIr),
    /// Operation corresponding to [conv1d](burn_tensor::ops::ModuleOps::conv1d).
    Conv1d(Conv1dOpIr),
    /// Operation corresponding to [conv2d](burn_tensor::ops::ModuleOps::conv2d).
    Conv2d(Conv2dOpIr),
    /// Operation corresponding to [conv3d](burn_tensor::ops::ModuleOps::conv3d).
    Conv3d(Conv3dOpIr),
    /// Operation corresponding to [deform_conv2d](burn_tensor::ops::ModuleOps::deform_conv2d)
    DeformableConv2d(Box<DeformConv2dOpIr>),
    /// Operation corresponding to [deform_conv2d_backward](burn_tensor::ops::ModuleOps::deform_conv2d_backward)
    DeformableConv2dBackward(Box<DeformConv2dBackwardOpIr>),
    /// Operation corresponding to [conv transpose 1d](burn_tensor::ops::ModuleOps::conv_transpose1d).
    ConvTranspose1d(ConvTranspose1dOpIr),
    /// Operation corresponding to [conv transpose 2d](burn_tensor::ops::ModuleOps::conv_transpose2d).
    ConvTranspose2d(ConvTranspose2dOpIr),
    /// Operation corresponding to [conv transpose 3d](burn_tensor::ops::ModuleOps::conv_transpose3d).
    ConvTranspose3d(ConvTranspose3dOpIr),
    /// Operation corresponding to [avg pool 1d](burn_tensor::ops::ModuleOps::avg_pool1d).
    AvgPool1d(AvgPool1dOpIr),
    /// Operation corresponding to [avg pool 2d](burn_tensor::ops::ModuleOps::avg_pool2d).
    AvgPool2d(AvgPool2dOpIr),
    /// Operation corresponding to
    /// [avg pool 1d backward](burn_tensor::ops::ModuleOps::avg_pool1d_backward).
    AvgPool1dBackward(AvgPool1dBackwardOpIr),
    /// Operation corresponding to
    /// [avg pool 2d backward](burn_tensor::ops::ModuleOps::avg_pool2d_backward).
    AvgPool2dBackward(AvgPool2dBackwardOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 1d](burn_tensor::ops::ModuleOps::adaptive_avg_pool1d).
    AdaptiveAvgPool1d(AdaptiveAvgPool1dOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 2d](burn_tensor::ops::ModuleOps::adaptive_avg_pool2d).
    AdaptiveAvgPool2d(AdaptiveAvgPool2dOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 1d backward](burn_tensor::ops::ModuleOps::adaptive_avg_pool1d_backward).
    AdaptiveAvgPool1dBackward(AdaptiveAvgPool1dBackwardOpIr),
    /// Operation corresponding to
    /// [adaptive avg pool 2d backward](burn_tensor::ops::ModuleOps::adaptive_avg_pool2d_backward).
    AdaptiveAvgPool2dBackward(AdaptiveAvgPool2dBackwardOpIr),
    /// Operation corresponding to
    /// [max pool 1d](burn_tensor::ops::ModuleOps::max_pool1d).
    MaxPool1d(MaxPool1dOpIr),
    /// Operation corresponding to
    /// [max pool 1d with indices](burn_tensor::ops::ModuleOps::max_pool1d_with_indices).
    MaxPool1dWithIndices(MaxPool1dWithIndicesOpIr),
    /// Operation corresponding to
    /// [max pool 1d with indices backward](burn_tensor::ops::ModuleOps::max_pool1d_with_indices_backward).
    MaxPool1dWithIndicesBackward(MaxPool1dWithIndicesBackwardOpIr),
    /// Operation corresponding to
    /// [max pool 2d](burn_tensor::ops::ModuleOps::max_pool1d).
    MaxPool2d(MaxPool2dOpIr),
    /// Operation corresponding to
    /// [max pool 2d with indices](burn_tensor::ops::ModuleOps::max_pool2d_with_indices).
    MaxPool2dWithIndices(MaxPool2dWithIndicesOpIr),
    /// Operation corresponding to
    /// [max pool 2d with indices backward](burn_tensor::ops::ModuleOps::max_pool2d_with_indices_backward).
    MaxPool2dWithIndicesBackward(MaxPool2dWithIndicesBackwardOpIr),
    /// Operation corresponding to [interpolate](burn_tensor::ops::ModuleOps::interpolate).
    Interpolate(InterpolateOpIr),
    /// Operation corresponding to [interpolate backward](burn_tensor::ops::ModuleOps::interpolate_backward).
    InterpolateBackward(InterpolateBackwardOpIr),
}

/// Basic operations that can be done on any tensor type.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BaseOperationIr {
    /// Operation corresponding to:
    ///
    /// Float => [to device](burn_tensor::ops::FloatTensorOps::float_to_device).
    /// Int => [to device](burn_tensor::ops::IntTensorOps::int_to_device).
    /// Bool => [to device](burn_tensor::ops::BoolTensorOps::bool_to_device).
    ToDevice(TensorIr),
    /// Operation corresponding to:
    ///
    /// Float => [reshape](burn_tensor::ops::FloatTensorOps::float_reshape).
    /// Int => [reshape](burn_tensor::ops::IntTensorOps::int_reshape).
    /// Bool => [reshape](burn_tensor::ops::BoolTensorOps::bool_reshape).
    Reshape(UnaryOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [swap_dims](burn_tensor::ops::FloatTensorOps::float_swap_dims).
    /// Int => [swap_dims](burn_tensor::ops::IntTensorOps::int_swap_dims).
    /// Bool => [swap_dims](burn_tensor::ops::BoolTensorOps::bool_swap_dims).
    SwapDims(SwapDimsOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [permute](burn_tensor::ops::FloatTensorOps::float_permute).
    /// Int => [permute](burn_tensor::ops::IntTensorOps::int_permute).
    /// Bool => [permute](burn_tensor::ops::BoolTensorOps::bool_permute).
    Permute(PermuteOpIr),

    /// Operation corresponding to:
    /// Float => [flip](burn_tensor::ops::FloatTensorOps::float_flip).
    /// Int => [flip](burn_tensor::ops::IntTensorOps::int_flip).
    /// Bool => [flip](burn_tensor::ops::BoolTensorOps::bool_flip).
    Flip(FlipOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [expand](burn_tensor::ops::FloatTensorOps::float_expand).
    /// Int => [expand](burn_tensor::ops::IntTensorOps::int_expand).
    /// Bool => [expand](burn_tensor::ops::BoolTensorOps::bool_expand).
    Expand(ExpandOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [slice](burn_tensor::ops::FloatTensorOps::float_slice).
    /// Int => [slice](burn_tensor::ops::IntTensorOps::int_slice).
    /// Bool => [slice](burn_tensor::ops::BoolTensorOps::bool_slice).
    Slice(SliceOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [slice assign](burn_tensor::ops::FloatTensorOps::float_slice_assign).
    /// Int => [slice assign](burn_tensor::ops::IntTensorOps::int_slice_assign).
    /// Bool => [slice assign](burn_tensor::ops::BoolTensorOps::bool_slice_assign).
    SliceAssign(SliceAssignOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [equal](burn_tensor::ops::FloatTensorOps::float_equal).
    /// Int => [equal](burn_tensor::ops::IntTensorOps::int_equal).
    /// Bool => [equal](burn_tensor::ops::BoolTensorOps::bool_equal).
    Equal(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [repeat dim](burn_tensor::ops::FloatTensorOps::float_repeat_dim).
    /// Int => [repeat dim](burn_tensor::ops::IntTensorOps::int_repeat_dim).
    /// Bool => [repeat dim](burn_tensor::ops::BoolTensorOps::bool_repeat_dim).
    RepeatDim(RepeatDimOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [cat](burn_tensor::ops::FloatTensorOps::float_cat).
    /// Int => [cat](burn_tensor::ops::IntTensorOps::int_cat).
    /// Bool => [cat](burn_tensor::ops::BoolTensorOps::bool_cat).
    Cat(CatOpIr),
    /// Cast operation, no direct operation and should be supported by fusion backend.
    Cast(UnaryOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [empty](burn_tensor::ops::FloatTensorOps::float_empty).
    /// Int => [empty](burn_tensor::ops::IntTensorOps::int_empty).
    /// Bool => [empty](burn_tensor::ops::BoolTensorOps::bool_empty).
    Empty(TensorIr),
}

/// Numeric operations on int and float tensors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NumericOperationIr<E> {
    /// Operation corresponding to:
    ///
    /// Float => [add](burn_tensor::ops::FloatTensorOps::float_add).
    /// Int => [add](burn_tensor::ops::IntTensorOps::int_add).
    Add(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [add scalar](burn_tensor::ops::FloatTensorOps::float_add_scalar).
    /// Int => [add scalar](burn_tensor::ops::IntTensorOps::int_add_scalar).
    AddScalar(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [sub](burn_tensor::ops::FloatTensorOps::float_sub).
    /// Int => [sub](burn_tensor::ops::IntTensorOps::int_sub).
    Sub(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sub scalar](burn_tensor::ops::FloatTensorOps::float_sub_scalar).
    /// Int => [sub scalar](burn_tensor::ops::IntTensorOps::int_sub_scalar).
    SubScalar(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [div](burn_tensor::ops::FloatTensorOps::float_div).
    /// Int => [div](burn_tensor::ops::IntTensorOps::int_div).
    Div(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [div scalar](burn_tensor::ops::FloatTensorOps::float_div_scalar).
    /// Int => [div scalar](burn_tensor::ops::IntTensorOps::int_div_scalar).
    DivScalar(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [rem](burn_tensor::ops::FloatTensorOps::float_remainder).
    /// Int => [rem](burn_tensor::ops::IntTensorOps::int_remainder).
    Rem(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [rem scalar](burn_tensor::ops::FloatTensorOps::float_remainder_scalar).
    /// Int => [rem scalar](burn_tensor::ops::IntTensorOps::int_remainder_scalar).
    RemScalar(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mul](burn_tensor::ops::FloatTensorOps::float_mul).
    /// Int => [mul](burn_tensor::ops::IntTensorOps::int_mul).
    Mul(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mul scalar](burn_tensor::ops::FloatTensorOps::float_mul_scalar).
    /// Int => [mul scalar](burn_tensor::ops::IntTensorOps::int_mul_scalar).
    MulScalar(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [abs](burn_tensor::ops::FloatTensorOps::float_abs).
    /// Int => [abs](burn_tensor::ops::IntTensorOps::int_abs).
    Abs(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [ones](burn_tensor::ops::FloatTensorOps::float_ones).
    /// Int => [ones](burn_tensor::ops::IntTensorOps::int_ones).
    Ones(TensorIr),
    /// Operation corresponding to:
    ///
    /// Float => [zeros](burn_tensor::ops::FloatTensorOps::float_zeros).
    /// Int => [zeros](burn_tensor::ops::IntTensorOps::int_zeros).
    Zeros(TensorIr),
    /// Operation corresponding to:
    ///
    /// Float => [full](burn_tensor::ops::FloatTensorOps::float_full).
    /// Int => [full](burn_tensor::ops::IntTensorOps::int_full).
    Full((TensorIr, E)),
    /// Operation corresponding to:
    ///
    /// Float => [gather](burn_tensor::ops::FloatTensorOps::float_gather).
    /// Int => [gather](burn_tensor::ops::IntTensorOps::int_gather).
    Gather(GatherOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [scatter](burn_tensor::ops::FloatTensorOps::float_scatter).
    /// Int => [scatter](burn_tensor::ops::IntTensorOps::int_scatter).
    Scatter(ScatterOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [select](burn_tensor::ops::FloatTensorOps::float_select).
    /// Int => [select](burn_tensor::ops::IntTensorOps::int_select).
    Select(SelectOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [select assign](burn_tensor::ops::FloatTensorOps::float_select_assign).
    /// Int => [select assign](burn_tensor::ops::IntTensorOps::int_select_assign).
    SelectAssign(SelectAssignOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mask where](burn_tensor::ops::FloatTensorOps::float_mask_where).
    /// Int => [mask where](burn_tensor::ops::IntTensorOps::int_mask_where).
    MaskWhere(MaskWhereOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [mask fill](burn_tensor::ops::FloatTensorOps::float_mask_fill).
    /// Int => [mask fill](burn_tensor::ops::IntTensorOps::int_mask_fill).
    MaskFill(MaskFillOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mean dim](burn_tensor::ops::FloatTensorOps::float_mean_dim).
    /// Int => [mean dim](burn_tensor::ops::IntTensorOps::int_mean_dim).
    MeanDim(ScalarOpIr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [mean](burn_tensor::ops::FloatTensorOps::float_mean).
    /// Int => [mean](burn_tensor::ops::IntTensorOps::int_mean).
    Mean(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sum](burn_tensor::ops::FloatTensorOps::float_sum).
    /// Int => [sum](burn_tensor::ops::IntTensorOps::int_sum).
    Sum(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [sum dim](burn_tensor::ops::FloatTensorOps::float_sum_dim).
    /// Int => [sum dim](burn_tensor::ops::IntTensorOps::int_sum_dim).
    SumDim(ScalarOpIr<usize>),

    /// Operation corresponding to:
    ///
    /// Float => [prod](burn_tensor::ops::FloatTensorOps::float_prod).
    /// Int => [prod](burn_tensor::ops::IntTensorOps::int_prod).
    Prod(UnaryOpIr),

    /// Operation corresponding to:
    ///
    /// Float => [prod dim](burn_tensor::ops::FloatTensorOps::float_prod_dim).
    /// Int => [prod dim](burn_tensor::ops::IntTensorOps::int_prod_dim).
    ProdDim(ScalarOpIr<usize>),

    /// Operation corresponding to:
    ///
    /// Float => [equal elem](burn_tensor::ops::FloatTensorOps::float_equal_elem).
    /// Int => [equal elem](burn_tensor::ops::IntTensorOps::int_equal_elem).
    EqualElem(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater](burn_tensor::ops::FloatTensorOps::float_greater).
    /// Int => [greater](burn_tensor::ops::IntTensorOps::int_greater).
    Greater(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [greater elem](burn_tensor::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterElem(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal](burn_tensor::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](burn_tensor::ops::IntTensorOps::int_greater_elem).
    GreaterEqual(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal elem](burn_tensor::ops::FloatTensorOps::float_greater_equal_elem).
    /// Int => [greater equal elem](burn_tensor::ops::IntTensorOps::int_greater_equal_elem).
    GreaterEqualElem(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower](burn_tensor::ops::FloatTensorOps::float_lower).
    /// Int => [lower](burn_tensor::ops::IntTensorOps::int_lower).
    Lower(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [lower elem](burn_tensor::ops::FloatTensorOps::float_lower_elem).
    /// Int => [lower elem](burn_tensor::ops::IntTensorOps::int_lower_elem).
    LowerElem(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal](burn_tensor::ops::FloatTensorOps::float_lower_equal).
    /// Int => [lower equal](burn_tensor::ops::IntTensorOps::int_lower_equal).
    LowerEqual(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal elem](burn_tensor::ops::FloatTensorOps::float_lower_equal_elem).
    /// Int => [lower equal elem](burn_tensor::ops::IntTensorOps::int_lower_equal_elem).
    LowerEqualElem(ScalarOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [argmax](burn_tensor::ops::FloatTensorOps::float_argmax).
    /// Int => [argmax](burn_tensor::ops::IntTensorOps::int_argmax).
    ArgMax(ScalarOpIr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [argmin](burn_tensor::ops::FloatTensorOps::float_argmin).
    /// Int => [argmin](burn_tensor::ops::IntTensorOps::int_argmin).
    ArgMin(ScalarOpIr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [max](burn_tensor::ops::FloatTensorOps::float_max).
    /// Int => [max](burn_tensor::ops::IntTensorOps::int_max).
    Max(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max dim with indices](burn_tensor::ops::FloatTensorOps::float_max_dim_with_indices).
    /// Int => [max dim with indices](burn_tensor::ops::IntTensorOps::int_max_dim_with_indices).
    MaxDimWithIndices(ReduceDimWithIndicesOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [min dim with indices](burn_tensor::ops::FloatTensorOps::float_min_dim_with_indices).
    /// Int => [min dim with indices](burn_tensor::ops::IntTensorOps::int_min_dim_with_indices).
    MinDimWithIndices(ReduceDimWithIndicesOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [min](burn_tensor::ops::FloatTensorOps::float_min).
    /// Int => [min](burn_tensor::ops::IntTensorOps::int_min).
    Min(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [max dim](burn_tensor::ops::FloatTensorOps::float_max_dim).
    /// Int => [max dim](burn_tensor::ops::IntTensorOps::int_max_dim).
    MaxDim(ScalarOpIr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [min dim](burn_tensor::ops::FloatTensorOps::float_min_dim).
    /// Int => [min dim](burn_tensor::ops::IntTensorOps::int_min_dim).
    MinDim(ScalarOpIr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [clamp](burn_tensor::ops::FloatTensorOps::float_clamp).
    /// Int => [clamp](burn_tensor::ops::IntTensorOps::int_clamp).
    Clamp(ClampOpIr<E>),
    /// Operation corresponding to:
    ///
    /// Int => [random](burn_tensor::ops::IntTensorOps::int_random).
    IntRandom(RandomOpIr),
    /// Operation corresponding to:
    ///
    /// Float => [powf](burn_tensor::ops::FloatTensorOps::float_powf).
    /// Int => [powf](burn_tensor::ops::IntTensorOps::int_powf).
    Powf(BinaryOpIr),
}

/// Operation intermediate representation specific to an int tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum IntOperationIr {
    /// Operation corresponding to [into float](burn_tensor::ops::IntTensorOps::int_into_float).
    IntoFloat(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise and](burn_tensor::ops::IntTensorOps::bitwise_and).
    BitwiseAnd(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise and scalar](burn_tensor::ops::IntTensorOps::bitwise_and_scalar).
    BitwiseAndScalar(ScalarOpIr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise or](burn_tensor::ops::IntTensorOps::bitwise_or).
    BitwiseOr(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise or scalar](burn_tensor::ops::IntTensorOps::bitwise_or_scalar).
    BitwiseOrScalar(ScalarOpIr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise xor](burn_tensor::ops::IntTensorOps::bitwise_xor).
    BitwiseXor(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise xor scalar](burn_tensor::ops::IntTensorOps::bitwise_xor_scalar).
    BitwiseXorScalar(ScalarOpIr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise not](burn_tensor::ops::IntTensorOps::bitwise_not).
    BitwiseNot(UnaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise left shift](burn_tensor::ops::IntTensorOps::bitwise_left_shift).
    BitwiseLeftShift(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise left shift scalar](burn_tensor::ops::IntTensorOps::bitwise_left_shift_scalar).
    BitwiseLeftShiftScalar(ScalarOpIr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise right shift](burn_tensor::ops::IntTensorOps::bitwise_right_shift).
    BitwiseRightShift(BinaryOpIr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise right shift scalar](burn_tensor::ops::IntTensorOps::bitwise_right_shift_scalar).
    BitwiseRightShiftScalar(ScalarOpIr<i32>),
}

/// Operation intermediate representation specific to a bool tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BoolOperationIr {
    /// Operation corresponding to [into float](burn_tensor::ops::BoolTensorOps::bool_into_float).
    IntoFloat(UnaryOpIr),
    /// Operation corresponding to [into int](burn_tensor::ops::BoolTensorOps::bool_into_int).
    IntoInt(UnaryOpIr),
    /// Operation corresponding to [not](burn_tensor::ops::BoolTensorOps::bool_not).
    Not(UnaryOpIr),
    /// Operation corresponding to [and](burn_tensor::ops::BoolTensorOps::bool_and).
    And(BinaryOpIr),
    /// Operation corresponding to [or](burn_tensor::ops::BoolTensorOps::bool_or).
    Or(BinaryOpIr),
}

/// Swap dim operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct SwapDimsOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The first dim to swap.
    pub dim1: usize,
    /// The second dim to swap.
    pub dim2: usize,
}

/// Permute operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct PermuteOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The new order of the dimensions.
    pub axes: Vec<usize>,
}

/// Expand operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExpandOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The new shape.
    pub shape: Vec<usize>,
}

/// Flip operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct FlipOpIr {
    /// Input tensor intermediate representation.
    pub input: TensorIr,
    /// Output tensor intermediate representation.
    pub out: TensorIr,
    /// The dimensions to flip.
    pub axes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RandomOpIr {
    pub out: TensorIr,
    pub distribution: Distribution,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Declares a tensor has been initialized.
///
/// It is necessary to register for proper orphan detection and avoid memory leak.
pub struct InitOperationIr {
    /// The initialized tensor.
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct BinaryOpIr {
    pub lhs: TensorIr,
    pub rhs: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct UnaryOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarOpIr<E> {
    pub lhs: TensorIr,
    pub rhs: E,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GatherOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScatterOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub value: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectAssignOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub indices: TensorIr,
    pub value: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceOpIr {
    pub tensor: TensorIr,
    pub ranges: Vec<Range<usize>>,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceAssignOpIr {
    pub tensor: TensorIr,
    pub ranges: Vec<Range<usize>>,
    pub value: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskWhereOpIr {
    pub tensor: TensorIr,
    pub mask: TensorIr,
    pub value: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskFillOpIr<E> {
    pub tensor: TensorIr,
    pub mask: TensorIr,
    pub value: E,
    pub out: TensorIr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ClampOpIr<E> {
    pub tensor: TensorIr,
    pub min: E,
    pub max: E,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RepeatDimOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub times: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CatOpIr {
    pub tensors: Vec<TensorIr>,
    pub dim: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ReduceDimWithIndicesOpIr {
    pub tensor: TensorIr,
    pub dim: usize,
    pub out: TensorIr,
    pub out_indices: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingOpIr {
    pub weights: TensorIr,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingBackwardOpIr {
    pub weights: TensorIr,
    pub out_grad: TensorIr,
    pub indices: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: Conv1dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: Conv2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformConv2dOpIr {
    pub x: TensorIr,
    pub offset: TensorIr,
    pub weight: TensorIr,
    pub mask: Option<TensorIr>,
    pub bias: Option<TensorIr>,
    pub options: DeformableConv2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformConv2dBackwardOpIr {
    pub x: TensorIr,
    pub offset: TensorIr,
    pub weight: TensorIr,
    pub mask: Option<TensorIr>,
    pub bias: Option<TensorIr>,
    pub out_grad: TensorIr,
    pub options: DeformableConv2dOptionsIr,
    pub input_grad: TensorIr,
    pub offset_grad: TensorIr,
    pub weight_grad: TensorIr,
    pub mask_grad: Option<TensorIr>,
    pub bias_grad: Option<TensorIr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: Conv3dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: ConvTranspose1dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: ConvTranspose2dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose3dOpIr {
    pub x: TensorIr,
    pub weight: TensorIr,
    pub bias: Option<TensorIr>,
    pub options: ConvTranspose3dOptionsIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOptionsIr {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOptionsIr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformableConv2dOptionsIr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub weight_groups: usize,
    pub offset_groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dOptionsIr {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOptionsIr {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub padding_out: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOptionsIr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose3dOptionsIr {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub padding_out: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

/// Quantization parameters intermediate representation.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationParametersIr {
    /// The scaling factor.
    pub scale: TensorIr,
    /// The zero-point offset.
    pub offset: Option<TensorIr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct QuantizeOpIr {
    pub tensor: TensorIr,
    pub qparams: QuantizationParametersIr,
    pub scheme: QuantizationScheme,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DequantizeOpIr {
    pub input: TensorIr,
    pub out: TensorIr,
}

impl From<ConvOptions<1>> for Conv1dOptionsIr {
    fn from(value: ConvOptions<1>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<2>> for Conv2dOptionsIr {
    fn from(value: ConvOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<3>> for Conv3dOptionsIr {
    fn from(value: ConvOptions<3>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<DeformConvOptions<2>> for DeformableConv2dOptionsIr {
    fn from(value: DeformConvOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            weight_groups: value.weight_groups,
            offset_groups: value.offset_groups,
        }
    }
}

impl From<ConvTransposeOptions<1>> for ConvTranspose1dOptionsIr {
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

impl From<ConvTransposeOptions<2>> for ConvTranspose2dOptionsIr {
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

impl From<ConvTransposeOptions<3>> for ConvTranspose3dOptionsIr {
    fn from(value: ConvTransposeOptions<3>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            padding_out: value.padding_out,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<Conv1dOptionsIr> for ConvOptions<1> {
    fn from(val: Conv1dOptionsIr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv2dOptionsIr> for ConvOptions<2> {
    fn from(val: Conv2dOptionsIr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv3dOptionsIr> for ConvOptions<3> {
    fn from(val: Conv3dOptionsIr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<DeformableConv2dOptionsIr> for DeformConvOptions<2> {
    fn from(value: DeformableConv2dOptionsIr) -> Self {
        DeformConvOptions {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            weight_groups: value.weight_groups,
            offset_groups: value.offset_groups,
        }
    }
}

impl From<ConvTranspose1dOptionsIr> for ConvTransposeOptions<1> {
    fn from(val: ConvTranspose1dOptionsIr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose2dOptionsIr> for ConvTransposeOptions<2> {
    fn from(val: ConvTranspose2dOptionsIr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose3dOptionsIr> for ConvTransposeOptions<3> {
    fn from(val: ConvTranspose3dOptionsIr) -> Self {
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
pub struct AvgPool1dOpIr {
    pub x: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dOpIr {
    pub x: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool1dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dOpIr {
    pub x: TensorIr,
    pub output_size: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dOpIr {
    pub x: TensorIr,
    pub output_size: [usize; 2],
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dOpIr {
    pub x: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesOpIr {
    pub x: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorIr,
    pub out_indices: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub indices: TensorIr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool2dOpIr {
    pub x: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorIr,
}

#[allow(missing_docs)]
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct MaxPool2dWithIndicesOpIr {
    pub x: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorIr,
    pub out_indices: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool2dWithIndicesBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub indices: TensorIr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum InterpolateModeIr {
    Nearest,
    Bilinear,
    Bicubic,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOptionsIr {
    pub mode: InterpolateModeIr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOpIr {
    pub x: TensorIr,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsIr,
    pub out: TensorIr,
}

impl From<InterpolateModeIr> for InterpolateMode {
    fn from(val: InterpolateModeIr) -> Self {
        match val {
            InterpolateModeIr::Nearest => Self::Nearest,
            InterpolateModeIr::Bilinear => Self::Bilinear,
            InterpolateModeIr::Bicubic => Self::Bicubic,
        }
    }
}

impl From<InterpolateOptionsIr> for InterpolateOptions {
    fn from(val: InterpolateOptionsIr) -> Self {
        Self {
            mode: val.mode.into(),
        }
    }
}

impl From<InterpolateMode> for InterpolateModeIr {
    fn from(val: InterpolateMode) -> Self {
        match val {
            InterpolateMode::Nearest => Self::Nearest,
            InterpolateMode::Bilinear => Self::Bilinear,
            InterpolateMode::Bicubic => Self::Bicubic,
        }
    }
}

impl From<InterpolateOptions> for InterpolateOptionsIr {
    fn from(val: InterpolateOptions) -> Self {
        Self {
            mode: val.mode.into(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateBackwardOpIr {
    pub x: TensorIr,
    pub grad: TensorIr,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsIr,
    pub out: TensorIr,
}

impl OperationIr {
    /// Cleanup the remaining tensor handles that have not been used.
    pub fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            OperationIr::BaseFloat(repr) => repr.nodes(),
            OperationIr::BaseInt(repr) => repr.nodes(),
            OperationIr::BaseBool(repr) => repr.nodes(),
            OperationIr::NumericFloat(_dtype, repr) => repr.nodes(),
            OperationIr::NumericInt(_dtype, repr) => repr.nodes(),
            OperationIr::Bool(repr) => repr.nodes(),
            OperationIr::Int(repr) => repr.nodes(),
            OperationIr::Float(_dtype, repr) => repr.nodes(),
            OperationIr::Module(repr) => repr.nodes(),
            OperationIr::Init(repr) => repr.nodes(),
            OperationIr::Custom(repr) => repr.nodes(),
        }
    }
}

impl BaseOperationIr {
    fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            BaseOperationIr::ToDevice(repr) => vec![repr],
            BaseOperationIr::Reshape(repr) => {
                vec![&repr.input, &repr.out]
            }
            BaseOperationIr::SwapDims(repr) => {
                vec![&repr.input, &repr.out]
            }
            BaseOperationIr::Permute(repr) => {
                vec![&repr.input, &repr.out]
            }

            BaseOperationIr::Expand(repr) => {
                vec![&repr.input, &repr.out]
            }

            BaseOperationIr::Flip(repr) => {
                vec![&repr.input, &repr.out]
            }
            BaseOperationIr::Slice(repr) => {
                vec![&repr.tensor, &repr.out]
            }
            BaseOperationIr::SliceAssign(repr) => {
                vec![&repr.tensor, &repr.value, &repr.out]
            }
            BaseOperationIr::Equal(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            BaseOperationIr::RepeatDim(repr) => {
                vec![&repr.tensor, &repr.out]
            }
            BaseOperationIr::Cat(repr) => repr.tensors.iter().collect(),
            BaseOperationIr::Cast(repr) => vec![&repr.input, &repr.out],
            BaseOperationIr::Empty(repr) => vec![repr],
        }
    }
}

impl<E: Element> NumericOperationIr<E> {
    fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            NumericOperationIr::Add(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::AddScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Sub(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::SubScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Mul(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::MulScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Div(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::DivScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Rem(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::RemScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Ones(repr) => vec![repr],
            NumericOperationIr::Gather(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.out]
            }
            NumericOperationIr::Scatter(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.value, &repr.out]
            }
            NumericOperationIr::Select(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.out]
            }
            NumericOperationIr::SelectAssign(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.value, &repr.out]
            }
            NumericOperationIr::MaskWhere(repr) => {
                vec![&repr.tensor, &repr.mask, &repr.value, &repr.out]
            }
            NumericOperationIr::MaskFill(repr) => {
                vec![&repr.tensor, &repr.mask, &repr.out]
            }
            NumericOperationIr::EqualElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::GreaterElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::GreaterEqualElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::LowerElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::LowerEqualElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Greater(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::GreaterEqual(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::Lower(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::LowerEqual(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationIr::ArgMax(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::ArgMin(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Clamp(repr) => {
                vec![&repr.tensor, &repr.out]
            }
            NumericOperationIr::Abs(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationIr::Zeros(repr) => vec![repr],
            NumericOperationIr::Full(repr) => vec![&repr.0],
            NumericOperationIr::MeanDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Mean(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationIr::Sum(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationIr::SumDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Prod(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationIr::ProdDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::Max(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationIr::MaxDimWithIndices(repr) => {
                vec![&repr.tensor, &repr.out_indices, &repr.out]
            }
            NumericOperationIr::MinDimWithIndices(repr) => {
                vec![&repr.tensor, &repr.out_indices, &repr.out]
            }
            NumericOperationIr::Min(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationIr::MaxDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::MinDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationIr::IntRandom(repr) => {
                vec![&repr.out]
            }
            NumericOperationIr::Powf(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
        }
    }
}

impl FloatOperationIr {
    fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            FloatOperationIr::Matmul(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            FloatOperationIr::Random(repr) => vec![&repr.out],
            FloatOperationIr::Exp(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Log(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Log1p(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Erf(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Recip(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::PowfScalar(repr) => vec![&repr.lhs, &repr.out],
            FloatOperationIr::Sqrt(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Cos(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Sin(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Tanh(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Round(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Floor(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Ceil(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::IntoInt(repr) => vec![&repr.input, &repr.out],
            FloatOperationIr::Quantize(repr) => {
                if let Some(offset) = &repr.qparams.offset {
                    vec![&repr.tensor, &repr.qparams.scale, &offset, &repr.out]
                } else {
                    vec![&repr.tensor, &repr.qparams.scale, &repr.out]
                }
            }
            FloatOperationIr::Dequantize(repr) => vec![&repr.input, &repr.out],
        }
    }
}

impl IntOperationIr {
    fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            IntOperationIr::IntoFloat(repr) => vec![&repr.input, &repr.out],
            IntOperationIr::BitwiseAnd(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationIr::BitwiseAndScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationIr::BitwiseOr(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationIr::BitwiseOrScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationIr::BitwiseXor(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationIr::BitwiseXorScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationIr::BitwiseNot(repr) => {
                vec![&repr.input, &repr.out]
            }
            IntOperationIr::BitwiseLeftShift(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationIr::BitwiseLeftShiftScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationIr::BitwiseRightShift(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationIr::BitwiseRightShiftScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
        }
    }
}

impl BoolOperationIr {
    fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            BoolOperationIr::IntoFloat(repr) => vec![&repr.input, &repr.out],
            BoolOperationIr::IntoInt(repr) => vec![&repr.input, &repr.out],
            BoolOperationIr::Not(repr) => vec![&repr.input, &repr.out],
            BoolOperationIr::And(repr) => vec![&repr.lhs, &repr.rhs, &repr.out],
            BoolOperationIr::Or(repr) => vec![&repr.lhs, &repr.rhs, &repr.out],
        }
    }
}

impl ModuleOperationIr {
    fn nodes(&self) -> Vec<&TensorIr> {
        match self {
            ModuleOperationIr::Embedding(repr) => {
                vec![&repr.weights, &repr.indices, &repr.out]
            }
            ModuleOperationIr::EmbeddingBackward(repr) => {
                vec![&repr.weights, &repr.out_grad, &repr.indices, &repr.out]
            }
            ModuleOperationIr::Conv1d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationIr::Conv2d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationIr::Conv3d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationIr::DeformableConv2d(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => vec![&repr.x, &repr.offset, &repr.weight, &mask, &bias],
                (Some(mask), None) => vec![&repr.x, &repr.offset, &repr.weight, &mask],
                (None, Some(bias)) => vec![&repr.x, &repr.offset, &repr.weight, &bias],
                (None, None) => vec![&repr.x, &repr.offset, &repr.weight],
            },
            ModuleOperationIr::DeformableConv2dBackward(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => {
                    vec![&repr.x, &repr.offset, &repr.weight, &mask, &bias]
                }
                (Some(mask), None) => vec![&repr.x, &repr.offset, &repr.weight, &mask],
                (None, Some(bias)) => vec![&repr.x, &repr.offset, &repr.weight, &bias],
                (None, None) => vec![&repr.x, &repr.offset, &repr.weight],
            },
            ModuleOperationIr::ConvTranspose1d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationIr::ConvTranspose2d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationIr::ConvTranspose3d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationIr::AvgPool1d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::AvgPool2d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::AvgPool1dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationIr::AvgPool2dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationIr::AdaptiveAvgPool1d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::AdaptiveAvgPool2d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::AdaptiveAvgPool1dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationIr::AdaptiveAvgPool2dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationIr::MaxPool1d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::MaxPool1dWithIndices(repr) => {
                vec![&repr.x, &repr.out, &repr.out_indices]
            }
            ModuleOperationIr::MaxPool1dWithIndicesBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.indices, &repr.grad]
            }
            ModuleOperationIr::MaxPool2d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::MaxPool2dWithIndices(repr) => {
                vec![&repr.x, &repr.out, &repr.out_indices]
            }
            ModuleOperationIr::MaxPool2dWithIndicesBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.indices, &repr.grad]
            }
            ModuleOperationIr::Interpolate(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationIr::InterpolateBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
        }
    }
}

impl core::hash::Hash for InitOperationIr {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.out.hash(state);
    }
}

impl InitOperationIr {
    fn nodes(&self) -> Vec<&TensorIr> {
        vec![&self.out]
    }
}

impl core::hash::Hash for RandomOpIr {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.out.hash(state);

        match self.distribution {
            Distribution::Default => 1u8.hash(state),
            Distribution::Bernoulli(_) => 2u8.hash(state),
            Distribution::Uniform(_, _) => 3u8.hash(state),
            Distribution::Normal(_, _) => 4u8.hash(state),
        }
    }
}

impl<E> core::hash::Hash for ScalarOpIr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.lhs.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for MaskFillOpIr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.mask.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for ClampOpIr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for NumericOperationIr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            NumericOperationIr::Add(repr) => repr.hash(state),
            NumericOperationIr::AddScalar(repr) => repr.hash(state),
            NumericOperationIr::Sub(repr) => repr.hash(state),
            NumericOperationIr::SubScalar(repr) => repr.hash(state),
            NumericOperationIr::Div(repr) => repr.hash(state),
            NumericOperationIr::DivScalar(repr) => repr.hash(state),
            NumericOperationIr::Rem(repr) => repr.hash(state),
            NumericOperationIr::RemScalar(repr) => repr.hash(state),
            NumericOperationIr::Mul(repr) => repr.hash(state),
            NumericOperationIr::MulScalar(repr) => repr.hash(state),
            NumericOperationIr::Abs(repr) => repr.hash(state),
            NumericOperationIr::Ones(repr) => repr.hash(state),
            NumericOperationIr::Zeros(repr) => repr.hash(state),
            NumericOperationIr::Full(repr) => repr.0.hash(state),
            NumericOperationIr::Gather(repr) => repr.hash(state),
            NumericOperationIr::Scatter(repr) => repr.hash(state),
            NumericOperationIr::Select(repr) => repr.hash(state),
            NumericOperationIr::SelectAssign(repr) => repr.hash(state),
            NumericOperationIr::MaskWhere(repr) => repr.hash(state),
            NumericOperationIr::MaskFill(repr) => repr.hash(state),
            NumericOperationIr::MeanDim(repr) => repr.hash(state),
            NumericOperationIr::Mean(repr) => repr.hash(state),
            NumericOperationIr::Sum(repr) => repr.hash(state),
            NumericOperationIr::SumDim(repr) => repr.hash(state),
            NumericOperationIr::Prod(repr) => repr.hash(state),
            NumericOperationIr::ProdDim(repr) => repr.hash(state),
            NumericOperationIr::EqualElem(repr) => repr.hash(state),
            NumericOperationIr::Greater(repr) => repr.hash(state),
            NumericOperationIr::GreaterElem(repr) => repr.hash(state),
            NumericOperationIr::GreaterEqual(repr) => repr.hash(state),
            NumericOperationIr::GreaterEqualElem(repr) => repr.hash(state),
            NumericOperationIr::Lower(repr) => repr.hash(state),
            NumericOperationIr::LowerElem(repr) => repr.hash(state),
            NumericOperationIr::LowerEqual(repr) => repr.hash(state),
            NumericOperationIr::LowerEqualElem(repr) => repr.hash(state),
            NumericOperationIr::ArgMax(repr) => repr.hash(state),
            NumericOperationIr::ArgMin(repr) => repr.hash(state),
            NumericOperationIr::Max(repr) => repr.hash(state),
            NumericOperationIr::MaxDimWithIndices(repr) => repr.hash(state),
            NumericOperationIr::MinDimWithIndices(repr) => repr.hash(state),
            NumericOperationIr::Min(repr) => repr.hash(state),
            NumericOperationIr::MaxDim(repr) => repr.hash(state),
            NumericOperationIr::MinDim(repr) => repr.hash(state),
            NumericOperationIr::Clamp(repr) => repr.hash(state),
            NumericOperationIr::IntRandom(repr) => repr.hash(state),
            NumericOperationIr::Powf(repr) => repr.hash(state),
        }
    }
}
