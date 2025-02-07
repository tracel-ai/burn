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

use crate::TensorRepr;

/// Custom operation in fusion stream, declaring it's inputs and outputs.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct CustomOpRepr {
    /// Unique identifier of the operation.
    pub id: String,
    /// Input tensors used in this the custom operation.
    pub inputs: Vec<TensorRepr>,
    /// Output tensors used in this the custom operation.
    pub outputs: Vec<TensorRepr>,
}

impl CustomOpRepr {
    /// Create a new custom operation intermediate representation.
    pub fn new(id: &'static str, inputs: &[TensorRepr], outputs: &[TensorRepr]) -> Self {
        Self {
            id: id.to_owned(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        }
    }

    /// Consume the intermediate representation, and get the in and output tensors.
    pub fn consume<const N_IN: usize, const N_OUT: usize>(
        self,
    ) -> ([TensorRepr; N_IN], [TensorRepr; N_OUT]) {
        (
            self.inputs.try_into().expect(
                "Wrong number of inputs expected (expected {D}, is {}), check your implementation",
            ),
            self.outputs.try_into().expect(
                "Wrong number of outputs expected (expected {D}, is {}), check your implementation",
            ),
        )
    }

    fn nodes(&self) -> Vec<&TensorRepr> {
        self.inputs.iter().chain(self.outputs.iter()).collect()
    }
}

/// irribe all tensor operations possible.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum OperationRepr {
    /// Basic operation on a float tensor.
    BaseFloat(BaseOperationRepr),
    /// Basic operation on an int tensor.
    BaseInt(BaseOperationRepr),
    /// Basic operation on a bool tensor.
    BaseBool(BaseOperationRepr),
    /// Numeric operation on a float tensor.
    NumericFloat(DType, NumericOperationRepr<f32>),
    /// Numeric operation on an int tensor.
    NumericInt(DType, NumericOperationRepr<i32>),
    /// Operation specific to a bool tensor.
    Bool(BoolOperationRepr),
    /// Operation specific to an int tensor.
    Int(IntOperationRepr),
    /// Operation specific to a float tensor.
    Float(DType, FloatOperationRepr),
    /// Module operation.
    Module(ModuleOperationRepr),
    /// Initialize operation.
    Init(InitOperationRepr),
    /// A custom operation.
    Custom(CustomOpRepr),
}

/// Operation intermediate representation specific to a float tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum FloatOperationRepr {
    /// Operation corresponding to [exp](crate::ops::FloatTensorOps::float_exp).
    Exp(UnaryOpRepr),
    /// Operation corresponding to [log](crate::ops::FloatTensorOps::float_log).
    Log(UnaryOpRepr),
    /// Operation corresponding to [log1p](crate::ops::FloatTensorOps::float_log1p).
    Log1p(UnaryOpRepr),
    /// Operation corresponding to [erf](crate::ops::FloatTensorOps::float_erf).
    Erf(UnaryOpRepr),
    /// Operation corresponding to [powf_scalar](crate::ops::FloatTensorOps::float_powf_scalar).
    PowfScalar(ScalarOpRepr<f32>),
    /// Operation corresponding to [sqrt](crate::ops::FloatTensorOps::float_sqrt).
    Sqrt(UnaryOpRepr),
    /// Operation corresponding to [cos](crate::ops::FloatTensorOps::float_cos).
    Cos(UnaryOpRepr),
    /// Operation corresponding to [sin](crate::ops::FloatTensorOps::float_sin).
    Sin(UnaryOpRepr),
    /// Operation corresponding to [tanh](crate::ops::FloatTensorOps::float_tanh).
    Tanh(UnaryOpRepr),
    /// Operation corresponding to [round](crate::ops::FloatTensorOps::float_round).
    Round(UnaryOpRepr),
    /// Operation corresponding to [floor](crate::ops::FloatTensorOps::float_floor).
    Floor(UnaryOpRepr),
    /// Operation corresponding to [ceil](crate::ops::FloatTensorOps::float_ceil).
    Ceil(UnaryOpRepr),
    /// Operation corresponding to [into_int](crate::ops::FloatTensorOps::float_into_int).
    IntoInt(UnaryOpRepr),
    /// Operation corresponding to [matmul](crate::ops::FloatTensorOps::float_matmul).
    Matmul(BinaryOpRepr),
    /// Operation corresponding to [random](crate::ops::FloatTensorOps::float_random).
    Random(RandomOpRepr),
    /// Operation corresponding to [recip](crate::ops::FloatTensorOps::float_recip).
    Recip(UnaryOpRepr),
    /// Operation corresponding to [quantize](crate::ops::QTensorOps::quantize).
    Quantize(QuantizeOpRepr),
    /// Operation corresponding to [dequantize](crate::ops::QTensorOps::dequantize).
    Dequantize(DequantizeOpRepr),
}

/// Operation intermediate representation specific to module.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum ModuleOperationRepr {
    /// Operation corresponding to [embedding](crate::ops::ModuleOps::embedding).
    Embedding(EmbeddingOpRepr),
    /// Operation corresponding to [embedding_backward](crate::ops::ModuleOps::embedding_backward).
    EmbeddingBackward(EmbeddingBackwardOpRepr),
    /// Operation corresponding to [conv1d](crate::ops::ModuleOps::conv1d).
    Conv1d(Conv1dOpRepr),
    /// Operation corresponding to [conv2d](crate::ops::ModuleOps::conv2d).
    Conv2d(Conv2dOpRepr),
    /// Operation corresponding to [conv3d](crate::ops::ModuleOps::conv3d).
    Conv3d(Conv3dOpRepr),
    /// Operation corresponding to [deform_conv2d](crate::ops::ModuleOps::deform_conv2d)
    DeformableConv2d(Box<DeformConv2dOpRepr>),
    /// Operation corresponding to [deform_conv2d_backward](crate::ops::ModuleOps::deform_conv2d_backward)
    DeformableConv2dBackward(Box<DeformConv2dBackwardOpRepr>),
    /// Operation corresponding to [conv transpose 1d](crate::ops::ModuleOps::conv_transpose1d).
    ConvTranspose1d(ConvTranspose1dOpRepr),
    /// Operation corresponding to [conv transpose 2d](crate::ops::ModuleOps::conv_transpose2d).
    ConvTranspose2d(ConvTranspose2dOpRepr),
    /// Operation corresponding to [conv transpose 3d](crate::ops::ModuleOps::conv_transpose3d).
    ConvTranspose3d(ConvTranspose3dOpRepr),
    /// Operation corresponding to [avg pool 1d](crate::ops::ModuleOps::avg_pool1d).
    AvgPool1d(AvgPool1dOpRepr),
    /// Operation corresponding to [avg pool 2d](crate::ops::ModuleOps::avg_pool2d).
    AvgPool2d(AvgPool2dOpRepr),
    /// Operation corresponding to
    /// [avg pool 1d backward](crate::ops::ModuleOps::avg_pool1d_backward).
    AvgPool1dBackward(AvgPool1dBackwardOpRepr),
    /// Operation corresponding to
    /// [avg pool 2d backward](crate::ops::ModuleOps::avg_pool2d_backward).
    AvgPool2dBackward(AvgPool2dBackwardOpRepr),
    /// Operation corresponding to
    /// [adaptive avg pool 1d](crate::ops::ModuleOps::adaptive_avg_pool1d).
    AdaptiveAvgPool1d(AdaptiveAvgPool1dOpRepr),
    /// Operation corresponding to
    /// [adaptive avg pool 2d](crate::ops::ModuleOps::adaptive_avg_pool2d).
    AdaptiveAvgPool2d(AdaptiveAvgPool2dOpRepr),
    /// Operation corresponding to
    /// [adaptive avg pool 1d backward](crate::ops::ModuleOps::adaptive_avg_pool1d_backward).
    AdaptiveAvgPool1dBackward(AdaptiveAvgPool1dBackwardOpRepr),
    /// Operation corresponding to
    /// [adaptive avg pool 2d backward](crate::ops::ModuleOps::adaptive_avg_pool2d_backward).
    AdaptiveAvgPool2dBackward(AdaptiveAvgPool2dBackwardOpRepr),
    /// Operation corresponding to
    /// [max pool 1d](crate::ops::ModuleOps::max_pool1d).
    MaxPool1d(MaxPool1dOpRepr),
    /// Operation corresponding to
    /// [max pool 1d with indices](crate::ops::ModuleOps::max_pool1d_with_indices).
    MaxPool1dWithIndices(MaxPool1dWithIndicesOpRepr),
    /// Operation corresponding to
    /// [max pool 1d with indices backward](crate::ops::ModuleOps::max_pool1d_with_indices_backward).
    MaxPool1dWithIndicesBackward(MaxPool1dWithIndicesBackwardOpRepr),
    /// Operation corresponding to
    /// [max pool 2d](crate::ops::ModuleOps::max_pool1d).
    MaxPool2d(MaxPool2dOpRepr),
    /// Operation corresponding to
    /// [max pool 2d with indices](crate::ops::ModuleOps::max_pool2d_with_indices).
    MaxPool2dWithIndices(MaxPool2dWithIndicesOpRepr),
    /// Operation corresponding to
    /// [max pool 2d with indices backward](crate::ops::ModuleOps::max_pool2d_with_indices_backward).
    MaxPool2dWithIndicesBackward(MaxPool2dWithIndicesBackwardOpRepr),
    /// Operation corresponding to [interpolate](crate::ops::ModuleOps::interpolate).
    Interpolate(InterpolateOpRepr),
    /// Operation corresponding to [interpolate backward](crate::ops::ModuleOps::interpolate_backward).
    InterpolateBackward(InterpolateBackwardRepr),
}

/// Basic operations that can be done on any tensor type.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BaseOperationRepr {
    /// Operation corresponding to:
    ///
    /// Float => [to device](crate::ops::FloatTensorOps::float_to_device).
    /// Int => [to device](crate::ops::IntTensorOps::int_to_device).
    /// Bool => [to device](crate::ops::BoolTensorOps::bool_to_device).
    ToDevice(TensorRepr),
    /// Operation corresponding to:
    ///
    /// Float => [reshape](crate::ops::FloatTensorOps::float_reshape).
    /// Int => [reshape](crate::ops::IntTensorOps::int_reshape).
    /// Bool => [reshape](crate::ops::BoolTensorOps::bool_reshape).
    Reshape(UnaryOpRepr),

    /// Operation corresponding to:
    ///
    /// Float => [swap_dims](crate::ops::FloatTensorOps::float_swap_dims).
    /// Int => [swap_dims](crate::ops::IntTensorOps::int_swap_dims).
    /// Bool => [swap_dims](crate::ops::BoolTensorOps::bool_swap_dims).
    SwapDims(SwapDimsOpRepr),

    /// Operation corresponding to:
    ///
    /// Float => [permute](crate::ops::FloatTensorOps::float_permute).
    /// Int => [permute](crate::ops::IntTensorOps::int_permute).
    /// Bool => [permute](crate::ops::BoolTensorOps::bool_permute).
    Permute(PermuteOpRepr),

    /// Operation corresponding to:
    /// Float => [flip](crate::ops::FloatTensorOps::float_flip).
    /// Int => [flip](crate::ops::IntTensorOps::int_flip).
    /// Bool => [flip](crate::ops::BoolTensorOps::bool_flip).
    Flip(FlipOpRepr),

    /// Operation corresponding to:
    ///
    /// Float => [expand](crate::ops::FloatTensorOps::float_expand).
    /// Int => [expand](crate::ops::IntTensorOps::int_expand).
    /// Bool => [expand](crate::ops::BoolTensorOps::bool_expand).
    Expand(ExpandOpRepr),

    /// Operation corresponding to:
    ///
    /// Float => [slice](crate::ops::FloatTensorOps::float_slice).
    /// Int => [slice](crate::ops::IntTensorOps::int_slice).
    /// Bool => [slice](crate::ops::BoolTensorOps::bool_slice).
    Slice(SliceOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [slice assign](crate::ops::FloatTensorOps::float_slice_assign).
    /// Int => [slice assign](crate::ops::IntTensorOps::int_slice_assign).
    /// Bool => [slice assign](crate::ops::BoolTensorOps::bool_slice_assign).
    SliceAssign(SliceAssignOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [equal](crate::ops::FloatTensorOps::float_equal).
    /// Int => [equal](crate::ops::IntTensorOps::int_equal).
    /// Bool => [equal](crate::ops::BoolTensorOps::bool_equal).
    Equal(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [repeat dim](crate::ops::FloatTensorOps::float_repeat_dim).
    /// Int => [repeat dim](crate::ops::IntTensorOps::int_repeat_dim).
    /// Bool => [repeat dim](crate::ops::BoolTensorOps::bool_repeat_dim).
    RepeatDim(RepeatDimOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [cat](crate::ops::FloatTensorOps::float_cat).
    /// Int => [cat](crate::ops::IntTensorOps::int_cat).
    /// Bool => [cat](crate::ops::BoolTensorOps::bool_cat).
    Cat(CatOpRepr),
    /// Cast operation, no direct operation and should be supported by fusion backend.
    Cast(UnaryOpRepr),

    /// Operation corresponding to:
    ///
    /// Float => [empty](crate::ops::FloatTensorOps::float_empty).
    /// Int => [empty](crate::ops::IntTensorOps::int_empty).
    /// Bool => [empty](crate::ops::BoolTensorOps::bool_empty).
    Empty(TensorRepr),
}

/// Numeric operations on int and float tensors.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum NumericOperationRepr<E> {
    /// Operation corresponding to:
    ///
    /// Float => [add](crate::ops::FloatTensorOps::float_add).
    /// Int => [add](crate::ops::IntTensorOps::int_add).
    Add(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [add scalar](crate::ops::FloatTensorOps::float_add_scalar).
    /// Int => [add scalar](crate::ops::IntTensorOps::int_add_scalar).
    AddScalar(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [sub](crate::ops::FloatTensorOps::float_sub).
    /// Int => [sub](crate::ops::IntTensorOps::int_sub).
    Sub(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [sub scalar](crate::ops::FloatTensorOps::float_sub_scalar).
    /// Int => [sub scalar](crate::ops::IntTensorOps::int_sub_scalar).
    SubScalar(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [div](crate::ops::FloatTensorOps::float_div).
    /// Int => [div](crate::ops::IntTensorOps::int_div).
    Div(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [div scalar](crate::ops::FloatTensorOps::float_div_scalar).
    /// Int => [div scalar](crate::ops::IntTensorOps::int_div_scalar).
    DivScalar(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [rem](crate::ops::FloatTensorOps::float_remainder).
    /// Int => [rem](crate::ops::IntTensorOps::int_remainder).
    Rem(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [rem scalar](crate::ops::FloatTensorOps::float_remainder_scalar).
    /// Int => [rem scalar](crate::ops::IntTensorOps::int_remainder_scalar).
    RemScalar(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mul](crate::ops::FloatTensorOps::float_mul).
    /// Int => [mul](crate::ops::IntTensorOps::int_mul).
    Mul(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [mul scalar](crate::ops::FloatTensorOps::float_mul_scalar).
    /// Int => [mul scalar](crate::ops::IntTensorOps::int_mul_scalar).
    MulScalar(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [abs](crate::ops::FloatTensorOps::float_abs).
    /// Int => [abs](crate::ops::IntTensorOps::int_abs).
    Abs(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [ones](crate::ops::FloatTensorOps::float_ones).
    /// Int => [ones](crate::ops::IntTensorOps::int_ones).
    Ones(TensorRepr),
    /// Operation corresponding to:
    ///
    /// Float => [zeros](crate::ops::FloatTensorOps::float_zeros).
    /// Int => [zeros](crate::ops::IntTensorOps::int_zeros).
    Zeros(TensorRepr),
    /// Operation corresponding to:
    ///
    /// Float => [full](crate::ops::FloatTensorOps::float_full).
    /// Int => [full](crate::ops::IntTensorOps::int_full).
    Full((TensorRepr, E)),
    /// Operation corresponding to:
    ///
    /// Float => [gather](crate::ops::FloatTensorOps::float_gather).
    /// Int => [gather](crate::ops::IntTensorOps::int_gather).
    Gather(GatherOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [scatter](crate::ops::FloatTensorOps::float_scatter).
    /// Int => [scatter](crate::ops::IntTensorOps::int_scatter).
    Scatter(ScatterOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [select](crate::ops::FloatTensorOps::float_select).
    /// Int => [select](crate::ops::IntTensorOps::int_select).
    Select(SelectOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [select assign](crate::ops::FloatTensorOps::float_select_assign).
    /// Int => [select assign](crate::ops::IntTensorOps::int_select_assign).
    SelectAssign(SelectAssignOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [mask where](crate::ops::FloatTensorOps::float_mask_where).
    /// Int => [mask where](crate::ops::IntTensorOps::int_mask_where).
    MaskWhere(MaskWhereOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [mask fill](crate::ops::FloatTensorOps::float_mask_fill).
    /// Int => [mask fill](crate::ops::IntTensorOps::int_mask_fill).
    MaskFill(MaskFillOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [mean dim](crate::ops::FloatTensorOps::float_mean_dim).
    /// Int => [mean dim](crate::ops::IntTensorOps::int_mean_dim).
    MeanDim(ScalarOpRepr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [mean](crate::ops::FloatTensorOps::float_mean).
    /// Int => [mean](crate::ops::IntTensorOps::int_mean).
    Mean(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [sum](crate::ops::FloatTensorOps::float_sum).
    /// Int => [sum](crate::ops::IntTensorOps::int_sum).
    Sum(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [sum dim](crate::ops::FloatTensorOps::float_sum_dim).
    /// Int => [sum dim](crate::ops::IntTensorOps::int_sum_dim).
    SumDim(ScalarOpRepr<usize>),

    /// Operation corresponding to:
    ///
    /// Float => [prod](crate::ops::FloatTensorOps::float_prod).
    /// Int => [prod](crate::ops::IntTensorOps::int_prod).
    Prod(UnaryOpRepr),

    /// Operation corresponding to:
    ///
    /// Float => [prod dim](crate::ops::FloatTensorOps::float_prod_dim).
    /// Int => [prod dim](crate::ops::IntTensorOps::int_prod_dim).
    ProdDim(ScalarOpRepr<usize>),

    /// Operation corresponding to:
    ///
    /// Float => [equal elem](crate::ops::FloatTensorOps::float_equal_elem).
    /// Int => [equal elem](crate::ops::IntTensorOps::int_equal_elem).
    EqualElem(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater](crate::ops::FloatTensorOps::float_greater).
    /// Int => [greater](crate::ops::IntTensorOps::int_greater).
    Greater(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [greater elem](crate::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](crate::ops::IntTensorOps::int_greater_elem).
    GreaterElem(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal](crate::ops::FloatTensorOps::float_greater_elem).
    /// Int => [greater elem](crate::ops::IntTensorOps::int_greater_elem).
    GreaterEqual(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [greater equal elem](crate::ops::FloatTensorOps::float_greater_equal_elem).
    /// Int => [greater equal elem](crate::ops::IntTensorOps::int_greater_equal_elem).
    GreaterEqualElem(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower](crate::ops::FloatTensorOps::float_lower).
    /// Int => [lower](crate::ops::IntTensorOps::int_lower).
    Lower(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [lower elem](crate::ops::FloatTensorOps::float_lower_elem).
    /// Int => [lower elem](crate::ops::IntTensorOps::int_lower_elem).
    LowerElem(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal](crate::ops::FloatTensorOps::float_lower_equal).
    /// Int => [lower equal](crate::ops::IntTensorOps::int_lower_equal).
    LowerEqual(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [lower equal elem](crate::ops::FloatTensorOps::float_lower_equal_elem).
    /// Int => [lower equal elem](crate::ops::IntTensorOps::int_lower_equal_elem).
    LowerEqualElem(ScalarOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Float => [argmax](crate::ops::FloatTensorOps::float_argmax).
    /// Int => [argmax](crate::ops::IntTensorOps::int_argmax).
    ArgMax(ScalarOpRepr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [argmin](crate::ops::FloatTensorOps::float_argmin).
    /// Int => [argmin](crate::ops::IntTensorOps::int_argmin).
    ArgMin(ScalarOpRepr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [max](crate::ops::FloatTensorOps::float_max).
    /// Int => [max](crate::ops::IntTensorOps::int_max).
    Max(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [max dim with indices](crate::ops::FloatTensorOps::float_max_dim_with_indices).
    /// Int => [max dim with indices](crate::ops::IntTensorOps::int_max_dim_with_indices).
    MaxDimWithIndices(ReduceDimWithIndicesOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [min dim with indices](crate::ops::FloatTensorOps::float_min_dim_with_indices).
    /// Int => [min dim with indices](crate::ops::IntTensorOps::int_min_dim_with_indices).
    MinDimWithIndices(ReduceDimWithIndicesOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [min](crate::ops::FloatTensorOps::float_min).
    /// Int => [min](crate::ops::IntTensorOps::int_min).
    Min(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [max dim](crate::ops::FloatTensorOps::float_max_dim).
    /// Int => [max dim](crate::ops::IntTensorOps::int_max_dim).
    MaxDim(ScalarOpRepr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [min dim](crate::ops::FloatTensorOps::float_min_dim).
    /// Int => [min dim](crate::ops::IntTensorOps::int_min_dim).
    MinDim(ScalarOpRepr<usize>),
    /// Operation corresponding to:
    ///
    /// Float => [clamp](crate::ops::FloatTensorOps::float_clamp).
    /// Int => [clamp](crate::ops::IntTensorOps::int_clamp).
    Clamp(ClampOpRepr<E>),
    /// Operation corresponding to:
    ///
    /// Int => [random](crate::ops::IntTensorOps::int_random).
    IntRandom(RandomOpRepr),
    /// Operation corresponding to:
    ///
    /// Float => [powf](crate::ops::FloatTensorOps::float_powf).
    /// Int => [powf](crate::ops::IntTensorOps::int_powf).
    Powf(BinaryOpRepr),
}

/// Operation intermediate representation specific to an int tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum IntOperationRepr {
    /// Operation corresponding to [into float](crate::ops::IntTensorOps::int_into_float).
    IntoFloat(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise and](crate::ops::IntTensorOps::bitwise_and).
    BitwiseAnd(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise and scalar](crate::ops::IntTensorOps::bitwise_and_scalar).
    BitwiseAndScalar(ScalarOpRepr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise or](crate::ops::IntTensorOps::bitwise_or).
    BitwiseOr(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise or scalar](crate::ops::IntTensorOps::bitwise_or_scalar).
    BitwiseOrScalar(ScalarOpRepr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise xor](crate::ops::IntTensorOps::bitwise_xor).
    BitwiseXor(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise xor scalar](crate::ops::IntTensorOps::bitwise_xor_scalar).
    BitwiseXorScalar(ScalarOpRepr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise not](crate::ops::IntTensorOps::bitwise_not).
    BitwiseNot(UnaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise left shift](crate::ops::IntTensorOps::bitwise_left_shift).
    BitwiseLeftShift(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise left shift scalar](crate::ops::IntTensorOps::bitwise_left_shift_scalar).
    BitwiseLeftShiftScalar(ScalarOpRepr<i32>),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise right shift](crate::ops::IntTensorOps::bitwise_right_shift).
    BitwiseRightShift(BinaryOpRepr),
    /// Operation corresponding to:
    ///
    /// Int => [bitwise right shift scalar](crate::ops::IntTensorOps::bitwise_right_shift_scalar).
    BitwiseRightShiftScalar(ScalarOpRepr<i32>),
}

/// Operation intermediate representation specific to a bool tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BoolOperationRepr {
    /// Operation corresponding to [into float](crate::ops::BoolTensorOps::bool_into_float).
    IntoFloat(UnaryOpRepr),
    /// Operation corresponding to [into int](crate::ops::BoolTensorOps::bool_into_int).
    IntoInt(UnaryOpRepr),
    /// Operation corresponding to [not](crate::ops::BoolTensorOps::bool_not).
    Not(UnaryOpRepr),
}

/// Swap dim operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct SwapDimsOpRepr {
    /// Input tensor intermediate representation.
    pub input: TensorRepr,
    /// Output tensor intermediate representation.
    pub out: TensorRepr,
    /// The first dim to swap.
    pub dim1: usize,
    /// The second dim to swap.
    pub dim2: usize,
}

/// Permute operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct PermuteOpRepr {
    /// Input tensor intermediate representation.
    pub input: TensorRepr,
    /// Output tensor intermediate representation.
    pub out: TensorRepr,
    /// The new order of the dimensions.
    pub axes: Vec<usize>,
}

/// Expand operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExpandOpRepr {
    /// Input tensor intermediate representation.
    pub input: TensorRepr,
    /// Output tensor intermediate representation.
    pub out: TensorRepr,
    /// The new shape.
    pub shape: Vec<usize>,
}

/// Flip operation intermediate representation.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct FlipOpRepr {
    /// Input tensor intermediate representation.
    pub input: TensorRepr,
    /// Output tensor intermediate representation.
    pub out: TensorRepr,
    /// The dimensions to flip.
    pub axes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RandomOpRepr {
    pub out: TensorRepr,
    pub distribution: Distribution,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
/// Declares a tensor has been initialized.
///
/// It is necessary to register for proper orphan detection and avoid memory leak.
pub struct InitOperationRepr {
    /// The initialized tensor.
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct BinaryOpRepr {
    pub lhs: TensorRepr,
    pub rhs: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct UnaryOpRepr {
    pub input: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScalarOpRepr<E> {
    pub lhs: TensorRepr,
    pub rhs: E,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct GatherOpRepr {
    pub tensor: TensorRepr,
    pub dim: usize,
    pub indices: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ScatterOpRepr {
    pub tensor: TensorRepr,
    pub dim: usize,
    pub indices: TensorRepr,
    pub value: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectOpRepr {
    pub tensor: TensorRepr,
    pub dim: usize,
    pub indices: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SelectAssignOpRepr {
    pub tensor: TensorRepr,
    pub dim: usize,
    pub indices: TensorRepr,
    pub value: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceOpRepr {
    pub tensor: TensorRepr,
    pub ranges: Vec<Range<usize>>,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct SliceAssignOpRepr {
    pub tensor: TensorRepr,
    pub ranges: Vec<Range<usize>>,
    pub value: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskWhereOpRepr {
    pub tensor: TensorRepr,
    pub mask: TensorRepr,
    pub value: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaskFillOpRepr<E> {
    pub tensor: TensorRepr,
    pub mask: TensorRepr,
    pub value: E,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ClampOpRepr<E> {
    pub tensor: TensorRepr,
    pub min: E,
    pub max: E,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct RepeatDimOpRepr {
    pub tensor: TensorRepr,
    pub dim: usize,
    pub times: usize,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct CatOpRepr {
    pub tensors: Vec<TensorRepr>,
    pub dim: usize,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ReduceDimWithIndicesOpRepr {
    pub tensor: TensorRepr,
    pub dim: usize,
    pub out: TensorRepr,
    pub out_indices: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingOpRepr {
    pub weights: TensorRepr,
    pub indices: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct EmbeddingBackwardOpRepr {
    pub weights: TensorRepr,
    pub out_grad: TensorRepr,
    pub indices: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOpRepr {
    pub x: TensorRepr,
    pub weight: TensorRepr,
    pub bias: Option<TensorRepr>,
    pub options: Conv1dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOpRepr {
    pub x: TensorRepr,
    pub weight: TensorRepr,
    pub bias: Option<TensorRepr>,
    pub options: Conv2dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformConv2dOpRepr {
    pub x: TensorRepr,
    pub offset: TensorRepr,
    pub weight: TensorRepr,
    pub mask: Option<TensorRepr>,
    pub bias: Option<TensorRepr>,
    pub options: DeformableConv2dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformConv2dBackwardOpRepr {
    pub x: TensorRepr,
    pub offset: TensorRepr,
    pub weight: TensorRepr,
    pub mask: Option<TensorRepr>,
    pub bias: Option<TensorRepr>,
    pub out_grad: TensorRepr,
    pub options: DeformableConv2dOptionsRepr,
    pub input_grad: TensorRepr,
    pub offset_grad: TensorRepr,
    pub weight_grad: TensorRepr,
    pub mask_grad: Option<TensorRepr>,
    pub bias_grad: Option<TensorRepr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dOpRepr {
    pub x: TensorRepr,
    pub weight: TensorRepr,
    pub bias: Option<TensorRepr>,
    pub options: Conv3dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOpRepr {
    pub x: TensorRepr,
    pub weight: TensorRepr,
    pub bias: Option<TensorRepr>,
    pub options: ConvTranspose1dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOpRepr {
    pub x: TensorRepr,
    pub weight: TensorRepr,
    pub bias: Option<TensorRepr>,
    pub options: ConvTranspose2dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose3dOpRepr {
    pub x: TensorRepr,
    pub weight: TensorRepr,
    pub bias: Option<TensorRepr>,
    pub options: ConvTranspose3dOptionsRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv1dOptionsRepr {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv2dOptionsRepr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DeformableConv2dOptionsRepr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub weight_groups: usize,
    pub offset_groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct Conv3dOptionsRepr {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose1dOptionsRepr {
    pub stride: [usize; 1],
    pub padding: [usize; 1],
    pub padding_out: [usize; 1],
    pub dilation: [usize; 1],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose2dOptionsRepr {
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub padding_out: [usize; 2],
    pub dilation: [usize; 2],
    pub groups: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct ConvTranspose3dOptionsRepr {
    pub stride: [usize; 3],
    pub padding: [usize; 3],
    pub padding_out: [usize; 3],
    pub dilation: [usize; 3],
    pub groups: usize,
}

/// Quantization parameters intermediate representation.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizationParametersRepr {
    /// The scaling factor.
    pub scale: TensorRepr,
    /// The zero-point offset.
    pub offset: Option<TensorRepr>,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct QuantizeOpRepr {
    pub tensor: TensorRepr,
    pub qparams: QuantizationParametersRepr,
    pub scheme: QuantizationScheme,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct DequantizeOpRepr {
    pub input: TensorRepr,
    pub out: TensorRepr,
}

impl From<ConvOptions<1>> for Conv1dOptionsRepr {
    fn from(value: ConvOptions<1>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<2>> for Conv2dOptionsRepr {
    fn from(value: ConvOptions<2>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<ConvOptions<3>> for Conv3dOptionsRepr {
    fn from(value: ConvOptions<3>) -> Self {
        Self {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            groups: value.groups,
        }
    }
}

impl From<DeformConvOptions<2>> for DeformableConv2dOptionsRepr {
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

impl From<ConvTransposeOptions<1>> for ConvTranspose1dOptionsRepr {
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

impl From<ConvTransposeOptions<2>> for ConvTranspose2dOptionsRepr {
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

impl From<ConvTransposeOptions<3>> for ConvTranspose3dOptionsRepr {
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

impl From<Conv1dOptionsRepr> for ConvOptions<1> {
    fn from(val: Conv1dOptionsRepr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv2dOptionsRepr> for ConvOptions<2> {
    fn from(val: Conv2dOptionsRepr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<Conv3dOptionsRepr> for ConvOptions<3> {
    fn from(val: Conv3dOptionsRepr) -> Self {
        ConvOptions {
            stride: val.stride,
            padding: val.padding,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<DeformableConv2dOptionsRepr> for DeformConvOptions<2> {
    fn from(value: DeformableConv2dOptionsRepr) -> Self {
        DeformConvOptions {
            stride: value.stride,
            padding: value.padding,
            dilation: value.dilation,
            weight_groups: value.weight_groups,
            offset_groups: value.offset_groups,
        }
    }
}

impl From<ConvTranspose1dOptionsRepr> for ConvTransposeOptions<1> {
    fn from(val: ConvTranspose1dOptionsRepr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose2dOptionsRepr> for ConvTransposeOptions<2> {
    fn from(val: ConvTranspose2dOptionsRepr) -> Self {
        ConvTransposeOptions {
            stride: val.stride,
            padding: val.padding,
            padding_out: val.padding_out,
            dilation: val.dilation,
            groups: val.groups,
        }
    }
}

impl From<ConvTranspose3dOptionsRepr> for ConvTransposeOptions<3> {
    fn from(val: ConvTranspose3dOptionsRepr) -> Self {
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
pub struct AvgPool1dOpRepr {
    pub x: TensorRepr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dOpRepr {
    pub x: TensorRepr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool1dBackwardOpRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub count_include_pad: bool,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AvgPool2dBackwardOpRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub count_include_pad: bool,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dOpRepr {
    pub x: TensorRepr,
    pub output_size: usize,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dOpRepr {
    pub x: TensorRepr,
    pub output_size: [usize; 2],
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool1dBackwardOpRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct AdaptiveAvgPool2dBackwardOpRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dOpRepr {
    pub x: TensorRepr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesOpRepr {
    pub x: TensorRepr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorRepr,
    pub out_indices: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool1dWithIndicesBackwardOpRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub indices: TensorRepr,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub dilation: usize,
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool2dOpRepr {
    pub x: TensorRepr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorRepr,
}

#[allow(missing_docs)]
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct MaxPool2dWithIndicesOpRepr {
    pub x: TensorRepr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorRepr,
    pub out_indices: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct MaxPool2dWithIndicesBackwardOpRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub indices: TensorRepr,
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
    pub dilation: [usize; 2],
    pub out: TensorRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub enum InterpolateModeRepr {
    Nearest,
    Bilinear,
    Bicubic,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOptionsRepr {
    pub mode: InterpolateModeRepr,
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateOpRepr {
    pub x: TensorRepr,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsRepr,
    pub out: TensorRepr,
}

impl From<InterpolateModeRepr> for InterpolateMode {
    fn from(val: InterpolateModeRepr) -> Self {
        match val {
            InterpolateModeRepr::Nearest => Self::Nearest,
            InterpolateModeRepr::Bilinear => Self::Bilinear,
            InterpolateModeRepr::Bicubic => Self::Bicubic,
        }
    }
}

impl From<InterpolateOptionsRepr> for InterpolateOptions {
    fn from(val: InterpolateOptionsRepr) -> Self {
        Self {
            mode: val.mode.into(),
        }
    }
}

impl From<InterpolateMode> for InterpolateModeRepr {
    fn from(val: InterpolateMode) -> Self {
        match val {
            InterpolateMode::Nearest => Self::Nearest,
            InterpolateMode::Bilinear => Self::Bilinear,
            InterpolateMode::Bicubic => Self::Bicubic,
        }
    }
}

impl From<InterpolateOptions> for InterpolateOptionsRepr {
    fn from(val: InterpolateOptions) -> Self {
        Self {
            mode: val.mode.into(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct InterpolateBackwardRepr {
    pub x: TensorRepr,
    pub grad: TensorRepr,
    pub output_size: [usize; 2],
    pub options: InterpolateOptionsRepr,
    pub out: TensorRepr,
}

impl OperationRepr {
    /// Cleanup the remaining tensor handles that have not been used.
    pub fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            OperationRepr::BaseFloat(repr) => repr.nodes(),
            OperationRepr::BaseInt(repr) => repr.nodes(),
            OperationRepr::BaseBool(repr) => repr.nodes(),
            OperationRepr::NumericFloat(_dtype, repr) => repr.nodes(),
            OperationRepr::NumericInt(_dtype, repr) => repr.nodes(),
            OperationRepr::Bool(repr) => repr.nodes(),
            OperationRepr::Int(repr) => repr.nodes(),
            OperationRepr::Float(_dtype, repr) => repr.nodes(),
            OperationRepr::Module(repr) => repr.nodes(),
            OperationRepr::Init(repr) => repr.nodes(),
            OperationRepr::Custom(repr) => repr.nodes(),
        }
    }
}

impl BaseOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            BaseOperationRepr::ToDevice(repr) => vec![repr],
            BaseOperationRepr::Reshape(repr) => {
                vec![&repr.input, &repr.out]
            }
            BaseOperationRepr::SwapDims(repr) => {
                vec![&repr.input, &repr.out]
            }
            BaseOperationRepr::Permute(repr) => {
                vec![&repr.input, &repr.out]
            }

            BaseOperationRepr::Expand(repr) => {
                vec![&repr.input, &repr.out]
            }

            BaseOperationRepr::Flip(repr) => {
                vec![&repr.input, &repr.out]
            }
            BaseOperationRepr::Slice(repr) => {
                vec![&repr.tensor, &repr.out]
            }
            BaseOperationRepr::SliceAssign(repr) => {
                vec![&repr.tensor, &repr.value, &repr.out]
            }
            BaseOperationRepr::Equal(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            BaseOperationRepr::RepeatDim(repr) => {
                vec![&repr.tensor, &repr.out]
            }
            BaseOperationRepr::Cat(repr) => repr.tensors.iter().collect(),
            BaseOperationRepr::Cast(repr) => vec![&repr.input, &repr.out],
            BaseOperationRepr::Empty(repr) => vec![repr],
        }
    }
}

impl<E: Element> NumericOperationRepr<E> {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            NumericOperationRepr::Add(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::AddScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Sub(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::SubScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Mul(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::MulScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Div(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::DivScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Rem(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::RemScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Ones(repr) => vec![repr],
            NumericOperationRepr::Gather(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.out]
            }
            NumericOperationRepr::Scatter(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.value, &repr.out]
            }
            NumericOperationRepr::Select(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.out]
            }
            NumericOperationRepr::SelectAssign(repr) => {
                vec![&repr.tensor, &repr.indices, &repr.value, &repr.out]
            }
            NumericOperationRepr::MaskWhere(repr) => {
                vec![&repr.tensor, &repr.mask, &repr.value, &repr.out]
            }
            NumericOperationRepr::MaskFill(repr) => {
                vec![&repr.tensor, &repr.mask, &repr.out]
            }
            NumericOperationRepr::EqualElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::GreaterElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::GreaterEqualElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::LowerElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::LowerEqualElem(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Greater(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::GreaterEqual(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::Lower(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::LowerEqual(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            NumericOperationRepr::ArgMax(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::ArgMin(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Clamp(repr) => {
                vec![&repr.tensor, &repr.out]
            }
            NumericOperationRepr::Abs(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationRepr::Zeros(repr) => vec![repr],
            NumericOperationRepr::Full(repr) => vec![&repr.0],
            NumericOperationRepr::MeanDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Mean(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationRepr::Sum(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationRepr::SumDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Prod(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationRepr::ProdDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::Max(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationRepr::MaxDimWithIndices(repr) => {
                vec![&repr.tensor, &repr.out_indices, &repr.out]
            }
            NumericOperationRepr::MinDimWithIndices(repr) => {
                vec![&repr.tensor, &repr.out_indices, &repr.out]
            }
            NumericOperationRepr::Min(repr) => {
                vec![&repr.input, &repr.out]
            }
            NumericOperationRepr::MaxDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::MinDim(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            NumericOperationRepr::IntRandom(repr) => {
                vec![&repr.out]
            }
            NumericOperationRepr::Powf(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
        }
    }
}

impl FloatOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            FloatOperationRepr::Matmul(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            FloatOperationRepr::Random(repr) => vec![&repr.out],
            FloatOperationRepr::Exp(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Log(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Log1p(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Erf(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Recip(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::PowfScalar(repr) => vec![&repr.lhs, &repr.out],
            FloatOperationRepr::Sqrt(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Cos(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Sin(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Tanh(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Round(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Floor(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Ceil(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::IntoInt(repr) => vec![&repr.input, &repr.out],
            FloatOperationRepr::Quantize(repr) => {
                if let Some(offset) = &repr.qparams.offset {
                    vec![&repr.tensor, &repr.qparams.scale, &offset, &repr.out]
                } else {
                    vec![&repr.tensor, &repr.qparams.scale, &repr.out]
                }
            }
            FloatOperationRepr::Dequantize(repr) => vec![&repr.input, &repr.out],
        }
    }
}

impl IntOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            IntOperationRepr::IntoFloat(repr) => vec![&repr.input, &repr.out],
            IntOperationRepr::BitwiseAnd(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationRepr::BitwiseAndScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationRepr::BitwiseOr(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationRepr::BitwiseOrScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationRepr::BitwiseXor(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationRepr::BitwiseXorScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationRepr::BitwiseNot(repr) => {
                vec![&repr.input, &repr.out]
            }
            IntOperationRepr::BitwiseLeftShift(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationRepr::BitwiseLeftShiftScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
            IntOperationRepr::BitwiseRightShift(repr) => {
                vec![&repr.lhs, &repr.rhs, &repr.out]
            }
            IntOperationRepr::BitwiseRightShiftScalar(repr) => {
                vec![&repr.lhs, &repr.out]
            }
        }
    }
}

impl BoolOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            BoolOperationRepr::IntoFloat(repr) => vec![&repr.input, &repr.out],
            BoolOperationRepr::IntoInt(repr) => vec![&repr.input, &repr.out],
            BoolOperationRepr::Not(repr) => vec![&repr.input, &repr.out],
        }
    }
}

impl ModuleOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            ModuleOperationRepr::Embedding(repr) => {
                vec![&repr.weights, &repr.indices, &repr.out]
            }
            ModuleOperationRepr::EmbeddingBackward(repr) => {
                vec![&repr.weights, &repr.out_grad, &repr.indices, &repr.out]
            }
            ModuleOperationRepr::Conv1d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationRepr::Conv2d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationRepr::Conv3d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationRepr::DeformableConv2d(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => vec![&repr.x, &repr.offset, &repr.weight, &mask, &bias],
                (Some(mask), None) => vec![&repr.x, &repr.offset, &repr.weight, &mask],
                (None, Some(bias)) => vec![&repr.x, &repr.offset, &repr.weight, &bias],
                (None, None) => vec![&repr.x, &repr.offset, &repr.weight],
            },
            ModuleOperationRepr::DeformableConv2dBackward(repr) => match (&repr.mask, &repr.bias) {
                (Some(mask), Some(bias)) => {
                    vec![&repr.x, &repr.offset, &repr.weight, &mask, &bias]
                }
                (Some(mask), None) => vec![&repr.x, &repr.offset, &repr.weight, &mask],
                (None, Some(bias)) => vec![&repr.x, &repr.offset, &repr.weight, &bias],
                (None, None) => vec![&repr.x, &repr.offset, &repr.weight],
            },
            ModuleOperationRepr::ConvTranspose1d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationRepr::ConvTranspose2d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationRepr::ConvTranspose3d(repr) => {
                if let Some(bias) = &repr.bias {
                    vec![&repr.x, &repr.weight, &bias, &repr.out]
                } else {
                    vec![&repr.x, &repr.weight, &repr.out]
                }
            }
            ModuleOperationRepr::AvgPool1d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::AvgPool2d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::AvgPool1dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationRepr::AvgPool2dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationRepr::AdaptiveAvgPool1d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::AdaptiveAvgPool2d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::AdaptiveAvgPool1dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationRepr::AdaptiveAvgPool2dBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
            ModuleOperationRepr::MaxPool1d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::MaxPool1dWithIndices(repr) => {
                vec![&repr.x, &repr.out, &repr.out_indices]
            }
            ModuleOperationRepr::MaxPool1dWithIndicesBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.indices, &repr.grad]
            }
            ModuleOperationRepr::MaxPool2d(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::MaxPool2dWithIndices(repr) => {
                vec![&repr.x, &repr.out, &repr.out_indices]
            }
            ModuleOperationRepr::MaxPool2dWithIndicesBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.indices, &repr.grad]
            }
            ModuleOperationRepr::Interpolate(repr) => {
                vec![&repr.x, &repr.out]
            }
            ModuleOperationRepr::InterpolateBackward(repr) => {
                vec![&repr.x, &repr.out, &repr.grad]
            }
        }
    }
}

impl core::hash::Hash for InitOperationRepr {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.out.hash(state);
    }
}

impl InitOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        vec![&self.out]
    }
}

impl core::hash::Hash for RandomOpRepr {
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

impl<E> core::hash::Hash for ScalarOpRepr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.lhs.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for MaskFillOpRepr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.mask.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for ClampOpRepr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.tensor.hash(state);
        self.out.hash(state);
    }
}

impl<E> core::hash::Hash for NumericOperationRepr<E> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        match self {
            NumericOperationRepr::Add(repr) => repr.hash(state),
            NumericOperationRepr::AddScalar(repr) => repr.hash(state),
            NumericOperationRepr::Sub(repr) => repr.hash(state),
            NumericOperationRepr::SubScalar(repr) => repr.hash(state),
            NumericOperationRepr::Div(repr) => repr.hash(state),
            NumericOperationRepr::DivScalar(repr) => repr.hash(state),
            NumericOperationRepr::Rem(repr) => repr.hash(state),
            NumericOperationRepr::RemScalar(repr) => repr.hash(state),
            NumericOperationRepr::Mul(repr) => repr.hash(state),
            NumericOperationRepr::MulScalar(repr) => repr.hash(state),
            NumericOperationRepr::Abs(repr) => repr.hash(state),
            NumericOperationRepr::Ones(repr) => repr.hash(state),
            NumericOperationRepr::Zeros(repr) => repr.hash(state),
            NumericOperationRepr::Full(repr) => repr.0.hash(state),
            NumericOperationRepr::Gather(repr) => repr.hash(state),
            NumericOperationRepr::Scatter(repr) => repr.hash(state),
            NumericOperationRepr::Select(repr) => repr.hash(state),
            NumericOperationRepr::SelectAssign(repr) => repr.hash(state),
            NumericOperationRepr::MaskWhere(repr) => repr.hash(state),
            NumericOperationRepr::MaskFill(repr) => repr.hash(state),
            NumericOperationRepr::MeanDim(repr) => repr.hash(state),
            NumericOperationRepr::Mean(repr) => repr.hash(state),
            NumericOperationRepr::Sum(repr) => repr.hash(state),
            NumericOperationRepr::SumDim(repr) => repr.hash(state),
            NumericOperationRepr::Prod(repr) => repr.hash(state),
            NumericOperationRepr::ProdDim(repr) => repr.hash(state),
            NumericOperationRepr::EqualElem(repr) => repr.hash(state),
            NumericOperationRepr::Greater(repr) => repr.hash(state),
            NumericOperationRepr::GreaterElem(repr) => repr.hash(state),
            NumericOperationRepr::GreaterEqual(repr) => repr.hash(state),
            NumericOperationRepr::GreaterEqualElem(repr) => repr.hash(state),
            NumericOperationRepr::Lower(repr) => repr.hash(state),
            NumericOperationRepr::LowerElem(repr) => repr.hash(state),
            NumericOperationRepr::LowerEqual(repr) => repr.hash(state),
            NumericOperationRepr::LowerEqualElem(repr) => repr.hash(state),
            NumericOperationRepr::ArgMax(repr) => repr.hash(state),
            NumericOperationRepr::ArgMin(repr) => repr.hash(state),
            NumericOperationRepr::Max(repr) => repr.hash(state),
            NumericOperationRepr::MaxDimWithIndices(repr) => repr.hash(state),
            NumericOperationRepr::MinDimWithIndices(repr) => repr.hash(state),
            NumericOperationRepr::Min(repr) => repr.hash(state),
            NumericOperationRepr::MaxDim(repr) => repr.hash(state),
            NumericOperationRepr::MinDim(repr) => repr.hash(state),
            NumericOperationRepr::Clamp(repr) => repr.hash(state),
            NumericOperationRepr::IntRandom(repr) => repr.hash(state),
            NumericOperationRepr::Powf(repr) => repr.hash(state),
        }
    }
}
