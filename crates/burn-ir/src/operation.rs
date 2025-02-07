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
    /// Create a new custom operation description.
    pub fn new(id: &'static str, inputs: &[TensorRepr], outputs: &[TensorRepr]) -> Self {
        Self {
            id: id.to_owned(),
            inputs: inputs.to_vec(),
            outputs: outputs.to_vec(),
        }
    }

    /// Consume the description, and get the in and output tensors.
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

/// Describe all tensor operations possible.
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

// TODO: ops should all have OpRepr suffix (not OperationRepr, and not Repr only)

/// Operation description specific to a float tensor.
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

/// Operation description specific to module.
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

/// Operation description specific to an int tensor.
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

/// Operation description specific to a bool tensor.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub enum BoolOperationRepr {
    /// Operation corresponding to [into float](crate::ops::BoolTensorOps::bool_into_float).
    IntoFloat(UnaryOpRepr),
    /// Operation corresponding to [into int](crate::ops::BoolTensorOps::bool_into_int).
    IntoInt(UnaryOpRepr),
    /// Operation corresponding to [not](crate::ops::BoolTensorOps::bool_not).
    Not(UnaryOpRepr),
}

/// Swap dim operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct SwapDimsOpRepr {
    /// Input tensor description.
    pub input: TensorRepr,
    /// Output tensor description.
    pub out: TensorRepr,
    /// The first dim to swap.
    pub dim1: usize,
    /// The second dim to swap.
    pub dim2: usize,
}

/// Permute operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct PermuteOpRepr {
    /// Input tensor description.
    pub input: TensorRepr,
    /// Output tensor description.
    pub out: TensorRepr,
    /// The new order of the dimensions.
    pub axes: Vec<usize>,
}

/// Expand operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct ExpandOpRepr {
    /// Input tensor description.
    pub input: TensorRepr,
    /// Output tensor description.
    pub out: TensorRepr,
    /// The new shape.
    pub shape: Vec<usize>,
}

/// Flip operation description.
#[derive(Clone, Debug, Hash, PartialEq, Serialize, Deserialize)]
pub struct FlipOpRepr {
    /// Input tensor description.
    pub input: TensorRepr,
    /// Output tensor description.
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

/// Quantization parameters description.
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
            OperationRepr::BaseFloat(ops) => ops.nodes(),
            OperationRepr::BaseInt(ops) => ops.nodes(),
            OperationRepr::BaseBool(ops) => ops.nodes(),
            OperationRepr::NumericFloat(_dtype, ops) => ops.nodes(),
            OperationRepr::NumericInt(_dtype, ops) => ops.nodes(),
            OperationRepr::Bool(ops) => ops.nodes(),
            OperationRepr::Int(ops) => ops.nodes(),
            OperationRepr::Float(_dtype, ops) => ops.nodes(),
            OperationRepr::Module(ops) => ops.nodes(),
            OperationRepr::Init(ops) => ops.nodes(),
            OperationRepr::Custom(ops) => ops.nodes(),
        }
    }
}

impl BaseOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            BaseOperationRepr::ToDevice(desc) => vec![desc],
            BaseOperationRepr::Reshape(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOperationRepr::SwapDims(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOperationRepr::Permute(desc) => {
                vec![&desc.input, &desc.out]
            }

            BaseOperationRepr::Expand(desc) => {
                vec![&desc.input, &desc.out]
            }

            BaseOperationRepr::Flip(desc) => {
                vec![&desc.input, &desc.out]
            }
            BaseOperationRepr::Slice(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            BaseOperationRepr::SliceAssign(desc) => {
                vec![&desc.tensor, &desc.value, &desc.out]
            }
            BaseOperationRepr::Equal(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            BaseOperationRepr::RepeatDim(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            BaseOperationRepr::Cat(desc) => desc.tensors.iter().collect(),
            BaseOperationRepr::Cast(desc) => vec![&desc.input, &desc.out],
            BaseOperationRepr::Empty(desc) => vec![desc],
        }
    }
}

impl<E: Element> NumericOperationRepr<E> {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            NumericOperationRepr::Add(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::AddScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Sub(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::SubScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Mul(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::MulScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Div(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::DivScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Rem(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::RemScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Ones(desc) => vec![desc],
            NumericOperationRepr::Gather(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.out]
            }
            NumericOperationRepr::Scatter(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.value, &desc.out]
            }
            NumericOperationRepr::Select(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.out]
            }
            NumericOperationRepr::SelectAssign(desc) => {
                vec![&desc.tensor, &desc.indices, &desc.value, &desc.out]
            }
            NumericOperationRepr::MaskWhere(desc) => {
                vec![&desc.tensor, &desc.mask, &desc.value, &desc.out]
            }
            NumericOperationRepr::MaskFill(desc) => {
                vec![&desc.tensor, &desc.mask, &desc.out]
            }
            NumericOperationRepr::EqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::GreaterElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::GreaterEqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::LowerElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::LowerEqualElem(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Greater(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::GreaterEqual(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::Lower(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::LowerEqual(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            NumericOperationRepr::ArgMax(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::ArgMin(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Clamp(desc) => {
                vec![&desc.tensor, &desc.out]
            }
            NumericOperationRepr::Abs(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationRepr::Zeros(desc) => vec![desc],
            NumericOperationRepr::Full(desc) => vec![&desc.0],
            NumericOperationRepr::MeanDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Mean(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationRepr::Sum(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationRepr::SumDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Prod(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationRepr::ProdDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::Max(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationRepr::MaxDimWithIndices(desc) => {
                vec![&desc.tensor, &desc.out_indices, &desc.out]
            }
            NumericOperationRepr::MinDimWithIndices(desc) => {
                vec![&desc.tensor, &desc.out_indices, &desc.out]
            }
            NumericOperationRepr::Min(desc) => {
                vec![&desc.input, &desc.out]
            }
            NumericOperationRepr::MaxDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::MinDim(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            NumericOperationRepr::IntRandom(desc) => {
                vec![&desc.out]
            }
            NumericOperationRepr::Powf(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
        }
    }
}

impl FloatOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            FloatOperationRepr::Matmul(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            FloatOperationRepr::Random(desc) => vec![&desc.out],
            FloatOperationRepr::Exp(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Log(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Log1p(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Erf(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Recip(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::PowfScalar(desc) => vec![&desc.lhs, &desc.out],
            FloatOperationRepr::Sqrt(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Cos(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Sin(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Tanh(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Round(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Floor(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Ceil(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::IntoInt(desc) => vec![&desc.input, &desc.out],
            FloatOperationRepr::Quantize(desc) => {
                if let Some(offset) = &desc.qparams.offset {
                    vec![&desc.tensor, &desc.qparams.scale, &offset, &desc.out]
                } else {
                    vec![&desc.tensor, &desc.qparams.scale, &desc.out]
                }
            }
            FloatOperationRepr::Dequantize(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl IntOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            IntOperationRepr::IntoFloat(desc) => vec![&desc.input, &desc.out],
            IntOperationRepr::BitwiseAnd(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            IntOperationRepr::BitwiseAndScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            IntOperationRepr::BitwiseOr(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            IntOperationRepr::BitwiseOrScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            IntOperationRepr::BitwiseXor(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            IntOperationRepr::BitwiseXorScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            IntOperationRepr::BitwiseNot(desc) => {
                vec![&desc.input, &desc.out]
            }
            IntOperationRepr::BitwiseLeftShift(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            IntOperationRepr::BitwiseLeftShiftScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
            IntOperationRepr::BitwiseRightShift(desc) => {
                vec![&desc.lhs, &desc.rhs, &desc.out]
            }
            IntOperationRepr::BitwiseRightShiftScalar(desc) => {
                vec![&desc.lhs, &desc.out]
            }
        }
    }
}

impl BoolOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            BoolOperationRepr::IntoFloat(desc) => vec![&desc.input, &desc.out],
            BoolOperationRepr::IntoInt(desc) => vec![&desc.input, &desc.out],
            BoolOperationRepr::Not(desc) => vec![&desc.input, &desc.out],
        }
    }
}

impl ModuleOperationRepr {
    fn nodes(&self) -> Vec<&TensorRepr> {
        match self {
            ModuleOperationRepr::Embedding(desc) => {
                vec![&desc.weights, &desc.indices, &desc.out]
            }
            ModuleOperationRepr::EmbeddingBackward(desc) => {
                vec![&desc.weights, &desc.out_grad, &desc.indices, &desc.out]
            }
            ModuleOperationRepr::Conv1d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationRepr::Conv2d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationRepr::Conv3d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationRepr::DeformableConv2d(desc) => match (&desc.mask, &desc.bias) {
                (Some(mask), Some(bias)) => vec![&desc.x, &desc.offset, &desc.weight, &mask, &bias],
                (Some(mask), None) => vec![&desc.x, &desc.offset, &desc.weight, &mask],
                (None, Some(bias)) => vec![&desc.x, &desc.offset, &desc.weight, &bias],
                (None, None) => vec![&desc.x, &desc.offset, &desc.weight],
            },
            ModuleOperationRepr::DeformableConv2dBackward(desc) => match (&desc.mask, &desc.bias) {
                (Some(mask), Some(bias)) => {
                    vec![&desc.x, &desc.offset, &desc.weight, &mask, &bias]
                }
                (Some(mask), None) => vec![&desc.x, &desc.offset, &desc.weight, &mask],
                (None, Some(bias)) => vec![&desc.x, &desc.offset, &desc.weight, &bias],
                (None, None) => vec![&desc.x, &desc.offset, &desc.weight],
            },
            ModuleOperationRepr::ConvTranspose1d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationRepr::ConvTranspose2d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationRepr::ConvTranspose3d(desc) => {
                if let Some(bias) = &desc.bias {
                    vec![&desc.x, &desc.weight, &bias, &desc.out]
                } else {
                    vec![&desc.x, &desc.weight, &desc.out]
                }
            }
            ModuleOperationRepr::AvgPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::AvgPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::AvgPool1dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationRepr::AvgPool2dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationRepr::AdaptiveAvgPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::AdaptiveAvgPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::AdaptiveAvgPool1dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationRepr::AdaptiveAvgPool2dBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
            }
            ModuleOperationRepr::MaxPool1d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::MaxPool1dWithIndices(desc) => {
                vec![&desc.x, &desc.out, &desc.out_indices]
            }
            ModuleOperationRepr::MaxPool1dWithIndicesBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.indices, &desc.grad]
            }
            ModuleOperationRepr::MaxPool2d(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::MaxPool2dWithIndices(desc) => {
                vec![&desc.x, &desc.out, &desc.out_indices]
            }
            ModuleOperationRepr::MaxPool2dWithIndicesBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.indices, &desc.grad]
            }
            ModuleOperationRepr::Interpolate(desc) => {
                vec![&desc.x, &desc.out]
            }
            ModuleOperationRepr::InterpolateBackward(desc) => {
                vec![&desc.x, &desc.out, &desc.grad]
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
            NumericOperationRepr::Add(desc) => desc.hash(state),
            NumericOperationRepr::AddScalar(desc) => desc.hash(state),
            NumericOperationRepr::Sub(desc) => desc.hash(state),
            NumericOperationRepr::SubScalar(desc) => desc.hash(state),
            NumericOperationRepr::Div(desc) => desc.hash(state),
            NumericOperationRepr::DivScalar(desc) => desc.hash(state),
            NumericOperationRepr::Rem(desc) => desc.hash(state),
            NumericOperationRepr::RemScalar(desc) => desc.hash(state),
            NumericOperationRepr::Mul(desc) => desc.hash(state),
            NumericOperationRepr::MulScalar(desc) => desc.hash(state),
            NumericOperationRepr::Abs(desc) => desc.hash(state),
            NumericOperationRepr::Ones(desc) => desc.hash(state),
            NumericOperationRepr::Zeros(desc) => desc.hash(state),
            NumericOperationRepr::Full(desc) => desc.0.hash(state),
            NumericOperationRepr::Gather(desc) => desc.hash(state),
            NumericOperationRepr::Scatter(desc) => desc.hash(state),
            NumericOperationRepr::Select(desc) => desc.hash(state),
            NumericOperationRepr::SelectAssign(desc) => desc.hash(state),
            NumericOperationRepr::MaskWhere(desc) => desc.hash(state),
            NumericOperationRepr::MaskFill(desc) => desc.hash(state),
            NumericOperationRepr::MeanDim(desc) => desc.hash(state),
            NumericOperationRepr::Mean(desc) => desc.hash(state),
            NumericOperationRepr::Sum(desc) => desc.hash(state),
            NumericOperationRepr::SumDim(desc) => desc.hash(state),
            NumericOperationRepr::Prod(desc) => desc.hash(state),
            NumericOperationRepr::ProdDim(desc) => desc.hash(state),
            NumericOperationRepr::EqualElem(desc) => desc.hash(state),
            NumericOperationRepr::Greater(desc) => desc.hash(state),
            NumericOperationRepr::GreaterElem(desc) => desc.hash(state),
            NumericOperationRepr::GreaterEqual(desc) => desc.hash(state),
            NumericOperationRepr::GreaterEqualElem(desc) => desc.hash(state),
            NumericOperationRepr::Lower(desc) => desc.hash(state),
            NumericOperationRepr::LowerElem(desc) => desc.hash(state),
            NumericOperationRepr::LowerEqual(desc) => desc.hash(state),
            NumericOperationRepr::LowerEqualElem(desc) => desc.hash(state),
            NumericOperationRepr::ArgMax(desc) => desc.hash(state),
            NumericOperationRepr::ArgMin(desc) => desc.hash(state),
            NumericOperationRepr::Max(desc) => desc.hash(state),
            NumericOperationRepr::MaxDimWithIndices(desc) => desc.hash(state),
            NumericOperationRepr::MinDimWithIndices(desc) => desc.hash(state),
            NumericOperationRepr::Min(desc) => desc.hash(state),
            NumericOperationRepr::MaxDim(desc) => desc.hash(state),
            NumericOperationRepr::MinDim(desc) => desc.hash(state),
            NumericOperationRepr::Clamp(desc) => desc.hash(state),
            NumericOperationRepr::IntRandom(desc) => desc.hash(state),
            NumericOperationRepr::Powf(desc) => desc.hash(state),
        }
    }
}
