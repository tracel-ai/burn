use crate::{
    TensorMetadata,
    backend::{Backend, DeviceOps},
    ops::{
        ActivationOps, BoolTensorOps, FloatTensorOps, IntTensorOps, ModuleOps, QTensorOps,
        TransactionOps,
    },
    quantization::QTensorPrimitive,
};

#[derive(Clone, Debug, Default)]
pub struct DummyBackend;

#[derive(Clone, Default, PartialEq, Debug)]
pub struct DummyDevice;

impl DeviceOps for DummyDevice {
    fn id(&self) -> super::DeviceId {
        todo!()
    }
}

impl TensorMetadata for DummyTensor {
    fn dtype(&self) -> crate::DType {
        todo!()
    }

    fn shape(&self) -> crate::Shape {
        todo!()
    }
}

#[derive(Clone, Debug, Default)]
pub struct DummyTensor;

impl QTensorPrimitive for DummyTensor {
    fn scheme(&self) -> &cubecl_quant::scheme::QuantScheme {
        todo!()
    }
}

impl Backend for DummyBackend {
    type Device = DummyDevice;

    type FloatTensorPrimitive = DummyTensor;

    type FloatElem = f32;

    type IntTensorPrimitive = DummyTensor;

    type IntElem = i32;

    type BoolTensorPrimitive = DummyTensor;

    type BoolElem = bool;

    type QuantizedTensorPrimitive = DummyTensor;

    type QuantizedEncoding = i8;

    fn name(_device: &Self::Device) -> String {
        unimplemented!()
    }

    fn seed(_seed: u64) {
        unimplemented!()
    }
}

impl FloatTensorOps<DummyBackend> for DummyBackend {
    fn float_from_data(
        data: crate::TensorData,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_random(
        shape: crate::Shape,
        distribution: crate::Distribution,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_into_data(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> impl Future<Output = crate::TensorData> + Send {
        async { todo!() }
    }

    fn float_device(tensor: &crate::ops::FloatTensor<DummyBackend>) -> crate::Device<DummyBackend> {
        todo!()
    }

    fn float_to_device(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_into_int(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn float_empty(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_add(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_add_scalar(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_sub(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_sub_scalar(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_mul(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_mul_scalar(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_div(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_div_scalar(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_remainder(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_remainder_scalar(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_matmul(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_recip(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_swap_dims(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim1: usize,
        dim2: usize,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_permute(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_flip(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_reshape(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_gather(
        dim: usize,
        tensor: crate::ops::FloatTensor<DummyBackend>,
        indices: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_scatter(
        dim: usize,
        tensor: crate::ops::FloatTensor<DummyBackend>,
        indices: crate::ops::IntTensor<DummyBackend>,
        value: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_select(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim: usize,
        indices: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_select_assign(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim: usize,
        indices: crate::ops::IntTensor<DummyBackend>,
        value: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_slice(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        ranges: &[core::ops::Range<usize>],
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_slice_assign(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        ranges: &[core::ops::Range<usize>],
        value: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_mask_where(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        mask: crate::ops::BoolTensor<DummyBackend>,
        value: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_mask_fill(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        mask: crate::ops::BoolTensor<DummyBackend>,
        value: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_equal(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_equal_elem(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_greater(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_greater_elem(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_greater_equal(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_greater_equal_elem(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_lower(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_lower_elem(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_lower_equal(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_lower_equal_elem(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn float_sum(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_sum_dim(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_mean_dim(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_cast(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dtype: crate::FloatDType,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_exp(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_log(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_log1p(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_powf(
        lhs: crate::ops::FloatTensor<DummyBackend>,
        rhs: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_powf_scalar(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        value: f32,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_sqrt(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_abs(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_cos(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_sin(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_round(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_floor(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_ceil(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_erf(
        tensor: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn float_argmax(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn float_argmin(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn float_expand(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }
}

impl TransactionOps<DummyBackend> for DummyBackend {}

impl QTensorOps<DummyBackend> for DummyBackend {
    fn q_from_data(
        data: crate::TensorData,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn quantize(
        tensor: crate::ops::FloatTensor<DummyBackend>,
        scheme: &cubecl_quant::scheme::QuantScheme,
        qparams: crate::quantization::QuantizationParametersPrimitive<DummyBackend>,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn dequantize(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn q_device(tensor: &crate::ops::QuantizedTensor<DummyBackend>) -> crate::Device<DummyBackend> {
        todo!()
    }

    fn q_to_device(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_reshape(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_into_data(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
    ) -> impl Future<Output = crate::TensorData> + Send {
        async { todo!() }
    }

    fn q_expand(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_swap_dims(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        dim1: usize,
        dim2: usize,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_permute(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_flip(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_select(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        dim: usize,
        indices: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }

    fn q_slice(
        tensor: crate::ops::QuantizedTensor<DummyBackend>,
        ranges: &[core::ops::Range<usize>],
    ) -> crate::ops::QuantizedTensor<DummyBackend> {
        todo!()
    }
}

impl ActivationOps<DummyBackend> for DummyBackend {}

impl BoolTensorOps<DummyBackend> for DummyBackend {
    fn bool_empty(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_zeros(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_ones(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_into_data(
        tensor: crate::ops::BoolTensor<DummyBackend>,
    ) -> impl Future<Output = crate::TensorData> + Send {
        async { todo!() }
    }

    fn bool_from_data(
        data: crate::TensorData,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_into_int(
        tensor: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bool_into_float(
        tensor: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn bool_device(tensor: &crate::ops::BoolTensor<DummyBackend>) -> crate::Device<DummyBackend> {
        todo!()
    }

    fn bool_to_device(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_reshape(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_slice(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        ranges: &[core::ops::Range<usize>],
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_slice_assign(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        ranges: &[core::ops::Range<usize>],
        value: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_equal(
        lhs: crate::ops::BoolTensor<DummyBackend>,
        rhs: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_not(
        tensor: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_and(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        rhs: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_or(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        rhs: crate::ops::BoolTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_swap_dims(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        dim1: usize,
        dim2: usize,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_permute(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_flip(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn bool_expand(
        tensor: crate::ops::BoolTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }
}

impl ModuleOps<DummyBackend> for DummyBackend {
    fn conv2d(
        x: crate::ops::FloatTensor<DummyBackend>,
        weight: crate::ops::FloatTensor<DummyBackend>,
        bias: Option<crate::ops::FloatTensor<DummyBackend>>,
        options: crate::ops::ConvOptions<2>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn deform_conv2d(
        x: crate::ops::FloatTensor<DummyBackend>,
        offset: crate::ops::FloatTensor<DummyBackend>,
        weight: crate::ops::FloatTensor<DummyBackend>,
        mask: Option<crate::ops::FloatTensor<DummyBackend>>,
        bias: Option<crate::ops::FloatTensor<DummyBackend>>,
        options: crate::ops::DeformConvOptions<2>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn deform_conv2d_backward(
        x: crate::ops::FloatTensor<DummyBackend>,
        offset: crate::ops::FloatTensor<DummyBackend>,
        weight: crate::ops::FloatTensor<DummyBackend>,
        mask: Option<crate::ops::FloatTensor<DummyBackend>>,
        bias: Option<crate::ops::FloatTensor<DummyBackend>>,
        output_grad: crate::ops::FloatTensor<DummyBackend>,
        options: crate::ops::DeformConvOptions<2>,
    ) -> crate::ops::DeformConv2dBackward<DummyBackend> {
        todo!()
    }

    fn conv3d(
        x: crate::ops::FloatTensor<DummyBackend>,
        weight: crate::ops::FloatTensor<DummyBackend>,
        bias: Option<crate::ops::FloatTensor<DummyBackend>>,
        options: crate::ops::ConvOptions<3>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn conv_transpose2d(
        x: crate::ops::FloatTensor<DummyBackend>,
        weight: crate::ops::FloatTensor<DummyBackend>,
        bias: Option<crate::ops::FloatTensor<DummyBackend>>,
        options: crate::ops::ConvTransposeOptions<2>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn conv_transpose3d(
        x: crate::ops::FloatTensor<DummyBackend>,
        weight: crate::ops::FloatTensor<DummyBackend>,
        bias: Option<crate::ops::FloatTensor<DummyBackend>>,
        options: crate::ops::ConvTransposeOptions<3>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn avg_pool2d(
        x: crate::ops::FloatTensor<DummyBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn avg_pool2d_backward(
        x: crate::ops::FloatTensor<DummyBackend>,
        grad: crate::ops::FloatTensor<DummyBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn adaptive_avg_pool2d(
        x: crate::ops::FloatTensor<DummyBackend>,
        output_size: [usize; 2],
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn adaptive_avg_pool2d_backward(
        x: crate::ops::FloatTensor<DummyBackend>,
        grad: crate::ops::FloatTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn max_pool2d(
        x: crate::ops::FloatTensor<DummyBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn max_pool2d_with_indices(
        x: crate::ops::FloatTensor<DummyBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> crate::ops::MaxPool2dWithIndices<DummyBackend> {
        todo!()
    }

    fn max_pool2d_with_indices_backward(
        x: crate::ops::FloatTensor<DummyBackend>,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
        output_grad: crate::ops::FloatTensor<DummyBackend>,
        indices: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::MaxPool2dBackward<DummyBackend> {
        todo!()
    }

    fn interpolate(
        x: crate::ops::FloatTensor<DummyBackend>,
        output_size: [usize; 2],
        options: crate::ops::InterpolateOptions,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn interpolate_backward(
        x: crate::ops::FloatTensor<DummyBackend>,
        grad: crate::ops::FloatTensor<DummyBackend>,
        output_size: [usize; 2],
        options: crate::ops::InterpolateOptions,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }
}

impl IntTensorOps<DummyBackend> for DummyBackend {
    fn int_empty(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_into_data(
        tensor: crate::ops::IntTensor<DummyBackend>,
    ) -> impl Future<Output = crate::TensorData> + Send {
        async { todo!() }
    }

    fn int_from_data(
        data: crate::TensorData,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_device(tensor: &crate::ops::IntTensor<DummyBackend>) -> crate::Device<DummyBackend> {
        todo!()
    }

    fn int_to_device(
        tensor: crate::ops::IntTensor<DummyBackend>,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_reshape(
        tensor: crate::ops::IntTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_slice(
        tensor: crate::ops::IntTensor<DummyBackend>,
        indices: &[core::ops::Range<usize>],
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_slice_assign(
        tensor: crate::ops::IntTensor<DummyBackend>,
        indices: &[core::ops::Range<usize>],
        value: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_into_float(
        tensor: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::FloatTensor<DummyBackend> {
        todo!()
    }

    fn int_mask_where(
        tensor: crate::ops::IntTensor<DummyBackend>,
        mask: crate::ops::BoolTensor<DummyBackend>,
        source: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_mask_fill(
        tensor: crate::ops::IntTensor<DummyBackend>,
        mask: crate::ops::BoolTensor<DummyBackend>,
        value: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_gather(
        dim: usize,
        tensor: crate::ops::IntTensor<DummyBackend>,
        indices: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_scatter(
        dim: usize,
        tensor: crate::ops::IntTensor<DummyBackend>,
        indices: crate::ops::IntTensor<DummyBackend>,
        value: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_select(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
        indices: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_select_assign(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
        indices: crate::ops::IntTensor<DummyBackend>,
        value: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_equal(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_equal_elem(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_greater(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_greater_elem(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_greater_equal(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_greater_equal_elem(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_lower(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_lower_elem(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_lower_equal(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_lower_equal_elem(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::BoolTensor<DummyBackend> {
        todo!()
    }

    fn int_add(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_add_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_sub(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_sub_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_mul(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_mul_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_div(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_div_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_remainder(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_remainder_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_matmul(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_zeros(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_ones(
        shape: crate::Shape,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_sum(tensor: crate::ops::IntTensor<DummyBackend>) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_sum_dim(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_prod(
        tensor: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_prod_dim(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_mean_dim(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_argmax(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_argmin(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_abs(tensor: crate::ops::IntTensor<DummyBackend>) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_swap_dims(
        tensor: crate::ops::IntTensor<DummyBackend>,
        dim1: usize,
        dim2: usize,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_permute(
        tensor: crate::ops::IntTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_flip(
        tensor: crate::ops::IntTensor<DummyBackend>,
        axes: &[usize],
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_random(
        shape: crate::Shape,
        distribution: crate::Distribution,
        device: &crate::Device<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn int_expand(
        tensor: crate::ops::IntTensor<DummyBackend>,
        shape: crate::Shape,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_and(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_and_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_or(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_or_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_xor(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_xor_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_not(
        tensor: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_left_shift(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_left_shift_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_right_shift(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntTensor<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }

    fn bitwise_right_shift_scalar(
        lhs: crate::ops::IntTensor<DummyBackend>,
        rhs: crate::ops::IntElem<DummyBackend>,
    ) -> crate::ops::IntTensor<DummyBackend> {
        todo!()
    }
}
