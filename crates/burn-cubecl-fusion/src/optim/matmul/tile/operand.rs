//! A fused product's operand as the trace holds it ([`TraceOperand`]).

use super::TileFusedMatmulError;
use crate::CubeFusionHandle;
use burn_backend::cubecl::dtype_to_storage_type;
use burn_fusion::stream::Context;
use burn_ir::TensorIr;
use burn_std::{DType, quantization::QParamTensor};
use cubecl::{prelude::TensorBinding, tile::Axis, zspace::metadata::Metadata};
use cubek::matmul::kernels::{
    plan::MatmulCallInput,
    select::problem::OperandKind,
    tensor::{PackedTensor, PlainTensor, StoredTensor},
};

/// One operand of a fused product: the tensor the trace names, and the handle it lies in.
pub(super) struct TraceOperand<'a> {
    tensor: &'a TensorIr,
    handle: &'a CubeFusionHandle,
}

impl<'a> TraceOperand<'a> {
    /// The operand `tensor` names in `context`.
    pub(super) fn new(context: &'a Context<CubeFusionHandle>, tensor: &TensorIr) -> Self {
        let tensor = &context.tensors[&tensor.id];
        let handle = context
            .handles
            .get_handle_ref(&tensor.id)
            .expect("a fused product's operand has a handle");
        Self { tensor, handle }
    }

    /// Its metadata and what kind of operand it is, as a matmul's autotune key reads them.
    pub(super) fn key(&self) -> (Metadata, OperandKind) {
        let kind = match self.tensor.dtype {
            DType::QFloat(scheme) => OperandKind::Quantized(scheme),
            _ => OperandKind::Float,
        };
        (self.handle.metadata(self.tensor.shape.clone()), kind)
    }

    /// It as one matrix over `axes`, bound as it is stored where its handle lies: plain, or,
    /// quantized, packed under its scales ([`PackedTensor::of_quantized`]).
    pub(super) fn call_input(
        &self,
        axes: [Axis; 2],
    ) -> Result<MatmulCallInput, TileFusedMatmulError> {
        let (tensor, handle) = (self.tensor, self.handle);
        let meta = handle.metadata(tensor.shape.clone());
        let values = handle.clone().binding(tensor.shape.clone());
        let DType::QFloat(scheme) = tensor.dtype else {
            let elem = dtype_to_storage_type(tensor.dtype);
            return Ok(MatmulCallInput::plain(PlainTensor::new(values, elem)));
        };
        let qparams = handle
            .qparams
            .as_ref()
            .expect("a quantized operand holds its scales");
        let block_scales = self.param(&qparams.scales);
        let global_scale = qparams.global.as_ref().map(|scale| self.param(scale));
        let packed = PackedTensor::of_quantized(
            &handle.client,
            values,
            scheme,
            block_scales,
            global_scale,
            axes,
        )
        .map_err(TileFusedMatmulError::Operand)?;
        Ok(MatmulCallInput {
            tensor: StoredTensor::Packed(packed),
            meta,
            framework_dequantize: None,
        })
    }

    /// The quantization parameter `param` of this quantized operand, bound where it lies in the
    /// operand's buffer.
    fn param(&self, param: &QParamTensor) -> PlainTensor {
        let mut buffer = self.handle.handle.clone();
        buffer.offset_start = Some(param.offset_start as u64);
        buffer.offset_end = Some(param.offset_end as u64);
        let meta = &param.metadata;
        let binding = TensorBinding {
            handle: buffer.binding(),
            strides: meta.strides.clone(),
            shape: meta.shape.clone(),
            tiling: meta.tiling,
        };
        PlainTensor::new(binding, dtype_to_storage_type(param.dtype))
    }
}
