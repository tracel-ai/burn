use crate::reduce::optimization::{ReduceOptimization, ReduceOptimizationState};
use std::marker::PhantomData;

use super::elemwise::optimization::{ElemwiseOptimization, ElemwiseOptimizationState};
use super::matmul::optimization::{MatmulOptimization, MatmulOptimizationState};

use burn_fusion::stream::Context;
use burn_tensor::DType;
use burn_tensor::quantization::{QParamTensor, QuantParam, QuantScheme};
use cubecl::prelude::{TensorArg, TensorHandleRef};
use cubecl::{CubeElement, Runtime};
use cubecl::{client::ComputeClient, ir::ElemType};
use serde::{Deserialize, Serialize};

/// Fusion optimization type for cubecl.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
pub enum CubeOptimization<R: Runtime> {
    /// Element wise optimization.
    ElementWise(ElemwiseOptimization<R>),
    /// Matrix multiplication optimization.
    Matmul(MatmulOptimization<R>),
    Reduce(ReduceOptimization<R>),
}

impl<R: Runtime> core::fmt::Debug for CubeOptimization<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = self.to_opt_state();
        f.write_fmt(format_args!("{value:?}"))
    }
}

impl<R: Runtime> CubeOptimization<R> {
    pub fn to_opt_state(&self) -> CubeOptimizationState {
        match self {
            Self::ElementWise(value) => CubeOptimizationState::ElementWise(value.to_state()),
            Self::Matmul(value) => CubeOptimizationState::Matmul(value.to_state()),
            Self::Reduce(value) => CubeOptimizationState::Reduce(value.to_state()),
        }
    }
}

impl<R: Runtime> burn_fusion::NumOperations for CubeOptimization<R> {
    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.num_ops_fused(),
            Self::Matmul(op) => op.num_ops_fused(),
            Self::Reduce(op) => op.num_ops_fused(),
        }
    }
}

/// Fusion optimization state type for cubecl.
///
/// More optimization variants should be added here.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize, Debug)]
pub enum CubeOptimizationState {
    /// Element wise state.
    ElementWise(ElemwiseOptimizationState),
    /// Matrix multiplication optimization state.
    Matmul(MatmulOptimizationState),
    Reduce(ReduceOptimizationState),
}

pub trait FallbackOperation<R: Runtime>: Send + Sync {
    fn run(&self, context: &mut Context<'_, CubeFusionHandle<R>>);
}

pub(crate) fn strides_dyn_rank(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];

    let mut current = 1;
    shape.iter().enumerate().rev().for_each(|(index, val)| {
        strides[index] = current;
        current *= val;
    });

    strides
}

pub(crate) fn elem_dtype<E: CubeElement>() -> DType {
    match E::cube_type().elem_type() {
        ElemType::Float(kind) => match kind {
            cubecl::ir::FloatKind::F64 => DType::F64,
            cubecl::ir::FloatKind::F16 => DType::F16,
            cubecl::ir::FloatKind::BF16 => DType::BF16,
            cubecl::ir::FloatKind::F32 => DType::F32,
            _ => todo!(),
        },
        ElemType::Int(kind) => match kind {
            cubecl::ir::IntKind::I64 => DType::I64,
            cubecl::ir::IntKind::I32 => DType::I32,
            cubecl::ir::IntKind::I16 => DType::I16,
            cubecl::ir::IntKind::I8 => DType::I8,
        },
        ElemType::UInt(kind) => match kind {
            cubecl::ir::UIntKind::U64 => DType::U64,
            cubecl::ir::UIntKind::U32 => DType::U32,
            cubecl::ir::UIntKind::U16 => DType::U16,
            cubecl::ir::UIntKind::U8 => DType::U8,
        },
        ElemType::Bool => DType::Bool,
    }
}

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
pub type QParams = burn_tensor::quantization::QParams<QParamTensor>;

/// Handle to be used when fusing operations.
pub struct CubeFusionHandle<R: Runtime> {
    /// Compute client for jit.
    pub client: ComputeClient<R::Server>,
    /// The buffer where the data are stored.
    pub handle: cubecl::server::Handle,
    /// The device of the current tensor.
    pub device: R::Device,
    /// The element type of the tensor.
    pub dtype: DType,
    /// The strides of the tensor.
    pub strides: Vec<usize>,
    /// Quantization runtime parameters, if applicable
    pub qparams: Option<QParams>,
}

impl<R: Runtime> core::fmt::Debug for CubeFusionHandle<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "CubeFusionHandle {{ device: {:?}, runtime: {}}}",
            self.device,
            R::name(&self.client),
        ))
    }
}

impl<R: Runtime> Clone for CubeFusionHandle<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
            qparams: self.qparams.clone(),
        }
    }
}

unsafe impl<R: Runtime> Send for CubeFusionHandle<R> {}
unsafe impl<R: Runtime> Sync for CubeFusionHandle<R> {}

impl<R: Runtime> CubeFusionHandle<R> {
    /// Return the reference to a tensor handle.
    pub fn as_handle_ref<'a>(&'a self, shape: &'a [usize]) -> TensorHandleRef<'a, R> {
        TensorHandleRef {
            handle: &self.handle,
            strides: &self.strides,
            shape,
            runtime: PhantomData,
            elem_size: self.dtype.size(),
        }
    }
    /// Return the reference to a tensor argument.
    pub fn as_tensor_arg<'a>(&'a self, shape: &'a [usize], vectorisation: u8) -> TensorArg<'a, R> {
        let handle: TensorHandleRef<'a, R> = self.as_handle_ref(shape);

        unsafe {
            TensorArg::from_raw_parts_and_size(
                handle.handle,
                handle.strides,
                handle.shape,
                vectorisation,
                self.dtype.size(),
            )
        }
    }
    /// Construct a separate tensor for the quantization scales, if present
    pub fn params(&self, scheme: QuantScheme) -> Option<Self> {
        let qparams = self.qparams.as_ref()?;
        let mut handle = self.handle.clone();
        handle.offset_start = Some(qparams.scales.offset_start as u64);
        handle.offset_end = Some(qparams.scales.offset_end as u64);

        Some(Self {
            client: self.client.clone(),
            handle,
            device: self.device.clone(),
            dtype: match scheme.param {
                QuantParam::F32 => DType::F32,
                QuantParam::F16 => DType::F16,
                QuantParam::BF16 => DType::BF16,
                QuantParam::UE8M0 | QuantParam::UE4M3 => unimplemented!("Not yet supported"),
            },
            strides: qparams.scales.strides.clone(),
            qparams: None,
        })
    }
}
