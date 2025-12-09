use burn_fusion::stream::Context;
use burn_tensor::{DType, quantization::QParamTensor};
use cubecl::quant::scheme::{QuantParam, QuantScheme};
use cubecl::{
    CubeElement, Runtime,
    client::ComputeClient,
    ir::ElemType,
    prelude::{TensorArg, TensorHandleRef},
};
use std::marker::PhantomData;

/// Defines a fallback operation when fusion isn't possible.
pub trait FallbackOperation<R: Runtime>: Send + Sync {
    /// Executes the fallback procedure.
    fn run(&self, context: &mut Context<'_, CubeFusionHandle<R>>);
}

/// Runtime parameters for quantization. Can be used to construct a scales handle from the base
/// tensor handle.
pub type QParams = burn_tensor::quantization::QParams<QParamTensor>;

/// Handle to be used when fusing operations.
pub struct CubeFusionHandle<R: Runtime> {
    /// Compute client for jit.
    pub client: ComputeClient<R>,
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
