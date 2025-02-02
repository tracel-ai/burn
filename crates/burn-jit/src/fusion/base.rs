use super::elemwise::optimization::{ElemwiseOptimization, ElemwiseOptimizationState};
use super::matmul::optimization::{MatmulOptimization, MatmulOptimizationState};
use crate::fusion::elemwise::builder::ElementWiseBuilder;
use crate::fusion::matmul::builder::MatmulBuilder;
use crate::BoolElement;
use crate::{kernel, tensor::JitTensor, FloatElement, IntElement, JitBackend, JitRuntime};

use burn_fusion::{client::MutexFusionClient, FusionBackend, FusionRuntime};
use burn_tensor::repr::TensorHandle;
use burn_tensor::DType;
use burn_tensor::{repr::ReprBackend, Shape};
use core::marker::PhantomData;
use cubecl::client::ComputeClient;
use cubecl::prelude::{TensorArg, TensorHandleRef};
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

/// Fusion optimization type for JIT.
///
/// More optimization variants should be added here.
pub enum JitOptimization<R: JitRuntime> {
    /// Element wise optimization.
    ElementWise(ElemwiseOptimization<R>),
    /// Matrix multiplication optimization.
    Matmul(MatmulOptimization<R>),
}

/// Fusion optimization state type for JIT.
///
/// More optimization variants should be added here.
#[derive(Serialize, Deserialize)]
pub enum JitOptimizationState {
    /// Element wise state.
    ElementWise(ElemwiseOptimizationState),
    /// Matrix multiplication optimization state.
    Matmul(MatmulOptimizationState),
}

impl<R, BT> burn_fusion::Optimization<FusionJitRuntime<R, BT>> for JitOptimization<R>
where
    R: JitRuntime,
    BT: BoolElement,
{
    fn execute(&mut self, context: &mut burn_fusion::stream::Context<'_, JitFusionHandle<R>>) {
        match self {
            Self::ElementWise(op) => op.execute::<BT>(context),
            Self::Matmul(op) => op.execute::<BT>(context),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.num_ops_fused(),
            Self::Matmul(op) => op.num_ops_fused(),
        }
    }

    fn to_state(&self) -> JitOptimizationState {
        match self {
            Self::ElementWise(value) => JitOptimizationState::ElementWise(value.to_state()),
            Self::Matmul(value) => JitOptimizationState::Matmul(value.to_state()),
        }
    }

    fn from_state(device: &R::Device, state: JitOptimizationState) -> Self {
        match state {
            JitOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElemwiseOptimization::from_state(device, state))
            }
            JitOptimizationState::Matmul(state) => {
                Self::Matmul(MatmulOptimization::from_state(device, state))
            }
        }
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> ReprBackend
    for JitBackend<R, F, I, BT>
{
    type Handle = JitFusionHandle<R>;

    fn float_tensor(handle: TensorHandle<Self::Handle>) -> burn_tensor::ops::FloatTensor<Self> {
        handle.handle.into_tensor(handle.shape)
    }

    fn int_tensor(handle: TensorHandle<Self::Handle>) -> burn_tensor::ops::IntTensor<Self> {
        handle.handle.into_tensor(handle.shape)
    }

    fn bool_tensor(handle: TensorHandle<Self::Handle>) -> burn_tensor::ops::BoolTensor<Self> {
        handle.handle.into_tensor(handle.shape)
    }

    fn quantized_tensor(
        handle: TensorHandle<Self::Handle>,
    ) -> burn_tensor::ops::QuantizedTensor<Self> {
        handle.handle.into_tensor(handle.shape)
    }

    fn float_tensor_handle(tensor: burn_tensor::ops::FloatTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle(tensor: burn_tensor::ops::IntTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle(tensor: burn_tensor::ops::BoolTensor<Self>) -> Self::Handle {
        tensor.into()
    }

    fn quantized_tensor_handle(tensor: burn_tensor::ops::QuantizedTensor<Self>) -> Self::Handle {
        tensor.into()
    }
}

impl<R: JitRuntime, BT: BoolElement> FusionRuntime for FusionJitRuntime<R, BT> {
    type OptimizationState = JitOptimizationState;
    type Optimization = JitOptimization<R>;
    type FusionHandle = JitFusionHandle<R>;
    type FusionDevice = R::JitDevice;
    type FusionClient = MutexFusionClient<Self>;
    type BoolRepr = BT;

    fn optimizations(
        device: R::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![
            Box::new(ElementWiseBuilder::<R>::new(
                device.clone(),
                BT::as_elem_native_unchecked().into(),
            )),
            Box::new(MatmulBuilder::<R>::new(
                device.clone(),
                BT::as_elem_native_unchecked().into(),
            )),
        ]
    }
}

/// Fusion runtime for JIT runtimes.
#[derive(Debug)]
pub struct FusionJitRuntime<R: JitRuntime, BT: BoolElement> {
    _b: PhantomData<R>,
    _bool: PhantomData<BT>,
}

impl<R: JitRuntime, F: FloatElement, I: IntElement, BT: BoolElement> FusionBackend
    for JitBackend<R, F, I, BT>
{
    type FusionRuntime = FusionJitRuntime<R, BT>;

    type FullPrecisionBackend = JitBackend<R, f32, i32, BT>;

    fn cast_float(
        tensor: burn_tensor::ops::FloatTensor<Self>,
        dtype: burn_tensor::DType,
    ) -> Self::Handle {
        fn cast<R: JitRuntime, F: FloatElement, FTarget: FloatElement>(
            tensor: JitTensor<R>,
        ) -> JitFusionHandle<R> {
            JitFusionHandle::from(kernel::cast::<R, F, FTarget>(tensor))
        }

        match dtype {
            burn_tensor::DType::F32 => cast::<R, F, f32>(tensor),
            burn_tensor::DType::F16 => cast::<R, F, f16>(tensor),
            burn_tensor::DType::BF16 => cast::<R, F, bf16>(tensor),
            _ => panic!("Casting error: {dtype:?} unsupported."),
        }
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

/// Handle to be used when fusing operations.
pub struct JitFusionHandle<R: JitRuntime> {
    /// Compute client for jit.
    pub client: ComputeClient<R::Server, R::Channel>,
    /// The buffer where the data are stored.
    pub handle: cubecl::server::Handle,
    /// The device of the current tensor.
    pub device: R::Device,
    pub(crate) dtype: DType,
    pub(crate) strides: Vec<usize>,
}

impl<R: JitRuntime> core::fmt::Debug for JitFusionHandle<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "JitFusionHandle {{ device: {:?}, runtime: {}}}",
            self.device,
            R::name(),
        ))
    }
}

impl<R: JitRuntime> Clone for JitFusionHandle<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
            dtype: self.dtype,
        }
    }
}

unsafe impl<R: JitRuntime> Send for JitFusionHandle<R> {}
unsafe impl<R: JitRuntime> Sync for JitFusionHandle<R> {}

impl<R: JitRuntime> JitFusionHandle<R> {
    pub(crate) fn into_tensor(self, shape: Shape) -> JitTensor<R> {
        JitTensor {
            client: self.client,
            handle: self.handle,
            device: self.device,
            shape,
            strides: self.strides,
            dtype: self.dtype,
        }
    }
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
}

impl<R: JitRuntime> From<JitTensor<R>> for JitFusionHandle<R> {
    fn from(value: JitTensor<R>) -> Self {
        Self {
            client: value.client,
            handle: value.handle,
            device: value.device,
            strides: value.strides,
            dtype: value.dtype,
        }
    }
}
