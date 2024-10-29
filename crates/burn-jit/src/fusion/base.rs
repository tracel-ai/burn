use super::elemwise::optimization::{ElemwiseOptimization, ElemwiseOptimizationState};
use crate::fusion::elemwise::builder::ElementWiseBuilder;
use crate::tensor::{JitQuantizationParameters, QJitTensor};
use crate::{
    element::JitElement, kernel, tensor::JitTensor, FloatElement, IntElement, JitBackend,
    JitRuntime,
};
use burn_fusion::{client::MutexFusionClient, FusionBackend, FusionRuntime};
use burn_tensor::quantization::QuantizationScheme;
use burn_tensor::repr::{QuantizedKind, TensorHandle};
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
    ElementWise2(ElemwiseOptimization<R>),
}

/// Fusion optimization state type for JIT.
///
/// More optimization variants should be added here.
#[derive(Serialize, Deserialize)]
pub enum JitOptimizationState {
    /// Element wise state.
    ElementWise(ElemwiseOptimizationState),
}

impl<R> burn_fusion::Optimization<FusionJitRuntime<R>> for JitOptimization<R>
where
    R: JitRuntime,
{
    fn execute(&mut self, context: &mut burn_fusion::stream::Context<'_, JitFusionHandle<R>>) {
        match self {
            Self::ElementWise2(op) => op.execute(context),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::ElementWise2(op) => op.num_ops_fused(),
        }
    }

    fn to_state(&self) -> JitOptimizationState {
        match self {
            Self::ElementWise2(value) => JitOptimizationState::ElementWise(value.to_state()),
        }
    }

    fn from_state(device: &R::Device, state: JitOptimizationState) -> Self {
        match state {
            JitOptimizationState::ElementWise(state) => {
                Self::ElementWise2(ElemwiseOptimization::from_state(device, state))
            }
        }
    }
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> ReprBackend for JitBackend<R, F, I> {
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
        handles: QuantizedKind<TensorHandle<Self::Handle>>,
        scheme: QuantizationScheme,
    ) -> burn_tensor::ops::QuantizedTensor<Self> {
        let qtensor = handles.tensor.handle.into_tensor(handles.tensor.shape);
        let scale = handles.scale.handle.into_tensor(handles.scale.shape);
        let offset = handles.offset;

        let qparams = JitQuantizationParameters {
            scale,
            offset: offset.map(|h| h.handle.into_tensor(h.shape)),
        };

        QJitTensor {
            qtensor,
            scheme,
            qparams,
        }
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

    fn quantized_tensor_handle(
        tensor: burn_tensor::ops::QuantizedTensor<Self>,
    ) -> QuantizedKind<Self::Handle> {
        let qtensor: JitFusionHandle<R> = tensor.qtensor.into();
        let scale: JitFusionHandle<R> = tensor.qparams.scale.into();

        QuantizedKind {
            tensor: qtensor,
            scale,
            offset: tensor.qparams.offset.map(|offset| offset.into()),
        }
    }
}

impl<R: JitRuntime> FusionRuntime for FusionJitRuntime<R> {
    type OptimizationState = JitOptimizationState;
    type Optimization = JitOptimization<R>;
    type FusionHandle = JitFusionHandle<R>;
    type FusionDevice = R::JitDevice;
    type FusionClient = MutexFusionClient<Self>;

    fn optimizations(
        device: R::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![Box::new(ElementWiseBuilder::<R>::new(device.clone()))]
    }
}

#[derive(Debug)]
pub struct FusionJitRuntime<R: JitRuntime> {
    _b: PhantomData<R>,
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> FusionBackend for JitBackend<R, F, I> {
    type FusionRuntime = FusionJitRuntime<R>;

    type FullPrecisionBackend = JitBackend<R, f32, i32>;

    fn cast_float(
        tensor: burn_tensor::ops::FloatTensor<Self>,
        dtype: burn_tensor::DType,
    ) -> Self::Handle {
        fn cast<R: JitRuntime, F: FloatElement, FTarget: FloatElement>(
            tensor: JitTensor<R, F>,
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

pub fn strides_dyn_rank(shape: &[usize]) -> Vec<usize> {
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
        }
    }
}

unsafe impl<R: JitRuntime> Send for JitFusionHandle<R> {}
unsafe impl<R: JitRuntime> Sync for JitFusionHandle<R> {}

impl<R: JitRuntime> JitFusionHandle<R> {
    pub(crate) fn into_tensor<E: JitElement>(self, shape: Shape) -> JitTensor<R, E> {
        JitTensor {
            client: self.client,
            handle: self.handle,
            device: self.device,
            shape,
            strides: self.strides,
            elem: PhantomData,
        }
    }
    /// Return the reference to a tensor handle.
    pub fn as_handle_ref<'a>(&'a self, shape: &'a [usize]) -> TensorHandleRef<'a, R> {
        TensorHandleRef {
            handle: &self.handle,
            strides: &self.strides,
            shape,
            runtime: PhantomData,
        }
    }
    /// Return the reference to a tensor argument.
    pub fn as_tensor_arg<'a>(&'a self, shape: &'a [usize], vectorisation: u8) -> TensorArg<'a, R> {
        let handle: TensorHandleRef<'a, R> = self.as_handle_ref(shape);

        unsafe {
            TensorArg::from_raw_parts(handle.handle, handle.strides, handle.shape, vectorisation)
        }
    }
}

impl<R: JitRuntime, E: JitElement> From<JitTensor<R, E>> for JitFusionHandle<R> {
    fn from(value: JitTensor<R, E>) -> Self {
        Self {
            client: value.client,
            handle: value.handle,
            device: value.device,
            strides: value.strides,
        }
    }
}
