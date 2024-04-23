use super::{ElementWise, ElementWiseState};
use crate::{
    element::JitElement, fusion::ElementWiseBuilder, tensor::JitTensor, JitBackend, Runtime,
};
use burn_compute::client::ComputeClient;
use burn_fusion::{client::MutexFusionClient, FusionBackend};
use burn_tensor::{repr::ReprBackend, Shape};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// Fusion optimization type for JIT.
///
/// More optimization variants should be added here.
pub enum JitOptimization<R: Runtime> {
    /// Element wise optimization.
    ElementWise(ElementWise<R>),
}

/// Fusion optimization state type for JIT.
///
/// More optimization variants should be added here.
#[derive(Serialize, Deserialize)]
pub enum JitOptimizationState {
    /// Element wise state.
    ElementWise(ElementWiseState),
}

impl<R: Runtime> burn_fusion::Optimization<JitBackend<R>> for JitOptimization<R> {
    fn execute(&mut self, context: &mut burn_fusion::stream::Context<'_, JitBackend<R>>) {
        match self {
            Self::ElementWise(op) => op.execute(context),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::ElementWise(op) => op.len(),
        }
    }

    fn to_state(&self) -> JitOptimizationState {
        match self {
            Self::ElementWise(value) => JitOptimizationState::ElementWise(value.to_state()),
        }
    }

    fn from_state(device: &R::Device, state: JitOptimizationState) -> Self {
        match state {
            JitOptimizationState::ElementWise(state) => {
                Self::ElementWise(ElementWise::from_state(device, state))
            }
        }
    }
}

impl<R: Runtime> ReprBackend for JitBackend<R> {
    type Handle = JitFusionHandle<R>;

    fn float_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> burn_tensor::ops::FloatTensor<Self, D> {
        handle.into_tensor(shape)
    }

    fn int_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> burn_tensor::ops::IntTensor<Self, D> {
        handle.into_tensor(shape)
    }

    fn bool_tensor<const D: usize>(
        handle: Self::Handle,
        shape: Shape<D>,
    ) -> burn_tensor::ops::BoolTensor<Self, D> {
        handle.into_tensor(shape)
    }

    fn float_tensor_handle<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
    ) -> Self::Handle {
        tensor.into()
    }

    fn int_tensor_handle<const D: usize>(
        tensor: burn_tensor::ops::IntTensor<Self, D>,
    ) -> Self::Handle {
        tensor.into()
    }

    fn bool_tensor_handle<const D: usize>(
        tensor: burn_tensor::ops::BoolTensor<Self, D>,
    ) -> Self::Handle {
        tensor.into()
    }
}

impl<R: Runtime> FusionBackend for JitBackend<R> {
    type OptimizationState = JitOptimizationState;
    type Optimization = JitOptimization<R>;
    type FusionClient = MutexFusionClient<Self>;

    fn optimizations(
        device: R::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![Box::new(ElementWiseBuilder::new(device))]
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
pub struct JitFusionHandle<R: Runtime> {
    /// Compute client for jit.
    pub client: ComputeClient<R::Server, R::Channel>,
    /// The buffer where the data are stored.
    pub handle: burn_compute::server::Handle<R::Server>,
    /// The device of the current tensor.
    pub device: R::Device,
    pub(crate) strides: Vec<usize>,
}

impl<R: Runtime> core::fmt::Debug for JitFusionHandle<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "JitFusionHandle {{ device: {:?}, runtime: {}}}",
            self.device,
            R::name(),
        ))
    }
}

impl<R: Runtime> Clone for JitFusionHandle<R> {
    fn clone(&self) -> Self {
        Self {
            client: self.client.clone(),
            handle: self.handle.clone(),
            device: self.device.clone(),
            strides: self.strides.clone(),
        }
    }
}

unsafe impl<R: Runtime> Send for JitFusionHandle<R> {}
unsafe impl<R: Runtime> Sync for JitFusionHandle<R> {}

impl<R: Runtime> JitFusionHandle<R> {
    pub(crate) fn into_tensor<const D: usize, E: JitElement>(
        self,
        shape: Shape<D>,
    ) -> JitTensor<R, E, D> {
        JitTensor {
            client: self.client,
            handle: self.handle,
            device: self.device,
            shape,
            strides: self.strides.try_into().expect("Wrong dimension"),
            elem: PhantomData,
        }
    }
}

impl<R: Runtime, E: JitElement, const D: usize> From<JitTensor<R, E, D>> for JitFusionHandle<R> {
    fn from(value: JitTensor<R, E, D>) -> Self {
        Self {
            client: value.client,
            handle: value.handle,
            device: value.device,
            strides: value.strides.into(),
        }
    }
}
