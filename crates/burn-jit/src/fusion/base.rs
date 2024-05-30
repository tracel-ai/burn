use super::{ElementWise, ElementWiseState};
use crate::{
    element::JitElement, fusion::ElementWiseBuilder, kernel, tensor::JitTensor, FloatElement,
    IntElement, JitBackend, JitRuntime,
};
use burn_compute::client::ComputeClient;
use burn_cube::{ir::ReadingStrategy, InplaceMapping, KernelExpansion, KernelSettings};
use burn_fusion::{client::MutexFusionClient, FusionBackend, FusionRuntime};
use burn_tensor::{repr::ReprBackend, Shape};
use core::marker::PhantomData;
use half::{bf16, f16};
use serde::{Deserialize, Serialize};

/// Fusion optimization type for JIT.
///
/// More optimization variants should be added here.
pub enum JitOptimization<R: JitRuntime> {
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

impl<R> burn_fusion::Optimization<FusionJitRuntime<R>> for JitOptimization<R>
where
    R: JitRuntime,
{
    fn execute(&mut self, context: &mut burn_fusion::stream::Context<'_, JitFusionHandle<R>>) {
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

impl<R: JitRuntime, F: FloatElement, I: IntElement> ReprBackend for JitBackend<R, F, I> {
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

impl<R: JitRuntime> FusionRuntime for FusionJitRuntime<R> {
    type OptimizationState = JitOptimizationState;
    type Optimization = JitOptimization<R>;
    type FusionHandle = JitFusionHandle<R>;
    type FusionDevice = R::JitDevice;
    type FusionClient = MutexFusionClient<Self>;

    fn optimizations(
        device: R::Device,
    ) -> Vec<Box<dyn burn_fusion::OptimizationBuilder<Self::Optimization>>> {
        vec![Box::new(ElementWiseBuilder::<R>::new(device))]
    }
}

#[derive(Debug)]
pub struct FusionJitRuntime<R: JitRuntime> {
    _b: PhantomData<R>,
}

impl<R: JitRuntime, F: FloatElement, I: IntElement> FusionBackend for JitBackend<R, F, I> {
    type FusionRuntime = FusionJitRuntime<R>;

    type FullPrecisionBackend = JitBackend<R, f32, i32>;

    fn cast_float<const D: usize>(
        tensor: burn_tensor::ops::FloatTensor<Self, D>,
        dtype: burn_tensor::DType,
    ) -> Self::Handle {
        fn cast<const D: usize, R: JitRuntime, F: FloatElement, FTarget: FloatElement>(
            tensor: JitTensor<R, F, D>,
        ) -> JitFusionHandle<R> {
            JitFusionHandle::from(kernel::cast::<R, F, FTarget, D>(tensor))
        }

        match dtype {
            burn_tensor::DType::F32 => cast::<D, R, F, f32>(tensor),
            burn_tensor::DType::F16 => cast::<D, R, F, f16>(tensor),
            burn_tensor::DType::BF16 => cast::<D, R, F, bf16>(tensor),
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
    pub handle: burn_compute::server::Handle<R::Server>,
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

impl<R: JitRuntime, E: JitElement, const D: usize> From<JitTensor<R, E, D>> for JitFusionHandle<R> {
    fn from(value: JitTensor<R, E, D>) -> Self {
        Self {
            client: value.client,
            handle: value.handle,
            device: value.device,
            strides: value.strides.into(),
        }
    }
}

/// Apply dynamic settings based on the runtime information captured by the `burn-fusion`
/// project.
///
/// Two optimizations are done here:
///
/// 1. Find and remove unnecessary broadcasting procedures based on runtime tensor layouts.
///
/// 2. (Optional) Find which inputs can be used inplaced based on runtime tensor layouts and captured tensor
///    descriptions. This is enabled only when stateful is set to true.
pub fn dynamic_settings<R: JitRuntime>(
    mut settings: KernelSettings,
    info: &KernelExpansion,
    inputs: &[&burn_tensor::repr::TensorDescription],
    outputs: &[&burn_tensor::repr::TensorDescription],
    handles_inputs: &[JitFusionHandle<R>],
    stateful: bool,
) -> KernelSettings {
    if stateful {
        settings = dynamic_inplace(settings, info, inputs, outputs, handles_inputs);
    }

    dynamic_reading_strategy(settings, info, inputs, outputs, handles_inputs)
}

fn dynamic_inplace<R: JitRuntime>(
    settings: KernelSettings,
    info: &KernelExpansion,
    inputs: &[&burn_tensor::repr::TensorDescription],
    outputs: &[&burn_tensor::repr::TensorDescription],
    handles_inputs: &[JitFusionHandle<R>],
) -> KernelSettings {
    let mut potential_inplace = inputs
        .iter()
        .zip(info.inputs.iter())
        .enumerate()
        .filter_map(|(pos, (desc, input))| {
            match desc.status {
                burn_tensor::repr::TensorStatus::ReadOnly => return None,
                burn_tensor::repr::TensorStatus::NotInit => return None,
                burn_tensor::repr::TensorStatus::ReadWrite => (),
            };

            let handle = &handles_inputs[pos];

            if handle.handle.can_mut() && is_contiguous(&handle.strides) {
                Some((pos, desc, input))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let mappings = outputs
        .iter()
        .zip(info.outputs.iter())
        .enumerate()
        .filter_map(|(pos, (desc, output))| {
            if potential_inplace.is_empty() {
                return None;
            }

            for (index, (_, desc_input, input)) in potential_inplace.iter().enumerate() {
                if desc.shape == desc_input.shape && input.item() == output.item() {
                    let (pos_input, _desc, _info) = potential_inplace.remove(index);
                    return Some(InplaceMapping::new(pos_input, pos));
                }
            }

            None
        })
        .collect();

    settings.inplace(mappings)
}

fn dynamic_reading_strategy<R: JitRuntime>(
    mut settings: KernelSettings,
    info: &KernelExpansion,
    inputs: &[&burn_tensor::repr::TensorDescription],
    outputs: &[&burn_tensor::repr::TensorDescription],
    handles_inputs: &[JitFusionHandle<R>],
) -> KernelSettings {
    // First output is chosen for the layout reference.
    // but all outputs should have the same shape anyways.
    let layout_shape = &outputs[0].shape;

    for (input_id, strategy) in info.scope.read_globals() {
        if let ReadingStrategy::Plain = strategy {
            continue;
        };

        let index = input_id as usize;
        let handle = &handles_inputs[index];
        let description_input = &inputs[index];

        if &description_input.shape != layout_shape {
            continue;
        }

        if is_contiguous(&handle.strides) {
            settings
                .reading_strategy
                .push((input_id, ReadingStrategy::Plain));
        }
    }
    settings
}

fn is_contiguous(strides: &[usize]) -> bool {
    let mut current = 0;

    for stride in strides.iter().rev() {
        if current > *stride {
            return false;
        }
        current = *stride;
    }

    true
}
