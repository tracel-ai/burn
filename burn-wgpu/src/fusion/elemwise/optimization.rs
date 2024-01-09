use super::kernel::{InplaceElementWise, ScalarElementWise, VecElementWise};
use crate::{
    codegen::{
        Elem, ElemWiseKernelCodegen, InplaceMapping, Input, Item, Operator, Output,
        ReadingStrategy, Vectorization, Visibility,
    },
    fusion::{
        kernel::{FusionKernel, FusionKernelSet},
        source::FusedKernelSource,
    },
    FloatElement, GraphicsApi, IntElement, Wgpu, WgpuDevice,
};
use burn_common::id::IdGenerator;
use burn_fusion::{graph::Context, TensorDescription};
use burn_tensor::Device;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(new)]
pub struct ElementWise<G, F, I, Phase = ExecutionPhase>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    inputs: Vec<(TensorDescription, Elem)>,
    outputs: Vec<(TensorDescription, Elem)>,
    scalars_f32: usize,
    scalars_u32: usize,
    scalars_i32: usize,
    device: Device<Wgpu<G, F, I>>,
    phase: Phase,
}

#[derive(new)]
pub struct CompilationPhase {
    locals: Vec<u16>,
    operators: Vec<Operator>,
}

#[derive(new)]
pub struct ExecutionPhase {
    operation_len: usize,
    kernel_set: FusionKernelSet,
}

#[derive(Serialize, Deserialize)]
pub struct ElementWiseState {
    inputs: Vec<(TensorDescription, Elem)>,
    outputs: Vec<(TensorDescription, Elem)>,
    operation_len: usize,
    scalars_f32: usize,
    scalars_u32: usize,
    scalars_i32: usize,
    kernels: Vec<FusedKernelSource>,
}

impl<G, F, I> ElementWise<G, F, I, CompilationPhase>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub(crate) fn compile(self) -> ElementWise<G, F, I, ExecutionPhase> {
        let mut inputs = self
            .inputs
            .iter()
            .map(|(_tensor, elem)| Input::Array {
                item: Item::Scalar(*elem),
                visibility: Visibility::Read,
                strategy: ReadingStrategy::OutputLayout,
            })
            .collect::<Vec<_>>();

        let outputs = self
            .outputs
            .iter()
            .zip(self.phase.locals.iter())
            .map(|((_tensor, elem), local)| Output::Array {
                item: Item::Scalar(*elem),
                local: *local,
            })
            .collect::<Vec<_>>();

        if self.scalars_f32 > 0 {
            inputs.push(Input::Scalar {
                elem: Elem::F32,
                size: self.scalars_f32,
            })
        }

        if self.scalars_u32 > 0 {
            inputs.push(Input::Scalar {
                elem: Elem::U32,
                size: self.scalars_u32,
            })
        }

        if self.scalars_i32 > 0 {
            inputs.push(Input::Scalar {
                elem: Elem::I32,
                size: self.scalars_i32,
            })
        }

        let scalar = ScalarElementWise::new(Arc::new(FusedKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .inputs(&inputs)
                .body(&self.phase.operators)
                .outputs(&outputs)
                .compile(),
        )));
        let vec2 = VecElementWise::<2>::new(Arc::new(FusedKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .vectorize(Vectorization::Vec2)
                .inputs(&inputs)
                .body(&self.phase.operators)
                .outputs(&outputs)
                .compile(),
        )));
        let vec4 = VecElementWise::<4>::new(Arc::new(FusedKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .vectorize(Vectorization::Vec4)
                .inputs(&inputs)
                .body(&self.phase.operators)
                .outputs(&outputs)
                .compile(),
        )));

        let mut potential_inplace = self
            .inputs
            .iter()
            .zip(inputs.iter())
            .enumerate()
            .filter(|(_pos, ((desc, _elem), _input))| match desc.status {
                burn_fusion::TensorStatus::ReadOnly => false,
                burn_fusion::TensorStatus::ReadWrite => true,
                burn_fusion::TensorStatus::NotInit => false,
            })
            .map(|(pos, ((desc, elem), input))| (pos, desc, elem, input))
            .collect::<Vec<_>>();

        let mut kernel_set: Vec<Box<dyn FusionKernel>> =
            vec![Box::new(scalar), Box::new(vec2), Box::new(vec4)];

        let mapping = self
            .outputs
            .iter()
            .zip(outputs.iter())
            .enumerate()
            .filter_map(|(pos, ((desc, elem), _output))| {
                if potential_inplace.is_empty() {
                    return None;
                }

                let mut chosen = None;
                for (index, (_pos_input, desc_input, elem_input, _input)) in
                    potential_inplace.iter().enumerate()
                {
                    if chosen.is_some() {
                        break;
                    }
                    if desc.shape == desc_input.shape && *elem_input == elem {
                        chosen = Some(index);
                    }
                }

                match chosen {
                    Some(index) => {
                        let input = potential_inplace.remove(index);
                        Some(InplaceMapping::new(input.0, pos))
                    }
                    None => None,
                }
            })
            .collect::<Vec<_>>();

        if !mapping.is_empty() {
            let scalar = ScalarElementWise::new(Arc::new(FusedKernelSource::new(
                IdGenerator::generate(),
                ElemWiseKernelCodegen::new()
                    .inputs(&inputs)
                    .body(&self.phase.operators)
                    .inplace_mapping(&mapping)
                    .outputs(&outputs)
                    .compile(),
            )));
            let vec2 = VecElementWise::<2>::new(Arc::new(FusedKernelSource::new(
                IdGenerator::generate(),
                ElemWiseKernelCodegen::new()
                    .vectorize(Vectorization::Vec2)
                    .inputs(&inputs)
                    .body(&self.phase.operators)
                    .inplace_mapping(&mapping)
                    .outputs(&outputs)
                    .compile(),
            )));
            let vec4 = VecElementWise::<4>::new(Arc::new(FusedKernelSource::new(
                IdGenerator::generate(),
                ElemWiseKernelCodegen::new()
                    .vectorize(Vectorization::Vec4)
                    .inputs(&inputs)
                    .body(&self.phase.operators)
                    .inplace_mapping(&mapping)
                    .outputs(&outputs)
                    .compile(),
            )));

            kernel_set.push(Box::new(InplaceElementWise::new(
                Box::new(scalar),
                mapping.clone(),
            )));
            kernel_set.push(Box::new(InplaceElementWise::new(
                Box::new(vec2),
                mapping.clone(),
            )));
            kernel_set.push(Box::new(InplaceElementWise::new(
                Box::new(vec4),
                mapping.clone(),
            )));
        }

        let kernel_set = FusionKernelSet::new(kernel_set);

        ElementWise {
            inputs: self.inputs,
            outputs: self.outputs,
            scalars_f32: self.scalars_f32,
            scalars_i32: self.scalars_i32,
            scalars_u32: self.scalars_u32,
            device: self.device,
            phase: ExecutionPhase::new(self.phase.operators.len(), kernel_set),
        }
    }
}

impl<G, F, I> ElementWise<G, F, I, ExecutionPhase>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub(crate) fn execute(&mut self, context: &mut Context<'_, Wgpu<G, F, I>>) {
        self.phase.kernel_set.execute(
            &self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            &self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            self.scalars_f32,
            self.scalars_i32,
            context,
            self.device.clone(),
        )
    }

    pub(crate) fn len(&self) -> usize {
        self.phase.operation_len
    }

    pub(crate) fn from_state(device: &WgpuDevice, mut state: ElementWiseState) -> Self {
        // The order is hardcoded from the list, not clear how to properly invalidate the cache
        // other than the burn version. TODO: Find a way to invalidate the cache.
        let vec4 = state.kernels.pop().unwrap();
        let vec2 = state.kernels.pop().unwrap();
        let scalar = state.kernels.pop().unwrap();

        let scalar =
            ScalarElementWise::new(Arc::new(FusedKernelSource::new(scalar.id, scalar.shader)));
        let vec2 = VecElementWise::<2>::new(Arc::new(FusedKernelSource::new(vec2.id, vec2.shader)));
        let vec4 = VecElementWise::<4>::new(Arc::new(FusedKernelSource::new(vec4.id, vec4.shader)));

        let kernel_set =
            FusionKernelSet::new(vec![Box::new(scalar), Box::new(vec2), Box::new(vec4)]);

        Self {
            inputs: state.inputs,
            outputs: state.outputs,
            scalars_f32: state.scalars_f32,
            scalars_u32: state.scalars_u32,
            scalars_i32: state.scalars_i32,
            device: device.clone(),
            phase: ExecutionPhase::new(state.operation_len, kernel_set),
        }
    }

    pub(crate) fn to_state(&self) -> ElementWiseState {
        ElementWiseState {
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            scalars_f32: self.scalars_f32,
            operation_len: self.phase.operation_len,
            scalars_u32: self.scalars_u32,
            scalars_i32: self.scalars_i32,
            kernels: self.phase.kernel_set.state(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_fusion::graph::Ops;
    use burn_fusion::{Fusion, FusionBackend};
    use burn_tensor::Int;
    use burn_tensor::{backend::Backend, Data, Tensor};

    #[test]
    fn test_fusion_same_behavior() {
        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let data_1 = Tensor::<FusedBackend, 2>::random(
            [1, 32],
            burn_tensor::Distribution::Default,
            &Default::default(),
        )
        .into_data();
        let data_2 = Tensor::<Backend, 2>::random(
            [32, 32],
            burn_tensor::Distribution::Default,
            &Default::default(),
        )
        .into_data();

        let result_ref = execute::<Backend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant1,
        );
        let result_fused = execute::<FusedBackend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant1,
        );

        result_ref.assert_approx_eq(&result_fused, 3);
    }

    #[test]
    fn test_fusion_same_behavior_int() {
        let data_1 = Tensor::<FusedBackend, 2>::random(
            [32, 32],
            burn_tensor::Distribution::Default,
            &Default::default(),
        )
        .into_data();
        let data_2 = Tensor::<Backend, 2>::random(
            [32, 32],
            burn_tensor::Distribution::Default,
            &Default::default(),
        )
        .into_data()
        .convert();

        fn func<B: burn_tensor::backend::Backend>(
            data1: Data<f32, 2>,
            data2: Data<i32, 2>,
        ) -> Data<f32, 2> {
            let x = Tensor::<B, 2>::from_data(data1.convert(), &Default::default());
            let y = Tensor::<B, 2, Int>::from_data(data2.convert(), &Default::default());

            let x_1 = x.clone().powf(2.0);
            let x_1 = x_1 + x;
            let y_1 = y * 6;
            let y_1 = y_1 + 4;

            let z = x_1 * y_1.float();

            z.into_data().convert()
        }

        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let result_fused = func::<FusedBackend>(data_1.clone(), data_2.clone());
        let result_ref = func::<Backend>(data_1.clone(), data_2.clone());

        result_ref.assert_approx_eq(&result_fused, 3);
    }

    #[test]
    fn test_fusion_same_behavior_different_variant() {
        type Backend = Wgpu;
        type FusedBackend = Fusion<Wgpu>;

        let data_1 = Tensor::<FusedBackend, 2>::random(
            [1, 32],
            burn_tensor::Distribution::Default,
            &Default::default(),
        )
        .into_data();
        let data_2 = Tensor::<Backend, 2>::random(
            [32, 32],
            burn_tensor::Distribution::Default,
            &Default::default(),
        )
        .into_data();

        let result_ref = execute::<Backend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant2,
        );
        let result_fused_variant1 = execute::<FusedBackend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant1,
        );
        let result_fused_variant2 = execute::<FusedBackend>(
            data_1.clone(),
            data_2.clone(),
            ImplementationDetails::Variant2,
        );

        result_ref.assert_approx_eq(&result_fused_variant1, 3);
        result_ref.assert_approx_eq(&result_fused_variant2, 3);
    }

    #[test]
    fn test_end_condition_scalar_ops() {
        type Backend = Fusion<Wgpu>;
        let device = Default::default();
        let tensor1 = Tensor::<Backend, 2>::ones([32, 32], &device);
        let tensor2 = Tensor::<Backend, 2>::ones([32, 42], &device);
        let output = tensor1.exp().log();

        // This will add a scalar to the context, even if the actual operation can't be fused with
        // the preceding ones because of the shape difference.
        let _ = tensor2 + 2;

        // When we try to execute the operations, the number of bindings can be different if we are
        // not careful.
        Backend::sync(&output.device());
    }

    struct FakeAddOps;

    impl<B: FusionBackend> Ops<B> for FakeAddOps {
        fn execute(self: Box<Self>, _: &mut burn_fusion::HandleContainer<B>) {
            panic!("Should always fused during tests.")
        }
    }

    enum ImplementationDetails {
        Variant1,
        Variant2,
    }

    fn execute<B: Backend>(
        data_1: Data<f32, 2>,
        data_2: Data<f32, 2>,
        variant: ImplementationDetails,
    ) -> Data<f32, 2> {
        let device = B::Device::default();
        let tensor_1 = Tensor::<B, 2>::from_data(data_1.convert(), &device);
        let tensor_2 = Tensor::<B, 2>::from_data(data_2.convert(), &device);
        let tensor_3 = tensor_1.clone() + tensor_2;
        let tensor_4 = tensor_3.clone() - tensor_1;
        let mut tensor_5 = tensor_4.clone() + 5.0;
        match variant {
            ImplementationDetails::Variant1 => {}
            ImplementationDetails::Variant2 => {
                tensor_5 = tensor_5 + 1;
                tensor_5 = tensor_5 - 1;
            }
        }
        let tensor_6 = burn_tensor::activation::gelu(tensor_5 + tensor_3.clone());
        let mask = tensor_4.lower_equal(tensor_3);
        let tmp = tensor_6.mask_fill(mask, 0.3);

        tmp.into_data().convert()
    }
}
