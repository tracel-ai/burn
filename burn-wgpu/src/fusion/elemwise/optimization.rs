use super::{
    kernel::{ScalarElementWise, VecElementWise},
    tune::ElementWiseAutotuneOperationSet,
    FusionElemWiseAutotuneKey,
};
use crate::{
    codegen::{
        Elem, ElemWiseKernelCodegen, InplaceMapping, Input, Item, Operator, Output,
        ReadingStrategy, Vectorization, Visibility, WorkgroupSize,
    },
    compute::{compute_client, WgpuAutotuneKey, WgpuComputeClient},
    fusion::{kernel::FusionKernelSet, source::DynKernelSource},
    FloatElement, GraphicsApi, IntElement, Wgpu, WgpuDevice,
};
use burn_common::id::IdGenerator;
use burn_fusion::{stream::Context, TensorDescription};
use burn_tensor::Device;
use serde::{Deserialize, Serialize};

#[derive(new)]
pub struct ElementWise<G, F, I, Phase = ExecutionPhase>
where
    G: GraphicsApi,
    F: FloatElement,
    I: IntElement,
{
    pub(super) inputs: Vec<(TensorDescription, Elem)>,
    pub(super) outputs: Vec<(TensorDescription, Elem)>,
    pub(super) locals: Vec<u16>,
    pub(super) scalars: Scalars,
    pub(super) operators: Vec<Operator>,
    pub(super) device: Device<Wgpu<G, F, I>>,
    pub(super) phase: Phase,
}

#[derive(new, Clone, Serialize, Deserialize)]
pub struct Scalars {
    pub(super) num_f32: usize,
    pub(super) num_u32: usize,
    pub(super) num_i32: usize,
}

pub struct CompilationPhase;

#[derive(new)]
pub struct ExecutionPhase {
    pub(super) kernel_set_1: FusionKernelSet,
    pub(super) kernel_set_2: FusionKernelSet,
}

#[derive(Serialize, Deserialize)]
pub struct ElementWiseState {
    inputs: Vec<(TensorDescription, Elem)>,
    outputs: Vec<(TensorDescription, Elem)>,
    scalars: Scalars,
    operators: Vec<Operator>,
    locals: Vec<u16>,
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
            .zip(self.locals.iter())
            .map(|((_tensor, elem), local)| Output::Array {
                item: Item::Scalar(*elem),
                local: *local,
            })
            .collect::<Vec<_>>();

        if self.scalars.num_f32 > 0 {
            inputs.push(Input::Scalar {
                elem: Elem::F32,
                size: self.scalars.num_f32,
            })
        }

        if self.scalars.num_u32 > 0 {
            inputs.push(Input::Scalar {
                elem: Elem::U32,
                size: self.scalars.num_u32,
            })
        }

        if self.scalars.num_i32 > 0 {
            inputs.push(Input::Scalar {
                elem: Elem::I32,
                size: self.scalars.num_i32,
            })
        }

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

        let mappings = self
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

        let kernel_set_1 = build_kernel_set(
            &inputs,
            &outputs,
            &self.operators,
            &mappings,
            WorkgroupSize::default(),
        );
        let kernel_set_2 = build_kernel_set(
            &inputs,
            &outputs,
            &self.operators,
            &mappings,
            WorkgroupSize::new(16, 16, 1),
        );

        ElementWise {
            inputs: self.inputs,
            outputs: self.outputs,
            scalars: self.scalars,
            device: self.device,
            operators: self.operators,
            locals: self.locals,
            phase: ExecutionPhase::new(kernel_set_1, kernel_set_2),
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
        let client = compute_client::<G>(&self.device);

        let key = WgpuAutotuneKey::FusionElemWise(FusionElemWiseAutotuneKey::new(
            self.operators.len(),
            self.autotune_shape(context),
        ));

        if let Some(index) = client.autotune_result(&key) {
            self.run_kernel(context, client, index)
        } else {
            self.run_autotune(context, client, key)
        }
    }

    fn run_kernel(
        &mut self,
        context: &mut Context<'_, Wgpu<G, F, I>>,
        client: WgpuComputeClient,
        fastest_set_index: usize,
    ) {
        let kernel_set = match fastest_set_index {
            0 => &self.phase.kernel_set_1,
            1 => &self.phase.kernel_set_2,
            _ => panic!("Should be 0 or 1, got {fastest_set_index}"),
        };

        let kernel = kernel_set.select(
            &self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            &self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            self.scalars.num_f32,
            self.scalars.num_i32,
            context,
            self.device.clone(),
            client,
            true,
        );

        kernel.execute();
    }

    fn run_autotune(
        &mut self,
        context: &mut Context<'_, Wgpu<G, F, I>>,
        client: WgpuComputeClient,
        key: WgpuAutotuneKey,
    ) {
        let kernel_1 = self.phase.kernel_set_1.select(
            &self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            &self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            self.scalars.num_f32,
            self.scalars.num_i32,
            context,
            self.device.clone(),
            client.clone(),
            false, // Should not mutate the context.
        );
        let kernel_2 = self.phase.kernel_set_1.select(
            &self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            &self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            self.scalars.num_f32,
            self.scalars.num_i32,
            context,
            self.device.clone(),
            client.clone(),
            false, // Should not mutate the context.
        );
        let kernel_default = self.phase.kernel_set_1.select(
            &self.inputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            &self.outputs.iter().map(|a| &a.0).collect::<Vec<_>>(),
            self.scalars.num_f32,
            self.scalars.num_i32,
            context,
            self.device.clone(),
            client.clone(),
            true, // Can do whatever with the context.
        );

        client.autotune_execute(Box::new(ElementWiseAutotuneOperationSet::new(
            key,
            kernel_1.into(),
            kernel_2.into(),
            kernel_default.into(),
        )));
    }

    pub(crate) fn len(&self) -> usize {
        self.operators.len()
    }

    /// The first output is chosen when possible, otherwise the first input is chosen.
    pub(crate) fn autotune_shape<'a>(
        &self,
        context: &mut Context<'a, Wgpu<G, F, I>>,
    ) -> &'a [usize] {
        if let Some(tensor) = self.outputs.first() {
            let tensor = context.tensors.get(&tensor.0.id).unwrap();
            return &tensor.shape;
        }

        if let Some(tensor) = self.inputs.first() {
            let tensor = context.tensors.get(&tensor.0.id).unwrap();
            return &tensor.shape;
        }

        &[]
    }

    pub(crate) fn from_state(device: &WgpuDevice, state: ElementWiseState) -> Self {
        // We don't save the compiled kernel structs since it's quick to compile and the output is
        // very large.
        //
        // It is still unclear if the deserialization would be that much faster than
        // simply recompiling it.
        ElementWise {
            inputs: state.inputs,
            outputs: state.outputs,
            scalars: state.scalars,
            device: device.clone(),
            locals: state.locals,
            operators: state.operators,
            phase: CompilationPhase,
        }
        .compile()
    }

    pub(crate) fn to_state(&self) -> ElementWiseState {
        ElementWiseState {
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            scalars: self.scalars.clone(),
            operators: self.operators.clone(),
            locals: self.locals.clone(),
        }
    }
}

fn build_kernel_set(
    inputs: &[Input],
    outputs: &[Output],
    operators: &[Operator],
    mappings: &[InplaceMapping],
    workgroup_size: WorkgroupSize,
) -> FusionKernelSet {
    let scalar = ScalarElementWise::new(
        DynKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .inputs(inputs)
                .body(operators)
                .outputs(outputs)
                .workgroup_size(workgroup_size)
                .compile(),
        ),
        DynKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .inplace(mappings)
                .inputs(inputs)
                .body(operators)
                .outputs(outputs)
                .workgroup_size(workgroup_size)
                .compile(),
        ),
        mappings.to_vec(),
        outputs.len(),
    );

    let vec2 = VecElementWise::new(
        DynKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .vectorize(Vectorization::Vec2)
                .inputs(inputs)
                .body(operators)
                .outputs(outputs)
                .workgroup_size(workgroup_size)
                .compile(),
        ),
        DynKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .vectorize(Vectorization::Vec2)
                .inplace(mappings)
                .inputs(inputs)
                .body(operators)
                .outputs(outputs)
                .workgroup_size(workgroup_size)
                .compile(),
        ),
        mappings.to_vec(),
        outputs.len(),
        2,
    );
    let vec4 = VecElementWise::new(
        DynKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .vectorize(Vectorization::Vec4)
                .inputs(inputs)
                .body(operators)
                .outputs(outputs)
                .workgroup_size(workgroup_size)
                .compile(),
        ),
        DynKernelSource::new(
            IdGenerator::generate(),
            ElemWiseKernelCodegen::new()
                .vectorize(Vectorization::Vec4)
                .inplace(mappings)
                .inputs(inputs)
                .body(operators)
                .outputs(outputs)
                .workgroup_size(workgroup_size)
                .compile(),
        ),
        mappings.to_vec(),
        outputs.len(),
        4,
    );

    FusionKernelSet::new(vec![Box::new(scalar), Box::new(vec2), Box::new(vec4)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_fusion::stream::Operation;
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

            let x_1 = x.clone().powf_scalar(2.0);
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

    impl<B: FusionBackend> Operation<B> for FakeAddOps {
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
