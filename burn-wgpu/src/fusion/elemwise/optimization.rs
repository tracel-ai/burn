use super::{
    kernel::{ScalarElementWise, VecElementWise},
    tune::ElementWiseAutotuneOperationSet,
    FusionElemWiseAutotuneKey,
};
use crate::{
    codegen::{
        dialect::gpu::{Vectorization, WorkgroupSize},
        Compilation, CompilationInfo, CompilationSettings,
    },
    compute::JitAutotuneKey,
    fusion::{kernel::FusionKernelSet, source::GpuKernelSource, tracing::Trace},
    JitBackend, Runtime,
};
use burn_common::id::IdGenerator;
use burn_compute::client::ComputeClient;
use burn_fusion::stream::Context;
use serde::{Deserialize, Serialize};

#[derive(new)]
pub struct ElementWise<R: Runtime, Phase = ExecutionPhase<R>> {
    pub(super) trace: Trace,
    pub(super) num_operations: usize,
    pub(super) device: R::Device,
    pub(super) phase: Phase,
}

pub struct CompilationPhase;

#[derive(new)]
pub struct ExecutionPhase<R: Runtime> {
    pub(super) kernel_set_1: FusionKernelSet<R>,
    pub(super) kernel_set_2: FusionKernelSet<R>,
}

#[derive(new, Serialize, Deserialize)]
pub struct ElementWiseState {
    trace: Trace,
    num_operations: usize,
}

impl<R: Runtime> ElementWise<R, CompilationPhase> {
    pub(crate) fn compile(self) -> ElementWise<R, ExecutionPhase<R>> {
        let info = self.trace.compiling();

        let kernel_set_1 = build_kernel_set::<R>(&info, WorkgroupSize::default());
        let kernel_set_2 = build_kernel_set::<R>(&info, WorkgroupSize::new(16, 16, 1));

        ElementWise {
            trace: self.trace,
            device: self.device,
            phase: ExecutionPhase::new(kernel_set_1, kernel_set_2),
            num_operations: self.num_operations,
        }
    }
}

impl<R: Runtime> ElementWise<R, ExecutionPhase<R>> {
    pub(crate) fn execute(&mut self, context: &mut Context<'_, JitBackend<R>>) {
        let client = R::client(&self.device);

        let key = JitAutotuneKey::FusionElemWise(FusionElemWiseAutotuneKey::new(
            self.num_operations,
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
        context: &mut Context<'_, JitBackend<R>>,
        client: ComputeClient<R::Server, R::Channel>,
        fastest_set_index: usize,
    ) {
        let info = self.trace.running();
        let kernel_set = match fastest_set_index {
            0 => &self.phase.kernel_set_1,
            1 => &self.phase.kernel_set_2,
            _ => panic!("Should be 0 or 1, got {fastest_set_index}"),
        };

        let kernel = kernel_set.select(&info, context, self.device.clone(), client, true);

        kernel.execute();
    }

    fn run_autotune(
        &mut self,
        context: &mut Context<'_, JitBackend<R>>,
        client: ComputeClient<R::Server, R::Channel>,
        key: JitAutotuneKey,
    ) {
        let info = self.trace.running();

        let kernel_1 = self.phase.kernel_set_1.select(
            &info,
            context,
            self.device.clone(),
            client.clone(),
            false, // Should not mutate the context.
        );
        let kernel_2 = self.phase.kernel_set_1.select(
            &info,
            context,
            self.device.clone(),
            client.clone(),
            false, // Should not mutate the context.
        );
        let kernel_default = self.phase.kernel_set_1.select(
            &info,
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
        self.num_operations
    }

    /// The first output is chosen when possible, otherwise the first input is chosen.
    pub(crate) fn autotune_shape<'a>(
        &self,
        context: &mut Context<'a, JitBackend<R>>,
    ) -> &'a [usize] {
        let info = self.trace.running();

        if let Some(tensor) = info.outputs.first() {
            let tensor = context.tensors.get(&tensor.id).unwrap();
            return &tensor.shape;
        }

        if let Some(tensor) = info.inputs.first() {
            let tensor = context.tensors.get(&tensor.id).unwrap();
            return &tensor.shape;
        }

        &[]
    }

    pub(crate) fn from_state(device: &R::Device, state: ElementWiseState) -> Self {
        // We don't save the compiled kernel structs since it's quick to compile and the output is
        // very large.
        //
        // It is still unclear if the deserialization would be that much faster than
        // simply recompiling it.
        ElementWise {
            trace: state.trace,
            device: device.clone(),
            phase: CompilationPhase,
            num_operations: state.num_operations,
        }
        .compile()
    }

    pub(crate) fn to_state(&self) -> ElementWiseState {
        ElementWiseState {
            trace: self.trace.clone(),
            num_operations: self.num_operations,
        }
    }
}

fn build_kernel_set<R: Runtime>(
    info: &CompilationInfo,
    workgroup_size: WorkgroupSize,
) -> FusionKernelSet<R> {
    let scalar = ScalarElementWise::<R>::new(
        GpuKernelSource::new(
            IdGenerator::generate(),
            Compilation::new(info.clone())
                .compile(CompilationSettings::default().workgroup_size(workgroup_size)),
        ),
        GpuKernelSource::new(
            IdGenerator::generate(),
            Compilation::new(info.clone()).compile(
                CompilationSettings::default()
                    .inplace(true)
                    .workgroup_size(workgroup_size),
            ),
        ),
        info.mappings.to_vec(),
        info.outputs.len(),
    );

    let vec2 = VecElementWise::<R>::new(
        GpuKernelSource::new(
            IdGenerator::generate(),
            Compilation::new(info.clone()).compile(
                CompilationSettings::default()
                    .vectorize(Vectorization::Vec2)
                    .workgroup_size(workgroup_size),
            ),
        ),
        GpuKernelSource::new(
            IdGenerator::generate(),
            Compilation::new(info.clone()).compile(
                CompilationSettings::default()
                    .inplace(true)
                    .vectorize(Vectorization::Vec2)
                    .workgroup_size(workgroup_size),
            ),
        ),
        info.mappings.to_vec(),
        info.outputs.len(),
        2,
    );
    let vec4 = VecElementWise::<R>::new(
        GpuKernelSource::new(
            IdGenerator::generate(),
            Compilation::new(info.clone()).compile(
                CompilationSettings::default()
                    .vectorize(Vectorization::Vec4)
                    .workgroup_size(workgroup_size),
            ),
        ),
        GpuKernelSource::new(
            IdGenerator::generate(),
            Compilation::new(info.clone()).compile(
                CompilationSettings::default()
                    .inplace(true)
                    .vectorize(Vectorization::Vec4)
                    .workgroup_size(workgroup_size),
            ),
        ),
        info.mappings.to_vec(),
        info.outputs.len(),
        4,
    );

    FusionKernelSet::new(vec![Box::new(scalar), Box::new(vec2), Box::new(vec4)])
}

#[cfg(test)]
mod tests {
    use crate::tests::TestRuntime;
    use crate::JitBackend;

    use burn_fusion::stream::Operation;
    use burn_fusion::{Fusion, FusionBackend};
    use burn_tensor::Int;
    use burn_tensor::{backend::Backend, Data, Tensor};

    #[test]
    fn test_fusion_same_behavior() {
        type Backend = JitBackend<TestRuntime>;
        type FusedBackend = Fusion<Backend>;

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

        type Backend = JitBackend<TestRuntime>;
        type FusedBackend = Fusion<Backend>;

        let result_fused = func::<FusedBackend>(data_1.clone(), data_2.clone());
        let result_ref = func::<Backend>(data_1.clone(), data_2.clone());

        result_ref.assert_approx_eq(&result_fused, 3);
    }

    #[test]
    fn test_fusion_same_behavior_different_variant() {
        type Backend = JitBackend<TestRuntime>;
        type FusedBackend = Fusion<Backend>;

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
        type Backend = JitBackend<TestRuntime>;
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
