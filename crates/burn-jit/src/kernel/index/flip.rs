use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{DynamicKernelSource, SourceTemplate},
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime, RuntimeInt,
};
use burn_tensor::ElementConversion;
use std::marker::PhantomData;

#[derive(new)]
struct FlipEagerKernel<R: Runtime, E: JitElement> {
    rank: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

pub struct FlipComputeShader {
    input: Variable,
    output: Variable,
    rank: usize,
}

impl FlipComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::Id;

        let offset_input = scope.zero(Elem::UInt);
        let offset_local = scope.create_local(Elem::UInt);

        let stride = scope.create_local(Elem::UInt);
        let shape = scope.create_local(Elem::UInt);
        let flip = scope.create_local(Elem::UInt);
        let flip_bool = scope.create_local(Elem::Bool);

        for i in 0..self.rank {
            gpu!(scope, stride = stride(input, i));
            gpu!(scope, shape = shape(output, i));
            gpu!(
                scope,
                flip = cast(Variable::GlobalScalar(i as u16, Elem::UInt))
            );
            gpu!(scope, flip_bool = flip == 1u32);

            gpu!(scope, offset_local = id / stride);
            gpu!(scope, offset_local = offset_local % shape);

            gpu!(scope, if(flip_bool).then(|scope| {
                gpu!(scope, offset_local = shape - offset_local);
                gpu!(scope, offset_local = offset_local - 1u32);
            }));
            gpu!(scope, offset_local = offset_local * stride);

            gpu!(scope, offset_input += offset_local);
        }

        let result = scope.create_local(input.item());
        gpu!(scope, result = input[offset_input]);
        gpu!(scope, output[id] = result);
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for FlipEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();

        let input = Variable::GlobalInputArray(0, item);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        FlipComputeShader {
            input,
            output,
            rank: self.rank,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let flip_dims = InputInfo::Scalar {
            elem: Elem::UInt,
            size: self.rank,
        };
        let output = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input, flip_dims],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}-rank={:?}", core::any::TypeId::of::<Self>(), self.rank)
    }
}

pub(crate) fn flip<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    indices: &[usize],
) -> JitTensor<R, E, D> {
    let output = empty_device(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );
    flip_on_output(tensor, output, indices)
}

pub(crate) fn flip_on_output<R: Runtime, E: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    output: JitTensor<R, E, D>,
    indices: &[usize],
) -> JitTensor<R, E, D> {
    let mut scalars = Vec::with_capacity(D);

    for i in 0..D {
        scalars.push((indices.contains(&i) as u32).elem());
    }

    let kernel = FlipEagerKernel::new(D);

    execute_dynamic::<R, FlipEagerKernel<R, E>, RuntimeInt<R>>(
        &[EagerHandle::new(
            &tensor.handle,
            &tensor.strides,
            &tensor.shape.dims,
        )],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        Some(&scalars),
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        tensor.client,
    );

    output
}
