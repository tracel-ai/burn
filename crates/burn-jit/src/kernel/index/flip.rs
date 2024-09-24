use crate::{
    element::JitElement, kernel::Kernel, ops::numeric::empty_device, tensor::JitTensor, JitRuntime,
};
use burn_tensor::ElementConversion;
use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

#[derive(new)]
struct FlipEagerKernel<R: JitRuntime, E: JitElement> {
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
        let id = Variable::AbsolutePos;

        let offset_input = scope.zero(Elem::UInt);
        let offset_local = scope.create_local(Elem::UInt);

        let stride = scope.create_local(Elem::UInt);
        let shape = scope.create_local(Elem::UInt);
        let flip = scope.create_local(Elem::UInt);
        let flip_bool = scope.create_local(Elem::Bool);

        for i in 0..self.rank {
            cpa!(scope, stride = stride(input, i));
            cpa!(scope, shape = shape(output, i));
            cpa!(
                scope,
                flip = cast(Variable::GlobalScalar {
                    id: i as u16,
                    elem: Elem::UInt
                })
            );
            cpa!(scope, flip_bool = flip == 1u32);

            cpa!(scope, offset_local = id / stride);
            cpa!(scope, offset_local = offset_local % shape);

            cpa!(scope, if(flip_bool).then(|scope| {
                cpa!(scope, offset_local = shape - offset_local);
                cpa!(scope, offset_local = offset_local - 1u32);
            }));
            cpa!(scope, offset_local = offset_local * stride);

            cpa!(scope, offset_input += offset_local);
        }

        let result = scope.create_local(input.item());
        cpa!(scope, result = input[offset_input]);
        cpa!(scope, output[id] = result);
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for FlipEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let input = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

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

        let info = KernelExpansion {
            inputs: vec![input, flip_dims],
            outputs: vec![output],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info(self.rank)
    }
}

pub(crate) fn flip<R: JitRuntime, E: JitElement>(
    tensor: JitTensor<R, E>,
    indices: &[usize],
) -> JitTensor<R, E> {
    let output = empty_device(
        tensor.client.clone(),
        tensor.device.clone(),
        tensor.shape.clone(),
    );
    flip_on_output(tensor, output, indices)
}

pub(crate) fn flip_on_output<R: JitRuntime, E: JitElement>(
    tensor: JitTensor<R, E>,
    output: JitTensor<R, E>,
    indices: &[usize],
) -> JitTensor<R, E> {
    let ndims = tensor.shape.num_dims();
    let mut scalars: Vec<u32> = Vec::with_capacity(ndims);

    for i in 0..ndims {
        scalars.push((indices.contains(&i) as u32).elem());
    }

    let kernel = FlipEagerKernel::<R, E>::new(ndims);

    Execution::start(kernel, tensor.client.clone())
        .inputs(&[tensor.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .with_scalars(&scalars)
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
