use crate::{element::JitElement, kernel::Kernel, tensor::JitTensor, JitRuntime};
use cubecl::{
    cpa,
    ir::{Elem, KernelDefinition, Scope, Variable, Visibility},
    CubeCountSettings, Execution, InputInfo, KernelExpansion, KernelIntegrator, KernelSettings,
    OutputInfo,
};
use std::marker::PhantomData;

pub struct RepeatComputeShader {
    input: Variable,
    output: Variable,
    dim: usize,
    rank: usize,
}

#[derive(new)]
struct RepeatEagerKernel<R: JitRuntime, E: JitElement> {
    dim: usize,
    rank: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl RepeatComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let output = self.output;
        let id = Variable::AbsolutePos;

        let offset_input = scope.zero(Elem::UInt);
        let offset_local = scope.zero(Elem::UInt);

        let stride_input = scope.create_local(Elem::UInt);
        let stride_output = scope.create_local(Elem::UInt);
        let shape = scope.create_local(Elem::UInt);

        for i in 0..self.rank {
            cpa!(scope, stride_input = stride(input, i));
            cpa!(scope, stride_output = stride(output, i));
            if i != self.dim {
                cpa!(scope, shape = shape(output, i));
            } else {
                cpa!(scope, shape = shape(input, i));
            }

            cpa!(scope, offset_local = id / stride_output);
            cpa!(scope, offset_local = offset_local % shape);
            cpa!(scope, offset_local = offset_local * stride_input);
            cpa!(scope, offset_input += offset_local);
        }

        let result = scope.create_local(input.item());
        cpa!(scope, result = input[offset_input]);
        cpa!(scope, output[id] = result);
    }
}
impl<R: JitRuntime, E: JitElement> Kernel for RepeatEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();

        let input = Variable::GlobalInputArray { id: 0, item };
        let output = Variable::GlobalOutputArray { id: 0, item };

        scope.write_global_custom(output);

        RepeatComputeShader {
            input,
            output,
            rank: self.rank,
            dim: self.dim,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let output = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![input],
            outputs: vec![output],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> cubecl::KernelId {
        cubecl::KernelId::new::<Self>().info((self.dim, self.rank))
    }
}

pub(crate) fn repeat_dim<R: JitRuntime, E: JitElement>(
    input: JitTensor<R, E>,
    dim: usize,
    times: usize,
) -> JitTensor<R, E> {
    let mut shape = input.shape.clone();
    let ndims = shape.num_dims();

    // Create output handle
    shape.dims[dim] *= times;
    let num_elems_output = shape.num_elements();
    let handle = input
        .client
        .empty(num_elems_output * core::mem::size_of::<E>());
    let output = JitTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape.clone(),
        handle,
    );

    let kernel = RepeatEagerKernel::<R, E>::new(dim, ndims);

    Execution::start(kernel, input.client.clone())
        .inputs(&[input.as_handle_ref()])
        .outputs(&[output.as_handle_ref()])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}
