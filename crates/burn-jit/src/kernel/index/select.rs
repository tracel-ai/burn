use crate::{
    element::JitElement, kernel::GpuComputeShaderPhase, ops::numeric::empty_device,
    tensor::JitTensor, JitRuntime,
};
use burn_cube::{
    cpa,
    dialect::{ComputeShader, Elem, IntKind, Item, Scope, Variable, Visibility},
    Compilation, CompilationInfo, CompilationSettings, Execution, InputInfo, OutputInfo,
    TensorHandle, WorkgroupLaunch,
};
use std::marker::PhantomData;

#[derive(new)]
struct SelectEagerKernel<R: JitRuntime, E: JitElement> {
    dim: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

pub struct SelectComputeShader {
    input: Variable,
    indices: Variable,
    output: Variable,
    dim: usize,
}

impl SelectComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        let input = self.input;
        let indices = self.indices;
        let output = self.output;
        let id = Variable::Id;
        let offset_input = scope.zero(Elem::UInt);

        cpa!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                cpa!(scope, stride_input = stride(input, i));
                cpa!(scope, stride_output = stride(output, i));
                cpa!(scope, shape_output = shape(output, i));

                let offset_local = scope.create_local(Elem::UInt);
                cpa!(scope, offset_local = id / stride_output);
                cpa!(scope, offset_local = offset_local % shape_output);

                let dim_index = scope.create_local(Elem::Bool);
                cpa!(scope, dim_index = i == self.dim);

                cpa!(scope, if(dim_index).then(|scope| {
                    cpa!(scope, offset_local = indices[offset_local]);
                    cpa!(scope, offset_local = offset_local * stride_input);
                }).else(|scope| {
                    cpa!(scope, offset_local = offset_local * stride_input);
                }));

                cpa!(scope, offset_input += offset_local);
            })
        );

        let value = scope.create_local(input.item());
        cpa!(scope, value = input[offset_input]);
        cpa!(scope, output[id] = value);
    }
}

impl<R: JitRuntime, E: JitElement> GpuComputeShaderPhase for SelectEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
        let mut scope = Scope::root();
        let item = E::cube_elem().into();
        let item_indices: Item = Elem::Int(IntKind::I32).into();

        let input = Variable::GlobalInputArray(0, item);
        let indices = Variable::GlobalInputArray(1, item_indices);
        let output = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(output);

        SelectComputeShader {
            input,
            indices,
            output,
            dim: self.dim,
        }
        .expand(&mut scope);

        let input = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let indices = InputInfo::Array {
            item: item_indices,
            visibility: Visibility::Read,
        };
        let output = OutputInfo::Array { item };

        let info = CompilationInfo {
            inputs: vec![input, indices],
            outputs: vec![output],
            scope,
        };

        let settings = CompilationSettings::default();
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

pub(crate) fn select<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
) -> JitTensor<R, E, D> {
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape.dims[0];

    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);
    let kernel = SelectEagerKernel::<R, E>::new(dim);

    Execution::start(kernel, tensor.client)
        .inputs(&[
            TensorHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            // This is a current hacks because the info buffer that contains the strides and shapes is
            // hardcoded to only contains information about tensors of the same rank. However, since
            // we don't rely on the shape and stride of the indices tensors, it doesn't matter
            // which value we put, it just needs to be of the same rank.
            TensorHandle::new(&indices.handle, &[1; D], &[1; D]),
        ])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Output { pos: 0 });

    output
}
