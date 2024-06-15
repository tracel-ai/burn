use crate::{
    element::JitElement,
    kernel::{self, Kernel},
    tensor::JitTensor,
    JitRuntime,
};
use burn_cube::{
    calculate_cube_count_elemwise, cpa, frontend::TensorHandle, CubeCountSettings, KernelExpansion,
    KernelIntegrator, KernelSettings,
};
use burn_cube::{
    ir::{Branch, Elem, IntKind, Item, KernelDefinition, Scope, Variable, Visibility},
    Execution,
};
use burn_cube::{InputInfo, SUBCUBE_DIM_APPROX};
use std::marker::PhantomData;

#[derive(new)]
struct ScatterEagerKernel<R: JitRuntime, E: JitElement> {
    dim: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct ScatterComputeShader {
    input: Variable,
    indices: Variable,
    value: Variable,
    dim: usize,
}

impl ScatterComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        match self.input {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Input variable must be an global array."),
        };
        match self.value {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Value variable must be an global array."),
        };

        let input = self.input;
        let value = self.value;
        let indices = self.indices;

        let stride_input = scope.create_local(Elem::UInt);
        let shape_value = scope.create_local(Elem::UInt);

        cpa!(scope, stride_input = stride(input, self.dim));
        cpa!(scope, shape_value = shape(value, self.dim));

        let id = Variable::AbsolutePos;
        let offset_input = scope.zero(Elem::UInt);
        let offset_value = scope.zero(Elem::UInt);

        let num_elems = scope.create_local(Elem::UInt);
        cpa!(scope, num_elems = cast(1usize));
        cpa!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let should_skip = scope.create_local(Elem::Bool);
                cpa!(scope, should_skip = i == self.dim);

                cpa!(scope, if(should_skip).then(|_| {
                    // Nothing to do.
                }).else(|scope| {
                    let shape_input_loop = scope.create_local(Elem::UInt);
                    let shape_value_loop = scope.create_local(Elem::UInt);
                    let stride_value_loop = scope.create_local(Elem::UInt);

                    let stride_tmp = scope.create_local(Elem::UInt);
                    let num_blocks = scope.create_local(Elem::UInt);
                    let offset_tmp = scope.create_local(Elem::UInt);
                    let stride_input_loop = scope.create_local(Elem::UInt);

                    cpa!(scope, stride_value_loop = stride(value, i));
                    cpa!(scope, stride_input_loop = stride(input, i));
                    cpa!(scope, stride_tmp = stride(indices, i));

                    cpa!(scope, shape_value_loop = shape(value, i));
                    cpa!(scope, shape_input_loop = shape(input, i));

                    cpa!(scope, num_blocks = id / stride_tmp);
                    cpa!(scope, num_blocks = num_blocks % shape_input_loop);

                    cpa!(scope, offset_tmp = num_blocks * stride_input_loop);
                    cpa!(scope, offset_input += offset_tmp);

                    cpa!(scope, offset_tmp = num_blocks * stride_value_loop);
                    cpa!(scope, offset_value += offset_tmp);

                    cpa!(scope, num_elems = num_elems * shape_value_loop);
                }));
            })
        );

        let should_stop = scope.create_local(Elem::Bool);
        cpa!(scope, should_stop = id >= num_elems);
        cpa!(scope, if (should_stop).then(|scope|{
            scope.register(Branch::Return);
        }));

        let index_input = scope.create_local(Elem::UInt);
        let index = scope.create_local(Elem::UInt);

        let result_input = scope.create_local(input.item());
        let result_value = scope.create_local(value.item());
        let result_indices = scope.create_local(Elem::UInt);

        cpa!(
            scope,
            range(0u32, shape_value).for_each(|i, scope| {
                cpa!(scope, index = stride_input * i);
                cpa!(scope, index += offset_value);

                cpa!(scope, result_value = value[index]);
                cpa!(scope, result_indices = indices[index]);

                cpa!(scope, index_input = stride_input * result_indices);
                cpa!(scope, index_input += offset_input);

                cpa!(scope, result_input = input[index_input]);
                cpa!(scope, result_input += result_value);
                cpa!(scope, input[index_input] = result_input);
            })
        );
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for ScatterEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        let mut scope = Scope::root();
        let item_value = E::cube_elem().into();
        let item_indices: Item = Elem::Int(IntKind::I32).into();

        let input_output = Variable::GlobalInputArray(0, item_value);
        let indices = Variable::GlobalInputArray(1, Elem::Int(IntKind::I32).into());
        let value = Variable::GlobalInputArray(2, item_value);

        scope.write_global_custom(input_output);

        ScatterComputeShader {
            input: input_output,
            indices,
            value,
            dim: self.dim,
        }
        .expand(&mut scope);

        let input_output = InputInfo::Array {
            item: item_value,
            visibility: Visibility::ReadWrite,
        };
        let indices = InputInfo::Array {
            item: item_indices,
            visibility: Visibility::Read,
        };
        let value = InputInfo::Array {
            item: item_value,
            visibility: Visibility::Read,
        };

        let info = KernelExpansion {
            inputs: vec![input_output, indices, value],
            outputs: vec![],
            scope,
        };

        let settings = KernelSettings::default();
        KernelIntegrator::new(info).integrate(settings)
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

pub(crate) fn scatter<R: JitRuntime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let mut indices = kernel::into_contiguous(indices);
    let tensor = kernel::into_contiguous(tensor);
    let value = kernel::into_contiguous(value);

    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let kernel = ScatterEagerKernel::<R, E>::new(dim);
    let mut strides = [0; D];
    let mut current = 1;
    let mut num_elems_per_workgroup = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            strides[index] = current;
            current *= val;
            num_elems_per_workgroup *= tensor.shape.dims[index];
        });

    // Fake strides of the virtual output where the strides of dim is hardcoded to one.
    indices.strides = strides;

    let workgroup = calculate_cube_count_elemwise(num_elems_per_workgroup, SUBCUBE_DIM_APPROX);

    Execution::start(kernel, indices.client)
        .inputs(&[
            TensorHandle::<R>::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            TensorHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
            TensorHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ])
        .execute(CubeCountSettings::Custom(workgroup));

    tensor
}
