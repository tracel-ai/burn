use crate::codegen::dialect::gpu::{gpu, Branch, Elem, Scope, Variable};
use crate::kernel::{elemwise_workgroup, WORKGROUP_DEFAULT};
use crate::{
    codegen::{
        dialect::gpu, execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler,
        EagerHandle, InputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{self, into_contiguous, DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

#[derive(new)]
struct ScatterEagerKernel<R: Runtime, E: JitElement> {
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
        let stride_value = scope.create_local(Elem::UInt);
        let stride_indices = scope.create_local(Elem::UInt);

        let shape_input = scope.zero(Elem::UInt);
        let shape_value = scope.create_local(Elem::UInt);
        let shape_indices = scope.create_local(Elem::UInt);

        let offset_input = scope.zero(Elem::UInt);
        let offset_value = scope.zero(Elem::UInt);
        let offset_indices = scope.zero(Elem::UInt);

        gpu!(scope, stride_input = stride(input, self.dim));
        gpu!(scope, stride_value = stride(value, self.dim));
        gpu!(scope, stride_indices = stride(indices, self.dim));

        gpu!(scope, shape_input = shape(input, self.dim));
        gpu!(scope, shape_value = shape(value, self.dim));
        gpu!(scope, shape_indices = shape(indices, self.dim));

        let id = Variable::Id;

        let should_stop = scope.create_local(Elem::Bool);
        let num_elems = scope.create_local(Elem::UInt);
        let array_size = scope.create_local(Elem::UInt);

        gpu!(scope, array_size = len(value));
        gpu!(scope, num_elems = array_size / shape_value);
        gpu!(scope, should_stop = id >= num_elems);
        gpu!(scope, if (should_stop).then(|scope|{
            scope.register(Branch::Return);
        }));

        let index_loop = scope.zero(Elem::UInt);
        gpu!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let should_skip = scope.create_local(Elem::Bool);
                gpu!(scope, should_skip = i == self.dim);
                let one: Variable = 0u32.into();

                gpu!(scope, if(should_skip).then(|_| {
                    // Nothing to do.
                }).else(|scope| {
                    let shape = scope.create_local(Elem::UInt);
                    let stride = scope.create_local(Elem::UInt);
                    let stride_layout = scope.create_local(Elem::UInt);
                    let num_blocks = scope.create_local(Elem::UInt);

                    gpu!(scope, index_loop += one);
                    gpu!(scope, stride_layout = stride(input, index_loop));
                    gpu!(scope, stride = stride(input, i));
                    gpu!(scope, shape = shape(input, i));

                    gpu!(scope, num_blocks = id / stride_layout);
                    gpu!(scope, num_blocks = num_blocks % shape);
                    gpu!(scope, num_blocks = num_blocks * stride_input);
                    gpu!(scope, offset_input += num_blocks);
                }));
            })
        );

        let index_input = scope.create_local(Elem::UInt);
        let index_value = scope.create_local(Elem::UInt);
        let index_indices = scope.create_local(Elem::UInt);

        let value_input = scope.create_local(Elem::UInt);
        let value_value = scope.create_local(Elem::UInt);
        let value_indices = scope.create_local(Elem::UInt);

        gpu!(
            scope,
            range(0u32, shape_value).for_each(|i, scope| {
                gpu!(scope, index_value = stride_value * i);
                gpu!(scope, index_value += offset_value);

                gpu!(scope, index_indices = stride_indices * i);
                gpu!(scope, index_indices += offset_indices);

                gpu!(scope, value_value = value[index_value]);
                gpu!(scope, value_indices = indices[index_indices]);

                gpu!(scope, index_input = stride_input * value_indices);
                gpu!(scope, index_input += offset_input);

                gpu!(scope, value_input = input[index_input]);
                gpu!(scope, value_input += value_value);
                gpu!(scope, input[offset_input] = index_indices);
            })
        );
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for ScatterEagerKernel<R, E> {
    fn source(&self) -> kernel::SourceTemplate {
        let mut scope = gpu::Scope::root();
        let item_value = E::gpu_elem().into();
        let item_indices: gpu::Item = gpu::Elem::Int.into();

        let input_output = gpu::Variable::GlobalInputArray(0, item_value);
        let indices = gpu::Variable::GlobalInputArray(1, Elem::Int.into());
        let value = gpu::Variable::GlobalInputArray(2, item_value);

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
            visibility: gpu::Visibility::ReadWrite,
        };
        let indices = InputInfo::Array {
            item: item_indices,
            visibility: gpu::Visibility::Read,
        };
        let value = InputInfo::Array {
            item: item_value,
            visibility: gpu::Visibility::Read,
        };

        let info = CompilationInfo {
            inputs: vec![input_output, indices, value],
            outputs: vec![],
            scope,
            mappings: vec![],
        };

        let settings = CompilationSettings::default();
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

pub(crate) fn scatter<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let indices = into_contiguous(indices);
    let value = into_contiguous(value);
    let tensor = into_contiguous(tensor);

    let tensor = if !tensor.can_mut() {
        tensor.copy()
    } else {
        tensor
    };
    let kernel = ScatterEagerKernel::new(dim);
    let mut current = 1;
    let mut strides = [0; D];
    let mut num_elems_per_workgroup = 1;

    tensor
        .shape
        .dims
        .iter()
        .enumerate()
        .rev()
        .filter(|(index, _val)| *index != dim)
        .for_each(|(index, val)| {
            current *= val;
            strides[index] = current;
            num_elems_per_workgroup *= tensor.shape.dims[index];
        });

    println!("Strides {strides:?}");
    println!("Strides Real {:?}", tensor.strides);
    let workgroup = elemwise_workgroup(num_elems_per_workgroup, WORKGROUP_DEFAULT);

    execute_dynamic::<R, ScatterEagerKernel<R, E>, E>(
        &[
            EagerHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
            EagerHandle::new(&value.handle, &value.strides, &value.shape.dims),
        ],
        &[],
        None,
        kernel,
        WorkgroupLaunch::Custom(workgroup),
        indices.client,
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend, TestRuntime};
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn scatter_should_work_with_multiple_workgroups_2d_dim0() {
        same_as_reference_same_shape(0, [256, 32]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_2d_dim1() {
        same_as_reference_same_shape(1, [32, 256]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim0() {
        same_as_reference_same_shape(0, [256, 6, 6]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim1() {
        same_as_reference_same_shape(1, [6, 256, 6]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_3d_dim2() {
        same_as_reference_same_shape(2, [6, 6, 256]);
    }

    #[test]
    fn scatter_should_work_with_multiple_workgroups_diff_shapes() {
        same_as_reference_diff_shape(1, [32, 128], [32, 1]);
    }

    fn same_as_reference_diff_shape<const D: usize>(
        dim: usize,
        shape1: [usize; D],
        shape2: [usize; D],
    ) {
        TestBackend::seed(0);
        let test_device = Default::default();
        let tensor = Tensor::<TestBackend, D>::random(shape1, Distribution::Default, &test_device);
        let value = Tensor::<TestBackend, D>::random(shape2, Distribution::Default, &test_device);
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape2.iter().product()],
                Distribution::Uniform(0., shape2[dim] as f64),
                &test_device,
            )
            .into_data()
            .convert(),
            &test_device,
        )
        .reshape(shape2);
        let ref_device = Default::default();
        let tensor_ref = Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &ref_device);
        let value_ref = Tensor::<ReferenceBackend, D>::from_data(value.to_data(), &ref_device);
        let indices_ref =
            Tensor::<ReferenceBackend, D, Int>::from_data(indices.to_data().convert(), &ref_device);

        let actual = Tensor::<TestBackend, D>::from_primitive(scatter::<TestRuntime, _, _, D>(
            dim,
            tensor.into_primitive(),
            indices.into_primitive(),
            value.into_primitive(),
        ));
        let expected = tensor_ref.scatter(dim, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }

    fn same_as_reference_same_shape<const D: usize>(dim: usize, shape: [usize; D]) {
        same_as_reference_diff_shape(dim, shape, shape);
    }
}
