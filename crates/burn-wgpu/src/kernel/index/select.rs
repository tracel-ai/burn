use std::marker::PhantomData;

use crate::{
    codegen::{
        dialect::gpu::{gpu, Elem, Item, Scope, Variable, Visibility},
        execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        InputInfo, OutputInfo, WorkgroupLaunch,
    },
    compute::StaticKernel,
    element::JitElement,
    kernel::{
        build_info, elemwise_workgroup, into_contiguous, DynamicKernelSource, KernelSettings,
        SourceTemplate, WORKGROUP_DEFAULT,
    },
    kernel_wgsl,
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};

kernel_wgsl!(IndexSelect, "../../template/index/select.wgsl");
kernel_wgsl!(
    SelectAssignInplace,
    "../../template/index/select_assign_inplace.wgsl"
);

#[derive(new)]
struct SelectEagerKernel<R: Runtime, E: JitElement> {
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

        gpu!(
            scope,
            range(0u32, Variable::Rank).for_each(|i, scope| {
                let stride_input = scope.create_local(Elem::UInt);
                let stride_output = scope.create_local(Elem::UInt);
                let shape_output = scope.create_local(Elem::UInt);

                gpu!(scope, stride_input = stride(input, i));
                gpu!(scope, stride_output = stride(output, i));
                gpu!(scope, shape_output = shape(output, i));

                let offset_local = scope.create_local(Elem::UInt);
                gpu!(scope, offset_local = id / stride_output);
                gpu!(scope, offset_local = offset_local % shape_output);

                let dim_index = scope.create_local(Elem::Bool);
                gpu!(scope, dim_index = i == self.dim);

                gpu!(scope, if(dim_index).then(|scope| {
                    gpu!(scope, offset_local = indices[offset_local]);
                    gpu!(scope, offset_local = offset_local * stride_input);
                }).else(|scope| {
                    gpu!(scope, offset_local = offset_local * stride_input);
                }));

                gpu!(scope, offset_input += offset_local);
            })
        );

        let value = scope.create_local(input.item());
        gpu!(scope, value = input[offset_input]);
        gpu!(scope, output[id] = value);
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for SelectEagerKernel<R, E> {
    fn source(&self) -> crate::kernel::SourceTemplate {
        let mut scope = Scope::root();
        let item = E::gpu_elem().into();
        let item_indices: Item = Elem::Int.into();

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
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!("{:?}dim={}", core::any::TypeId::of::<Self>(), self.dim)
    }
}

pub(crate) fn select<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
) -> JitTensor<R, E, D> {
    let mut shape_output = tensor.shape.clone();
    shape_output.dims[dim] = indices.shape.dims[0];

    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);
    let kernel = SelectEagerKernel::new(dim);
    println!("output {:?} - {:?}", output.strides, output.shape);

    execute_dynamic::<R, SelectEagerKernel<R, E>, E>(
        &[
            EagerHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&indices.handle, &[1; D], &[1; D]), // This is a current hacks because
                                                                 // the info buffer that contains
                                                                 // the strides and shapes is
                                                                 // hardcoded to only contains
                                                                 // information about tensors of
                                                                 // the same rank.
        ],
        &[EagerHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )],
        None,
        kernel,
        WorkgroupLaunch::Output { pos: 0 },
        tensor.client,
    );

    output
}

pub(crate) fn select_assign<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    tensor: JitTensor<R, E, D>,
    dim: usize,
    indices: JitTensor<R, I, 1>,
    value: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    let tensor = match tensor.can_mut() {
        true => tensor,
        false => tensor.copy(),
    };

    let mut info = build_info(&[&tensor, &value]);
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

    strides
        .into_iter()
        .for_each(|stride| info.push(stride as u32));

    info.push(dim as u32);

    let info_handle = tensor.client.create(bytemuck::cast_slice(&info));

    let kernel = StaticKernel::<
        KernelSettings<SelectAssignInplace, E, I, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT, 1>,
    >::new(elemwise_workgroup(
        num_elems_per_workgroup,
        WORKGROUP_DEFAULT,
    ));

    tensor.client.execute(
        Box::new(kernel),
        &[&tensor.handle, &indices.handle, &value.handle, &info_handle],
    );

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{ReferenceBackend, TestBackend, TestRuntime};
    use burn_tensor::{backend::Backend, Distribution, Int, Tensor};

    #[test]
    fn select_should_work_with_multiple_workgroups() {
        let tensor =
            Tensor::<TestBackend, 2>::random([6, 256], Distribution::Default, &Default::default());
        let indices = Tensor::<TestBackend, 1, Int>::arange(0..100, &Default::default());
        let tensor_ref =
            Tensor::<ReferenceBackend, 2>::from_data(tensor.to_data(), &Default::default());
        let indices_ref = Tensor::<ReferenceBackend, 1, Int>::from_data(
            indices.to_data().convert(),
            &Default::default(),
        );

        let actual = select(tensor.into_primitive(), 1, indices.into_primitive());
        let expected = tensor_ref.select(1, indices_ref);

        expected.into_data().assert_approx_eq(
            &Tensor::<TestBackend, 2>::from_primitive(actual).into_data(),
            3,
        );
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_2d_dim0() {
        select_assign_same_as_ref(0, [256, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_2d_dim1() {
        select_assign_same_as_ref(1, [6, 256]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim0() {
        select_assign_same_as_ref(0, [256, 6, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim1() {
        select_assign_same_as_ref(1, [6, 256, 6]);
    }

    #[test]
    fn select_assign_should_work_with_multiple_workgroups_3d_dim2() {
        select_assign_same_as_ref(2, [6, 6, 256]);
    }

    fn select_assign_same_as_ref<const D: usize>(dim: usize, shape: [usize; D]) {
        TestBackend::seed(0);
        let tensor =
            Tensor::<TestBackend, D>::random(shape, Distribution::Default, &Default::default());
        let value =
            Tensor::<TestBackend, D>::random(shape, Distribution::Default, &Default::default());
        let indices = Tensor::<TestBackend, 1, Int>::from_data(
            Tensor::<TestBackend, 1>::random(
                [shape[dim]],
                Distribution::Uniform(0., shape[dim] as f64),
                &Default::default(),
            )
            .into_data()
            .convert(),
            &Default::default(),
        );
        let tensor_ref =
            Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &Default::default());
        let value_ref =
            Tensor::<ReferenceBackend, D>::from_data(value.to_data(), &Default::default());
        let indices_ref = Tensor::<ReferenceBackend, 1, Int>::from_data(
            indices.to_data().convert(),
            &Default::default(),
        );

        let actual =
            Tensor::<TestBackend, D>::from_primitive(select_assign::<TestRuntime, _, _, D>(
                tensor.into_primitive(),
                dim,
                indices.into_primitive(),
                value.into_primitive(),
            ));
        let expected = tensor_ref.select_assign(dim, indices_ref, value_ref);

        expected
            .into_data()
            .assert_approx_eq(&actual.into_data(), 3);
    }
}
