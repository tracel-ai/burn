use crate::codegen::dialect::gpu::{gpu, Elem, Scope, Variable};
use crate::{
    codegen::{
        dialect::gpu, execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler,
        EagerHandle, InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    kernel::{self, DynamicKernelSource, SourceTemplate},
    ops::numeric::empty_device,
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

#[derive(new)]
struct GatherEagerKernel<R: Runtime, E: JitElement> {
    dim: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct GatherComputeShader {
    tensor: Variable,
    indices: Variable,
    out: Variable,
    dim: usize,
}

impl GatherComputeShader {
    pub fn expand(self, scope: &mut Scope) {
        match self.tensor {
            Variable::GlobalInputArray(_, _) => (),
            Variable::GlobalOutputArray(_, _) => (),
            _ => panic!("Tensor variable must be an global array."),
        };

        let tensor = self.tensor;
        let output = self.out;

        let stride = scope.create_local(Elem::UInt);
        let offset = scope.create_local(Elem::UInt);

        // The offset of the `dim` dimension is obtained by the indices tensor.
        gpu!(scope, offset = cast(self.indices));
        gpu!(scope, stride = stride(tensor, self.dim));
        gpu!(scope, offset = offset * stride);

        // We fetch the offset before the `dim` dimension.
        if self.dim > 0 {
            let offset_before = scope.create_local(Elem::UInt);
            scope.index_offset_with_output_layout(gpu::IndexOffsetGlobalWithLayout {
                tensors: vec![tensor],
                indexes: vec![offset_before],
                layout: Variable::Id, // Will be updated.
                index_ref: Variable::Id,
                dim_start: 0u32.into(),
                dim_end: self.dim.into(),
            });
            gpu!(scope, offset += offset_before);
        }

        let offset_after = scope.create_local(Elem::UInt);
        scope.index_offset_with_output_layout(gpu::IndexOffsetGlobalWithLayout {
            tensors: vec![tensor],
            indexes: vec![offset_after],
            layout: Variable::Id, // Will be updated.
            index_ref: Variable::Id,
            dim_start: (self.dim + 1).into(),
            dim_end: Variable::Rank,
        });
        gpu!(scope, offset += offset_after);

        gpu!(scope, output = tensor[offset]);
    }
}

impl<R: Runtime, E: JitElement> DynamicKernelSource for GatherEagerKernel<R, E> {
    fn source(&self) -> kernel::SourceTemplate {
        let mut scope = gpu::Scope::root();
        let item_tensor = E::gpu_elem().into();
        let item_indices: gpu::Item = gpu::Elem::Int.into();

        let tensor = gpu::Variable::GlobalInputArray(0, item_tensor);
        let indices = scope.read_array(1, item_indices);

        let output_array = gpu::Variable::GlobalOutputArray(0, item_tensor);
        let output_local = scope.create_local(item_tensor);

        GatherComputeShader {
            tensor,
            indices,
            out: output_local,
            dim: self.dim,
        }
        .expand(&mut scope);

        scope.write_global(output_local, output_array);

        let tensor = InputInfo::Array {
            item: item_tensor,
            visibility: gpu::Visibility::Read,
        };
        let indices = InputInfo::Array {
            item: gpu::Elem::Int.into(),
            visibility: gpu::Visibility::Read,
        };
        let out = OutputInfo::Array { item: item_tensor };

        let info = CompilationInfo {
            inputs: vec![tensor, indices],
            outputs: vec![out],
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

pub(crate) fn gather<R: Runtime, E: JitElement, I: JitElement, const D: usize>(
    dim: usize,
    tensor: JitTensor<R, E, D>,
    indices: JitTensor<R, I, D>,
) -> JitTensor<R, E, D> {
    let shape_output = indices.shape.clone();
    let output = empty_device(tensor.client.clone(), tensor.device.clone(), shape_output);
    let kernel = GatherEagerKernel::new(dim);

    execute_dynamic::<R, GatherEagerKernel<R, E>, E>(
        &[
            EagerHandle::new(&tensor.handle, &tensor.strides, &tensor.shape.dims),
            EagerHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use crate::tests::{ReferenceBackend, TestBackend};
//     use burn_tensor::{backend::Backend, Distribution, Int, Shape, Tensor};
//
//     #[test]
//     fn gather_should_work_with_multiple_workgroups_dim0() {
//         test_same_as_ref([6, 256], 0);
//     }
//
//     #[test]
//     fn gather_should_work_with_multiple_workgroups_dim1() {
//         test_same_as_ref([6, 256], 1);
//     }
//
//     fn test_same_as_ref<const D: usize>(shape: [usize; D], dim: usize) {
//         TestBackend::seed(0);
//         let max = shape[dim];
//         let shape = Shape::new(shape);
//         let tensor = Tensor::<TestBackend, D>::random(
//             shape.clone(),
//             Distribution::Default,
//             &Default::default(),
//         );
//         let indices = Tensor::<TestBackend, 1, Int>::from_data(
//             Tensor::<TestBackend, 1>::random(
//                 [shape.num_elements()],
//                 Distribution::Uniform(0., max as f64),
//                 &Default::default(),
//             )
//             .into_data()
//             .convert(),
//             &Default::default(),
//         )
//         .reshape(shape);
//         let tensor_ref =
//             Tensor::<ReferenceBackend, D>::from_data(tensor.to_data(), &Default::default());
//         let indices_ref = Tensor::<ReferenceBackend, D, Int>::from_data(
//             indices.to_data().convert(),
//             &Default::default(),
//         );
//
//         let actual = Tensor::<TestBackend, D>::from_primitive(gather(
//             dim,
//             tensor.into_primitive(),
//             indices.into_primitive(),
//         ));
//         let expected = tensor_ref.gather(dim, indices_ref);
//
//         expected
//             .into_data()
//             .assert_approx_eq(&actual.into_data(), 3);
//     }
// }
