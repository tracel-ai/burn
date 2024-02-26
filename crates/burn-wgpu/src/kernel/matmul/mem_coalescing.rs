use crate::codegen::dialect::gpu::{
    gpu, BinaryOperator, Branch, Elem, IndexOffsetGlobalWithLayout, Scope, Variable,
};
use crate::{
    codegen::{
        dialect::gpu, execute_dynamic, Compilation, CompilationInfo, CompilationSettings, Compiler,
        EagerHandle, InputInfo, OutputInfo, WorkgroupLaunch,
    },
    compute::WorkGroup,
    element::JitElement,
    kernel::{into_contiguous, DynamicKernelSource, SourceTemplate, WORKGROUP_DEFAULT},
    tensor::JitTensor,
    Runtime,
};
use burn_tensor::Shape;
use std::marker::PhantomData;

#[derive(new, Debug)]
struct MatmulEagerKernel<R: Runtime> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _runtime: PhantomData<R>,
}

struct MatmulComputeShader {
    variables: BinaryOperator,
    block_size: usize,
}

impl MatmulComputeShader {
    fn expand(self, scope: &mut Scope) {
        // Define out global variables.
        let local_idx = Variable::InvocationIndex;
        let batch = Variable::GlobalInvocationIdZ;
        let rank = Variable::Rank;
        let block_size: Variable = self.block_size.into();

        // Extract tensor variables.
        let lhs = self.variables.lhs;
        let rhs = self.variables.rhs;
        let out = self.variables.out;

        // Define where we have to work on the current matrix.
        let tmp_index = scope.create_local(Elem::UInt);
        let batch_dims = scope.create_local(Elem::UInt);
        let row = scope.create_local(Elem::UInt);
        let col = scope.create_local(Elem::UInt);

        // Row position.
        gpu!(scope, tmp_index = local_idx / block_size);
        gpu!(scope, row = block_size * Variable::WorkgroupIdX);
        gpu!(scope, row = row + tmp_index);

        // Col position.
        gpu!(scope, tmp_index = local_idx % block_size);
        gpu!(scope, col = block_size * Variable::WorkgroupIdY);
        gpu!(scope, col = col + tmp_index);

        // Batch position.
        gpu!(scope, batch_dims = rank - 2u32);

        // Define the matrix size.
        let n_rows = scope.create_local(Elem::UInt);
        let n_cols = scope.create_local(Elem::UInt);
        let k = scope.create_local(Elem::UInt);

        // Number of rows.
        gpu!(scope, n_rows = shape(out, batch_dims));

        // Number of cols.
        gpu!(scope, tmp_index = batch_dims + 1u32);
        gpu!(scope, n_cols = shape(out, tmp_index));

        // The dimension that is going to be squashed.
        gpu!(scope, k = shape(lhs, tmp_index));

        // Check if there is some work to be done.
        let should_stop = scope.create_local(Elem::Bool);
        gpu!(scope, should_stop = row >= n_rows);
        gpu!(scope, if (should_stop).then(|scope| {
            scope.register(Branch::Return);
        }));

        gpu!(scope, should_stop = col >= n_cols);
        gpu!(scope, if (should_stop).then(|scope| {
            scope.register(Branch::Return);
        }));

        // Calculate the batch offset.
        let offset_lhs = scope.zero(Elem::UInt);
        let offset_rhs = scope.zero(Elem::UInt);
        let offset_output = scope.create_local(Elem::UInt);

        // Batch offset for the output.
        gpu!(scope, offset_output = n_rows * n_cols);
        gpu!(scope, offset_output = offset_output * batch);

        // Batch offset for the lhs & rhs matrices.
        IndexOffsetGlobalWithLayout {
            tensors: vec![lhs, rhs],
            indexes: vec![offset_lhs, offset_rhs],
            layout: out,
            index_ref: offset_output,
            dim_start: 0u32.into(),
            dim_end: batch_dims,
        }
        .expand(scope);

        // Calculate the dot product (row X col).
        let sum = scope.create_local(out.item());

        // Initialize the sum to zero.
        let zero: Variable = 0f32.into();
        gpu!(scope, sum = zero);

        // Loop over the k dimension.
        gpu!(
            scope,
            range(0u32, k).for_each(|i, scope| {
                let lhs_index = scope.create_local(Elem::UInt);
                let rhs_index = scope.create_local(Elem::UInt);

                let lhs_value = scope.create_local(lhs.item());
                let rhs_value = scope.create_local(rhs.item());
                let out_value = scope.create_local(out.item());

                gpu!(scope, lhs_index = row * k);
                gpu!(scope, lhs_index = lhs_index + i);
                gpu!(scope, lhs_index = lhs_index + offset_lhs);

                gpu!(scope, rhs_index = i * n_cols);
                gpu!(scope, rhs_index = rhs_index + col);
                gpu!(scope, rhs_index = rhs_index + offset_rhs);

                gpu!(scope, lhs_value = lhs[lhs_index]);
                gpu!(scope, rhs_value = rhs[rhs_index]);

                gpu!(scope, out_value = lhs_value * rhs_value);
                gpu!(scope, sum += out_value);
            })
        );

        let out_index = scope.create_local(Elem::UInt);

        gpu!(scope, out_index = row * n_cols);
        gpu!(scope, out_index += col);
        gpu!(scope, out_index += offset_output);
        gpu!(scope, out[out_index] = sum);
    }
}

impl<R: Runtime> DynamicKernelSource for MatmulEagerKernel<R> {
    fn source(&self) -> SourceTemplate {
        assert_eq!(
            self.workgroup_size_x, self.workgroup_size_y,
            "Only square grid is supported."
        );

        let mut scope = gpu::Scope::root();
        let lhs = gpu::Variable::GlobalInputArray(0, gpu::Elem::Float.into());
        let rhs = gpu::Variable::GlobalInputArray(1, gpu::Elem::Float.into());
        let out = gpu::Variable::GlobalOutputArray(0, gpu::Elem::Float.into());

        scope.write_global_custom(out);

        MatmulComputeShader {
            variables: gpu::BinaryOperator { lhs, rhs, out },
            block_size: self.workgroup_size_x,
        }
        .expand(&mut scope);

        let lhs = InputInfo::Array {
            item: gpu::Elem::Float.into(),
            visibility: gpu::Visibility::Read,
        };
        let rhs = InputInfo::Array {
            item: gpu::Elem::Float.into(),
            visibility: gpu::Visibility::Read,
        };
        let out = OutputInfo::Array {
            item: gpu::Elem::Float.into(),
        };

        let info = CompilationInfo {
            inputs: vec![lhs, rhs],
            outputs: vec![out],
            scope,
            mappings: vec![],
        };

        let settings = CompilationSettings::default().workgroup_size(gpu::WorkgroupSize::new(
            self.workgroup_size_x as u32,
            self.workgroup_size_y as u32,
            1,
        ));
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!(
            "{:?}x={}y={}",
            core::any::TypeId::of::<Self>(),
            self.workgroup_size_x,
            self.workgroup_size_y,
        )
    }
}

/// Matrix multiplication using memory coalescing algorithm with workgroups of size 16
pub fn matmul_mem_coalescing_default<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    matmul_mem_coalescing::<R, E, D>(lhs, rhs, out, WORKGROUP_DEFAULT, WORKGROUP_DEFAULT)
}

/// Matrix multiplication using memory coalescing algorithm with custom workgroup sizes
pub fn matmul_mem_coalescing<R: Runtime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> JitTensor<R, E, D> {
    lhs.assert_is_on_same_device(&rhs);
    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let workgroup = launch_options(
        &lhs.shape,
        &rhs.shape,
        &out.shape,
        workgroup_size_x,
        workgroup_size_y,
    );

    let kernel = MatmulEagerKernel::new(workgroup_size_x, workgroup_size_y);

    execute_dynamic::<R, MatmulEagerKernel<R>, E>(
        &[
            EagerHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            EagerHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            EagerHandle::new(&out.handle, &out.strides, &out.shape.dims),
        ],
        &[],
        None,
        kernel,
        WorkgroupLaunch::Custom(workgroup),
        rhs.client,
    );

    out
}

fn launch_options<const D: usize>(
    lhs_shape: &Shape<D>,
    rhs_shape: &Shape<D>,
    output_shape: &Shape<D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> WorkGroup {
    let num_rows = lhs_shape.dims[D - 2];
    let num_cols = rhs_shape.dims[D - 1];

    // set number of workgroups
    let blocks_needed_in_x = f32::ceil(num_rows as f32 / workgroup_size_x as f32) as u32;
    let blocks_needed_in_y = f32::ceil(num_cols as f32 / workgroup_size_y as f32) as u32;
    let mut num_iter = 1;
    for i in 0..D - 2 {
        num_iter *= output_shape.dims[i];
    }

    WorkGroup::new(blocks_needed_in_x, blocks_needed_in_y, num_iter as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        kernel::matmul::utils::tests::{same_as_reference, same_as_reference_swapped_dims},
        tests::TestRuntime,
    };

    #[test]
    pub fn test_matmul_mem_coalescing_straightforward() {
        test_with_params::<2, 2>(1, 2, 1, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_shapes_smaller_than_blocks() {
        test_with_params::<16, 16>(8, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_n_smaller_than_m() {
        test_with_params::<2, 2>(8, 8, 3, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_m_smaller_than_n() {
        test_with_params::<2, 2>(3, 8, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_k_smaller_than_m_n() {
        test_with_params::<2, 2>(8, 3, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_k_larger_than_m_n() {
        test_with_params::<2, 2>(8, 48, 8, 1, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_multibatch_1_dim() {
        test_with_params::<2, 2>(8, 8, 8, 3, 1);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_multibatch_2_dims() {
        test_with_params::<2, 2>(8, 8, 8, 3, 4);
    }

    #[test]
    pub fn test_matmul_mem_coalescing_blocks_divide_shapes_unevenly() {
        test_with_params::<3, 3>(7, 7, 7, 1, 1);
    }

    fn test_with_params<const WORKGROUP_SIZE_X: usize, const WORKGROUP_SIZE_Y: usize>(
        m: usize,
        k: usize,
        n: usize,
        batch_1: usize,
        batch_2: usize,
    ) {
        let func = |lhs, rhs, out| {
            matmul_mem_coalescing::<TestRuntime, f32, 4>(
                lhs,
                rhs,
                out,
                WORKGROUP_SIZE_X,
                WORKGROUP_SIZE_Y,
            )
        };
        let shape_lhs = [batch_1, batch_2, m, k];
        let shape_rhs = [batch_1, batch_2, k, n];
        same_as_reference(func, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_batches_no_padding() {
        let matmul_func =
            |lhs, rhs, out| matmul_mem_coalescing::<TestRuntime, f32, 4>(lhs, rhs, out, 2, 2);
        let swap = [0, 1];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap, swap, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_col_no_padding() {
        let matmul_func =
            |lhs, rhs, out| matmul_mem_coalescing::<TestRuntime, f32, 4>(lhs, rhs, out, 2, 2);
        let swap_lhs = [0, 0];
        let swap_rhs = [2, 3];
        let shape_lhs = [3, 2, 4, 4];
        let shape_rhs = [3, 2, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }

    #[test]
    fn test_matmul_naive_swapped_row_with_batch_no_padding() {
        let matmul_func =
            |lhs, rhs, out| matmul_mem_coalescing::<TestRuntime, f32, 4>(lhs, rhs, out, 2, 2);
        let swap_lhs = [0, 3];
        let swap_rhs = [0, 2];
        let shape_lhs = [4, 4, 4, 4];
        let shape_rhs = [4, 4, 4, 4];
        same_as_reference_swapped_dims(matmul_func, swap_lhs, swap_rhs, shape_lhs, shape_rhs);
    }
}
