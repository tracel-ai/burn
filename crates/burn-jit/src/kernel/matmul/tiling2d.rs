use burn_tensor::Element;

use crate::{
    codegen::{
        dialect::gpu, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        Execution, InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::{gpu, BinaryOperator, Branch, Elem, IndexOffsetGlobalWithLayout, Scope, Variable},
    kernel::{into_contiguous, DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

use super::{tiling2d_launch_options, Tiling2dConfig};

#[derive(new, Debug)]
struct MatmulTiling2d<E: JitElement> {
    _elem: PhantomData<E>,
}

#[derive(new, Debug)]
struct MatmulTiling2dEagerKernel<R: Runtime> {
    config: Tiling2dConfig,
    _runtime: PhantomData<R>,
}

struct MatmulTiling2dShader {
    variables: BinaryOperator,
    block_size: usize,
}

impl MatmulTiling2dShader {
    fn expand(self, scope: &mut Scope) {
        // Define out global variables.
        let local_idx = Variable::LocalInvocationIndex;
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

impl<R: Runtime> DynamicKernelSource for MatmulTiling2dEagerKernel<R> {
    fn source(&self) -> SourceTemplate {
        let mut scope = gpu::Scope::root();
        let lhs = gpu::Variable::GlobalInputArray(0, gpu::Elem::Float.into());
        let rhs = gpu::Variable::GlobalInputArray(1, gpu::Elem::Float.into());
        let out = gpu::Variable::GlobalOutputArray(0, gpu::Elem::Float.into());

        scope.write_global_custom(out);

        MatmulTiling2dShader {
            variables: gpu::BinaryOperator { lhs, rhs, out },
            block_size: self.config.grid_x, // TODO
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
        };

        let settings = CompilationSettings::default().workgroup_size(gpu::WorkgroupSize::new(
            self.config.grid_x as u32,
            self.config.grid_y as u32,
            1,
        ));
        let shader = Compilation::new(info).compile(settings);
        let shader = <R::Compiler as Compiler>::compile(shader);
        SourceTemplate::new(shader.to_string())
    }

    fn id(&self) -> String {
        format!(
            "{:?}config={:?}",
            core::any::TypeId::of::<Self>(),
            self.config,
        )
    }
}

/// Matrix multiplication using tiling 2d algorithm with
/// vec4 primitive on both lhs and rhs, with no padding needed
pub fn matmul_tiling_2d<R: Runtime, E: JitElement + Element, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let kernel = MatmulTiling2dEagerKernel::<R>::new(config.clone());
    let client = lhs.client.clone();

    let lhs = match lhs.batch_swapped_with_row_col() {
        true => into_contiguous(lhs),
        false => lhs,
    };
    let rhs = match rhs.batch_swapped_with_row_col() {
        true => into_contiguous(rhs),
        false => rhs,
    };

    Execution::start(kernel, client)
        .inputs(&[
            EagerHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            EagerHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        ])
        .outputs(&[EagerHandle::new(&out.handle, &out.strides, &out.shape.dims)])
        .execute(WorkgroupLaunch::Custom(tiling2d_launch_options(
            &out.shape, config,
        )));

    out
}
