use burn_tensor::Element;

use crate::{
    codegen::{
        dialect::gpu, Compilation, CompilationInfo, CompilationSettings, Compiler, EagerHandle,
        Execution, InputInfo, OutputInfo, WorkgroupLaunch,
    },
    element::JitElement,
    gpu::{
        gpu, BinaryOperator, Branch, Elem, IndexOffsetGlobalWithLayout, Item, Scope,
        Synchronization, Variable,
    },
    kernel::{into_contiguous, DynamicKernelSource, SourceTemplate},
    tensor::JitTensor,
    Runtime,
};
use std::marker::PhantomData;

use super::{
    launch_options,
    padding::{crop, pad_round, PaddingOutput},
    shape_out, Tiling2dConfig,
};

#[derive(new, Debug)]
struct MatmulTiling2dPadded<E: JitElement> {
    _elem: PhantomData<E>,
}

#[derive(new, Debug)]
struct MatmulTiling2dPaddedEagerKernel<R: Runtime> {
    config: Tiling2dConfig,
    _runtime: PhantomData<R>,
}

struct MatmulTiling2dPaddedShader {
    variables: BinaryOperator,
    config: Tiling2dConfig,
}

impl MatmulTiling2dPaddedShader {
    fn expand(self, scope: &mut Scope) {
        // Phase 1: Gather information: input, shader and offsets

        // Inputs
        let lhs = self.variables.lhs;
        let rhs = self.variables.rhs;
        let out = self.variables.out;

        // Config variables
        let block_size_m: Variable = self.config.block_size_m.into();
        let block_size_k: Variable = self.config.block_size_k.into();
        let block_size_n: Variable = self.config.block_size_n.into();
        let tile_size_m: Variable = self.config.tile_size_m.into();
        let tile_size_n: Variable = self.config.tile_size_n.into();
        let n_threads_per_row: Variable =
            (((self.config.block_size_n - 1) / self.config.tile_size_n) + 1).into();
        let results_size = (self.config.tile_size_m * self.config.tile_size_n) as u32;

        // Shader info
        let local_idx = Variable::LocalInvocationIndex;
        let batch = Variable::GlobalInvocationIdZ;

        // Shapes
        let rank = Variable::Rank;
        let penultimate_dim = scope.create_local(Elem::UInt);
        gpu!(scope, penultimate_dim = rank - 1u32);
        let M = scope.create_local(Elem::UInt);
        let K = scope.create_local(Elem::UInt);
        let N = scope.create_local(Elem::UInt);
        gpu!(scope, M = shape(lhs, penultimate_dim));
        gpu!(scope, K = shape(rhs, penultimate_dim));
        gpu!(scope, N = shape(rhs, rank));

        // Strides
        let lhs_stride_row = scope.create_local(Elem::UInt);
        let lhs_stride_col = scope.create_local(Elem::UInt);
        let rhs_stride_row = scope.create_local(Elem::UInt);
        let rhs_stride_col = scope.create_local(Elem::UInt);
        let out_stride_row = scope.create_local(Elem::UInt);
        let out_stride_col = scope.create_local(Elem::UInt);
        gpu!(scope, lhs_stride_row = stride(lhs, penultimate_dim));
        gpu!(scope, lhs_stride_col = stride(lhs, rank));
        gpu!(scope, rhs_stride_row = stride(rhs, penultimate_dim));
        gpu!(scope, rhs_stride_col = stride(rhs, rank));
        gpu!(scope, out_stride_row = stride(out, penultimate_dim));
        gpu!(scope, out_stride_col = stride(out, rank));

        // Workgroup offset
        let skip_row = scope.create_local(Elem::UInt);
        let workgroup_id_x = Variable::WorkgroupIdX;
        gpu!(scope, skip_row = workgroup_id_x);
        gpu!(scope, skip_row *= block_size_m);
        let skip_col = scope.create_local(Elem::UInt);
        let workgroup_id_y = Variable::WorkgroupIdY;
        gpu!(scope, skip_col = workgroup_id_y);
        gpu!(scope, skip_col *= block_size_n);

        // Invocation offset
        let thread_row = scope.create_local(Elem::UInt);
        gpu!(scope, thread_row = local_idx / n_threads_per_row);
        gpu!(scope, thread_row *= tile_size_m);
        let thread_col = scope.create_local(Elem::UInt);
        gpu!(scope, thread_col = local_idx % n_threads_per_row);
        gpu!(scope, thread_col *= tile_size_n);

        // Row and col
        let row = scope.create_local(Elem::UInt);
        let col = scope.create_local(Elem::UInt);
        gpu!(scope, row = skip_row + thread_row);
        gpu!(scope, col = skip_col + thread_col);

        // Calculate offset.
        let offset_lhs = scope.create_local(Elem::UInt);
        let offset_rhs = scope.create_local(Elem::UInt);
        gpu!(scope, offset_lhs = skip_row * lhs_stride_row);
        gpu!(scope, offset_rhs = skip_col * rhs_stride_col);

        // Batch offset for the output.
        let offset_output = scope.create_local(Elem::UInt);
        let batch_dims = scope.create_local(Elem::UInt);
        gpu!(scope, offset_output = M * N);
        gpu!(scope, offset_output = offset_output * batch);

        // Batch offset for the lhs & rhs matrices.
        gpu!(scope, batch_dims = rank - 2u32);
        IndexOffsetGlobalWithLayout {
            tensors: vec![lhs, rhs],
            indexes: vec![offset_lhs, offset_rhs],
            layout: out,
            index_ref: offset_output,
            dim_start: 0u32.into(),
            dim_end: batch_dims,
        }
        .expand(scope);

        // Phase 2: Loop over k for loading and computing

        let results = scope.create_local_array(lhs.item().elem(), results_size);
        let register_m = scope.create_local(Item::Vec4(lhs.item().elem()));
        let register_n = scope.create_local(Item::Vec4(lhs.item().elem()));
        let shared_lhs = scope.create_shared(
            Item::Vec4(lhs.item().elem()),
            self.config.block_size_m as u32 * self.config.block_size_k as u32 / 4u32,
        );
        let shared_rhs = scope.create_shared(
            Item::Vec4(rhs.item().elem()),
            self.config.block_size_k as u32 * self.config.block_size_n as u32 / 4u32,
        );

        let n_loops = scope.create_local(Elem::UInt);
        gpu!(scope, n_loops = K / block_size_k); // assumes padding, otherwise ceil
        gpu!(
            scope,
            range(0u32, n_loops).for_each(|i, scope| {
                // Equivalent of looping from 0 to K with steps block_size_k
                let k = scope.create_local(Elem::UInt);
                gpu!(scope, k = i * block_size_k);

                // Phase 2.1: Load to shared memory

                // LHS
                load_shared_memory(
                    scope,
                    k,
                    block_size_k,
                    block_size_n,
                    thread_col,
                    thread_row,
                    lhs_stride_col,
                    lhs_stride_row,
                    lhs,
                    offset_lhs,
                    shared_lhs,
                    true,
                );

                // RHS
                load_shared_memory(
                    scope,
                    k,
                    block_size_k,
                    block_size_n,
                    thread_row,
                    thread_col,
                    rhs_stride_row,
                    rhs_stride_col,
                    rhs,
                    offset_rhs,
                    shared_rhs,
                    false,
                );

                scope.register(Synchronization::WorkgroupBarrier);

                // Phase 2.2: Compute intermediate results

                computation_loop();

                scope.register(Synchronization::WorkgroupBarrier);
            })
        );

        // Phase 3: Write to output
    }

    fn load_shared_memory(
        scope: &mut Scope,
        k: Variable,
        block_size_k: Variable,
        block_size_n: Variable,
        thread_idx_1: Variable,
        thread_idx_2: Variable,
        stride_1: Variable,
        stride_2: Variable,
        input: Variable,
        input_offset: Variable,
        shared_memory: Variable,
        is_lhs: bool,
    ) {
        for j in 0u32..4u32 {
            let current_col = scope.create_local(Elem::UInt);
            gpu!(scope, current_col = thread_idx_1 + j);

            let aligned_with_shared_memory = scope.create_local(Elem::Bool);
            gpu!(
                scope,
                aligned_with_shared_memory = current_col < block_size_k
            );

            gpu!(scope, if(aligned_with_shared_memory).then(|scope|{
                let lhs_sm_position = scope.create_local(Elem::UInt);
                if is_lhs {
                    gpu!(scope, lhs_sm_position = thread_idx_2 / 4u32);
                    gpu!(scope, lhs_sm_position *= block_size_k);
                    gpu!(scope, lhs_sm_position += current_col);
                } else {
                    gpu!(scope, lhs_sm_position = current_col * block_size_n);
                    gpu!(scope, lhs_sm_position += thread_idx_2);
                    gpu!(scope, lhs_sm_position = lhs_sm_position / 4u32);
                }

                let lhs_position_0 = scope.create_local(Elem::UInt);
                gpu!(scope, lhs_position_0 = k + current_col);
                gpu!(scope, lhs_position_0 *= stride_1);
                let tmp = scope.create_local(Elem::UInt);
                gpu!(scope, tmp = thread_idx_2 * stride_2);
                gpu!(scope, lhs_position_0 += tmp);
                gpu!(scope, lhs_position_0 += input_offset);
                let lhs_position_1 = scope.create_local(Elem::UInt);
                let lhs_position_2 = scope.create_local(Elem::UInt);
                let lhs_position_3 = scope.create_local(Elem::UInt);
                gpu!(scope, lhs_position_1 = lhs_position_0 + stride_2);
                gpu!(scope, lhs_position_2 = lhs_position_1 + stride_2);
                gpu!(scope, lhs_position_3 = lhs_position_2 + stride_2);

                let lhs_0 = scope.create_local(input.item().elem());
                let lhs_1 = scope.create_local(input.item().elem());
                let lhs_2 = scope.create_local(input.item().elem());
                let lhs_3 = scope.create_local(input.item().elem());
                gpu!(scope, lhs_0 = input[lhs_position_0]);
                gpu!(scope, lhs_1 = input[lhs_position_1]);
                gpu!(scope, lhs_2 = input[lhs_position_2]);
                gpu!(scope, lhs_3 = input[lhs_position_3]);

                let lhs_vec4 = scope.create_local(shared_memory.item());
                gpu!(scope, lhs_vec4 = vec4(lhs_0, lhs_1, lhs_2, lhs_3));
                gpu!(scope, shared_memory[lhs_sm_position] = lhs_vec4);

            }).else(|scope|{
                scope.register(Branch::Break); // TODO test if faster, else remove
            }));
        }
    }

    fn computation_loop(
        scope: &mut Scope,
        block_size_k: Variable, // needed in computation, but also use attribute for unrolling
        thread_row: Variable,
        shared_lhs: Variable,
        register_m: Variable,
        block_size_n: Variable,
        thread_col: Variable,
        shared_rhs: Variable,
        register_n: Variable, // TM/TN use attribute for unrolling
        results: Variable,
    ) {
    }
}

impl<R: Runtime> DynamicKernelSource for MatmulTiling2dPaddedEagerKernel<R> {
    fn source(&self) -> SourceTemplate {
        let mut scope = gpu::Scope::root();
        let lhs = gpu::Variable::GlobalInputArray(0, gpu::Elem::Float.into());
        let rhs = gpu::Variable::GlobalInputArray(1, gpu::Elem::Float.into());
        let out = gpu::Variable::GlobalOutputArray(0, gpu::Elem::Float.into());

        scope.write_global_custom(out);

        MatmulTiling2dPaddedShader {
            variables: gpu::BinaryOperator { lhs, rhs, out },
            config: self.config.clone(),
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
pub fn matmul_tiling_2d_padded<R: Runtime, E: JitElement + Element, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let kernel = MatmulTiling2dPaddedEagerKernel::<R>::new(config.clone());
    let client = lhs.client.clone();

    // A tensor may need to be padded, in which case it will implicitly become contiguous
    // If not needed, it is only turned into contiguous if some batch dim has been swapped with row or col dim.
    // If batches were swapped among themselves, or if the last two dims are transposed, the underlying
    // kernel handles it without needing to turn it into contiguous.
    let round_lhs = pad_round::<R, E, D>(lhs, config.block_size_m, config.block_size_k);
    let lhs = match round_lhs {
        PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
            into_contiguous(tensor)
        }
        _ => round_lhs.into_tensor(),
    };
    let round_rhs = pad_round::<R, E, D>(rhs, config.block_size_k, config.block_size_n);
    let rhs = match round_rhs {
        PaddingOutput::Unchanged(tensor) if tensor.batch_swapped_with_row_col() => {
            into_contiguous(tensor)
        }
        _ => round_rhs.into_tensor(),
    };

    let rounded_output_shape = shape_out(&lhs, &rhs);

    let num_elems = rounded_output_shape.num_elements();
    let buffer = client.empty(num_elems * core::mem::size_of::<E>());
    let rounded_output = JitTensor::new(
        rhs.client.clone(),
        rhs.device.clone(),
        rounded_output_shape.clone(),
        buffer,
    );

    Execution::start(kernel, client)
        .inputs(&[
            EagerHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            EagerHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        ])
        .outputs(&[EagerHandle::new(
            &rounded_output.handle,
            &rounded_output.strides,
            &rounded_output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Custom(launch_options(
            &lhs.shape,
            &rhs.shape,
            &out.shape,
            config.grid_x,
            config.grid_y,
        )));

    crop(rounded_output, out)
}
