use crate::{
    element::JitElement,
    kernel::{into_contiguous, Kernel, SUBCUBE_DIM_APPROX},
    tensor::JitTensor,
    JitRuntime,
};
use burn_cube::{
    cpa, frontend::TensorHandle, CubeCountSettings, InputInfo, KernelExpansion, KernelIntegrator,
    KernelSettings, OutputInfo,
};
use burn_cube::{
    ir::{
        BinaryOperator, Branch, CubeDim, Elem, FloatKind, IndexOffsetGlobalWithLayout,
        KernelDefinition, Scope, Variable, Visibility,
    },
    Execution,
};
use std::marker::PhantomData;

use super::simple_launch_options;

#[derive(new, Debug)]
struct MatmulEagerKernel<R: JitRuntime, E: JitElement> {
    workgroup_size_x: usize,
    workgroup_size_y: usize,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

struct MatmulComputeShader {
    variables: BinaryOperator,
    block_size: usize,
}

impl MatmulComputeShader {
    fn expand(self, scope: &mut Scope) {
        // Define out global variables.
        let local_idx = Variable::UnitPos;
        let batch = Variable::AbsolutePosZ;
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
        cpa!(scope, tmp_index = local_idx / block_size);
        cpa!(scope, row = block_size * Variable::CubePosX);
        cpa!(scope, row = row + tmp_index);

        // Col position.
        cpa!(scope, tmp_index = local_idx % block_size);
        cpa!(scope, col = block_size * Variable::CubePosY);
        cpa!(scope, col = col + tmp_index);

        // Batch position.
        cpa!(scope, batch_dims = rank - 2u32);

        // Define the matrix size.
        let n_rows = scope.create_local(Elem::UInt);
        let n_cols = scope.create_local(Elem::UInt);
        let k = scope.create_local(Elem::UInt);

        // Number of rows.
        cpa!(scope, n_rows = shape(out, batch_dims));

        // Number of cols.
        cpa!(scope, tmp_index = batch_dims + 1u32);
        cpa!(scope, n_cols = shape(out, tmp_index));

        // The dimension that is going to be squashed.
        cpa!(scope, k = shape(lhs, tmp_index));

        // Check if there is some work to be done.
        let should_stop = scope.create_local(Elem::Bool);
        cpa!(scope, should_stop = row >= n_rows);
        cpa!(scope, if (should_stop).then(|scope| {
            scope.register(Branch::Return);
        }));

        cpa!(scope, should_stop = col >= n_cols);
        cpa!(scope, if (should_stop).then(|scope| {
            scope.register(Branch::Return);
        }));

        // Calculate the batch offset.
        let offset_lhs = scope.zero(Elem::UInt);
        let offset_rhs = scope.zero(Elem::UInt);
        let offset_output = scope.create_local(Elem::UInt);

        // Batch offset for the output.
        cpa!(scope, offset_output = n_rows * n_cols);
        cpa!(scope, offset_output = offset_output * batch);

        // Batch offset for the lhs & rhs matrices.
        IndexOffsetGlobalWithLayout {
            tensors: vec![lhs, rhs],
            indexes: vec![offset_lhs, offset_rhs],
            layout: out,
            position: offset_output,
            dim_start: 0u32.into(),
            dim_end: batch_dims,
        }
        .expand(scope);

        // Calculate the dot product (row X col).
        let sum = scope.create_local(out.item());

        // Initialize the sum to zero.
        let zero: Variable = 0f32.into();
        cpa!(scope, sum = zero);

        // Loop over the k dimension.
        cpa!(
            scope,
            range(0u32, k).for_each(|i, scope| {
                let lhs_index = scope.create_local(Elem::UInt);
                let rhs_index = scope.create_local(Elem::UInt);

                let lhs_value = scope.create_local(lhs.item());
                let rhs_value = scope.create_local(rhs.item());
                let out_value = scope.create_local(out.item());

                cpa!(scope, lhs_index = row * k);
                cpa!(scope, lhs_index = lhs_index + i);
                cpa!(scope, lhs_index = lhs_index + offset_lhs);

                cpa!(scope, rhs_index = i * n_cols);
                cpa!(scope, rhs_index = rhs_index + col);
                cpa!(scope, rhs_index = rhs_index + offset_rhs);

                cpa!(scope, lhs_value = lhs[lhs_index]);
                cpa!(scope, rhs_value = rhs[rhs_index]);

                cpa!(scope, out_value = lhs_value * rhs_value);
                cpa!(scope, sum += out_value);
            })
        );

        let out_index = scope.create_local(Elem::UInt);

        cpa!(scope, out_index = row * n_cols);
        cpa!(scope, out_index += col);
        cpa!(scope, out_index += offset_output);
        cpa!(scope, out[out_index] = sum);
    }
}

impl<R: JitRuntime, E: JitElement> Kernel for MatmulEagerKernel<R, E> {
    fn define(&self) -> KernelDefinition {
        assert_eq!(
            self.workgroup_size_x, self.workgroup_size_y,
            "Only square grid is supported."
        );

        let mut scope = Scope::root();
        let elem = E::cube_elem();
        assert!(
            elem == Elem::Float(FloatKind::F32) || elem == Elem::Float(FloatKind::F64),
            "Only float elements are supported."
        );
        let item = elem.into();

        let lhs = Variable::GlobalInputArray(0, item);
        let rhs = Variable::GlobalInputArray(1, item);
        let out = Variable::GlobalOutputArray(0, item);

        scope.write_global_custom(out);

        MatmulComputeShader {
            variables: BinaryOperator { lhs, rhs, out },
            block_size: self.workgroup_size_x,
        }
        .expand(&mut scope);

        let lhs = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let rhs = InputInfo::Array {
            item,
            visibility: Visibility::Read,
        };
        let out = OutputInfo::Array { item };

        let info = KernelExpansion {
            inputs: vec![lhs, rhs],
            outputs: vec![out],
            scope,
        };

        let settings = KernelSettings::default().cube_dim(CubeDim::new(
            self.workgroup_size_x as u32,
            self.workgroup_size_y as u32,
            1,
        ));
        KernelIntegrator::new(info).integrate(settings)
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
pub fn matmul_mem_coalescing_default<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    matmul_simple::<R, E, D>(lhs, rhs, out, SUBCUBE_DIM_APPROX, SUBCUBE_DIM_APPROX)
}

/// Matrix multiplication using memory coalescing algorithm with custom workgroup sizes
pub fn matmul_simple<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    workgroup_size_x: usize,
    workgroup_size_y: usize,
) -> JitTensor<R, E, D> {
    lhs.assert_is_on_same_device(&rhs);
    let lhs = into_contiguous(lhs);
    let rhs = into_contiguous(rhs);

    let workgroup = simple_launch_options(
        &lhs.shape,
        &rhs.shape,
        &out.shape,
        workgroup_size_x,
        workgroup_size_y,
    );

    let kernel = MatmulEagerKernel::<R, E>::new(workgroup_size_x, workgroup_size_y);

    Execution::start(kernel, rhs.client)
        .inputs(&[
            TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        ])
        .outputs(&[TensorHandle::new(
            &out.handle,
            &out.strides,
            &out.shape.dims,
        )])
        .execute(CubeCountSettings::Custom(workgroup));

    out
}
