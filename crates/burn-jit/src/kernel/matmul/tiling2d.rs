use burn_cube::{
    dialect::{
        BinaryOperator, ComputeShader, Elem, FloatKind, Scope, Variable, Visibility, WorkgroupSize,
    },
    Compilation, CompilationInfo, CompilationSettings, Execution, InputInfo, OutputInfo,
    TensorHandle, WorkgroupLaunch,
};
use burn_tensor::{Element, Shape};

use crate::{
    element::JitElement,
    kernel::{into_contiguous, GpuComputeShaderPhase},
    tensor::JitTensor,
    JitRuntime,
};
use std::marker::PhantomData;

use super::{
    padding::{crop, pad_round, PaddingOutput},
    shape_out, tiling2d_launch_options,
    tiling2d_shader::MatmulTiling2dShader,
    Tiling2dConfig,
};

#[derive(new, Debug)]
struct MatmulTiling2dEagerKernel<R: JitRuntime, E: JitElement> {
    config: Tiling2dConfig,
    bounds_check_required: bool,
    _runtime: PhantomData<R>,
    _elem: PhantomData<E>,
}

impl<R: JitRuntime, E: JitElement> GpuComputeShaderPhase for MatmulTiling2dEagerKernel<R, E> {
    fn compile(&self) -> ComputeShader {
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

        MatmulTiling2dShader {
            variables: BinaryOperator { lhs, rhs, out },
            config: self.config.clone(),
            bounds_check_required: self.bounds_check_required,
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

        let info = CompilationInfo {
            inputs: vec![lhs, rhs],
            outputs: vec![out],
            scope,
        };

        let settings = CompilationSettings::default().workgroup_size(WorkgroupSize::new(
            self.config.grid_x as u32,
            self.config.grid_y as u32,
            1,
        ));
        Compilation::new(info).compile(settings)
    }

    fn id(&self) -> String {
        format!(
            "{:?}config={:?}boundcheck={:?}",
            core::any::TypeId::of::<Self>(),
            self.config,
            self.bounds_check_required
        )
    }
}

/// Matrix multiplication using tiling 2d algorithm with
/// vec4 primitive on both lhs and rhs, with no padding needed
pub fn matmul_tiling_2d<R: JitRuntime, E: JitElement + Element, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let bounds_check_required = check_bound_requirement(&lhs.shape, &rhs.shape, &config);

    let kernel = MatmulTiling2dEagerKernel::<R, E>::new(config.clone(), bounds_check_required);
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
            TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        ])
        .outputs(&[TensorHandle::new(
            &out.handle,
            &out.strides,
            &out.shape.dims,
        )])
        .execute(WorkgroupLaunch::Custom(tiling2d_launch_options(
            &out.shape, config,
        )));

    out
}

/// Matrix multiplication using tiling 2d algorithm with padding needed
pub fn matmul_tiling_2d_padded<R: JitRuntime, E: JitElement + Element, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    out: JitTensor<R, E, D>,
    config: Tiling2dConfig,
) -> JitTensor<R, E, D> {
    let kernel = MatmulTiling2dEagerKernel::<R, E>::new(config.clone(), false);
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
            TensorHandle::<R>::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
            TensorHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
        ])
        .outputs(&[TensorHandle::new(
            &rounded_output.handle,
            &rounded_output.strides,
            &rounded_output.shape.dims,
        )])
        .execute(WorkgroupLaunch::Custom(tiling2d_launch_options(
            &rounded_output.shape,
            config,
        )));

    crop(rounded_output, out)
}

fn check_bound_requirement<const D: usize>(
    lhs_shape: &Shape<D>,
    rhs_shape: &Shape<D>,
    config: &Tiling2dConfig,
) -> bool {
    lhs_shape.dims[D - 2] % config.block_size_m != 0
        || lhs_shape.dims[D - 1] % config.block_size_k != 0
        || rhs_shape.dims[D - 1] % config.block_size_n != 0
}
