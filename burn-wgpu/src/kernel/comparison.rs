use crate::{
    binary,
    codegen::{execute_static, Elem, GridLaunch, Operator, StaticHandle, Variable},
    element::WgpuElement,
    kernel::StaticKernelSource,
    tensor::WgpuTensor,
    unary,
};
use burn_tensor::Shape;
use std::mem;

macro_rules! comparison {
    (
        binary: $ops:expr,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operator: $ops, elem_in: $elem, elem_out: $elem);

        launch_binary::<Ops<E, u32>, OpsInplaceLhs<E, u32>, OpsInplaceRhs<E, u32>, E, D>($lhs, $rhs)
    }};

    (
        unary: $ops:expr,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        unary!($ops, scalar 1);

        launch_unary::<Ops<E>, OpsInplace<E>, E, D>($lhs, $rhs)
    }};
}

pub fn equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::Equal {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::Greater {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::GreaterEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::Lower {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D> {
    comparison!(
        binary: |elem: Elem| Operator::LowerEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Input(1, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::Equal {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::Greater {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::Lower {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::GreaterEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal_elem<E: WgpuElement, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: E,
) -> WgpuTensor<u32, D> {
    comparison!(
        unary: |elem: Elem| Operator::LowerEqual {
            lhs: Variable::Input(0, elem),
            rhs: Variable::Scalar(0, elem),
            out: Variable::Local(0, Elem::Bool),
        },
        input: lhs; rhs,
        elem: E
    )
}

/// Launch an binary comparison operation.
fn launch_binary<Kernel, KernelInplaceLhs, KernelInplaceRhs, E, const D: usize>(
    lhs: WgpuTensor<E, D>,
    rhs: WgpuTensor<E, D>,
) -> WgpuTensor<u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplaceLhs: StaticKernelSource,
    KernelInplaceRhs: StaticKernelSource,
    E: WgpuElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && lhs.can_mut_broadcast(&rhs) {
        execute_static::<KernelInplaceLhs, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[],
            None,
            GridLaunch::Input { pos: 0 },
            rhs.client,
        );

        WgpuTensor::new(lhs.client, lhs.device, lhs.shape, lhs.handle)
    } else if can_be_used_as_bool && rhs.can_mut_broadcast(&lhs) {
        execute_static::<KernelInplaceRhs, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[],
            None,
            GridLaunch::Input { pos: 1 },
            lhs.client,
        );

        WgpuTensor::new(rhs.client, rhs.device, rhs.shape, rhs.handle)
    } else {
        let mut shape_out = [0; D];
        lhs.shape
            .dims
            .iter()
            .zip(rhs.shape.dims.iter())
            .enumerate()
            .for_each(|(index, (dim_lhs, dim_rhs))| {
                shape_out[index] = usize::max(*dim_lhs, *dim_rhs);
            });

        let shape_out = Shape::new(shape_out);
        let num_elems = shape_out.num_elements();
        let buffer = lhs.client.empty(num_elems * core::mem::size_of::<E>());
        let out = WgpuTensor::new(lhs.client.clone(), lhs.device, shape_out, buffer);

        execute_static::<Kernel, E>(
            &[
                StaticHandle::new(&lhs.handle, &lhs.strides, &lhs.shape.dims),
                StaticHandle::new(&rhs.handle, &rhs.strides, &rhs.shape.dims),
            ],
            &[StaticHandle::new(
                &out.handle,
                &out.strides,
                &out.shape.dims,
            )],
            None,
            GridLaunch::Output { pos: 0 },
            lhs.client,
        );

        out
    }
}

fn launch_unary<Kernel, KernelInplace, E, const D: usize>(
    tensor: WgpuTensor<E, D>,
    scalars: E,
) -> WgpuTensor<u32, D>
where
    Kernel: StaticKernelSource,
    KernelInplace: StaticKernelSource,
    E: WgpuElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    if can_be_used_as_bool && tensor.can_mut() {
        execute_static::<KernelInplace, E>(
            &[StaticHandle::new(
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            )],
            &[],
            Some(&[scalars]),
            GridLaunch::Input { pos: 0 },
            tensor.client.clone(),
        );

        WgpuTensor::new(tensor.client, tensor.device, tensor.shape, tensor.handle)
    } else {
        let num_elems = tensor.shape.num_elements();
        let buffer = tensor.client.empty(num_elems * core::mem::size_of::<E>());
        let output = WgpuTensor::new(
            tensor.client.clone(),
            tensor.device,
            tensor.shape.clone(),
            buffer,
        );

        execute_static::<Kernel, E>(
            &[StaticHandle::new(
                &tensor.handle,
                &tensor.strides,
                &tensor.shape.dims,
            )],
            &[StaticHandle::new(
                &output.handle,
                &output.strides,
                &output.shape.dims,
            )],
            Some(&[scalars]),
            GridLaunch::Output { pos: 0 },
            tensor.client,
        );

        output
    }
}
