use crate::{
    binary,
    element::JitElement,
    kernel::{binary::binary, unary::unary},
    tensor::JitTensor,
    unary, JitRuntime,
};
use burn_cube::{
    ir::{BinaryOperator, Elem, Operator, Scope, Variable},
    Runtime,
};
use std::mem;

use super::Kernel;

macro_rules! comparison {
    (
        binary: $ops:expr,
        runtime: $runtime:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        binary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler, elem_in: $elem, elem_out: u32);

        launch_binary::<
            Ops<<$runtime as Runtime>::Compiler, E, u32>,
            OpsInplaceLhs<<$runtime as Runtime>::Compiler, E, u32>,
            OpsInplaceRhs<<$runtime as Runtime>::Compiler, E, u32>,
            $runtime,
            E,
            D
        >($lhs, $rhs, Ops::new(), OpsInplaceLhs::new(), OpsInplaceRhs::new())
    }};

    (
        unary: $ops:expr,
        runtime: $runtime:ty,
        input: $lhs:expr; $rhs:expr,
        elem: $elem:ty
    ) => {{
        unary!(operation: $ops, compiler: <$runtime as Runtime>::Compiler, scalar 1);

        let kernel = Ops::<<$runtime as Runtime>::Compiler, E>::new();
        let kernel_inplace = OpsInplace::<<$runtime as Runtime>::Compiler, E>::new();

        launch_unary::<
            Ops<<$runtime as Runtime>::Compiler, E>,
            OpsInplace<<$runtime as Runtime>::Compiler, E>,
            $runtime,
            E,
            D
        >($lhs, $rhs, kernel, kernel_inplace)
    }};
}

pub fn equal<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Equal(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Greater(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::GreaterEqual(BinaryOperator {
            lhs: scope.read_array(0, elem,position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Lower(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, u32, D> {
    comparison!(
        binary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::LowerEqual(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn equal_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Equal(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Greater(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Lower(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn greater_equal_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::GreaterEqual(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn lower_equal_elem<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, u32, D> {
    comparison!(
        unary: |scope: &mut Scope, elem: Elem, position: Variable| Operator::LowerEqual(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_scalar(0, elem),
            out: scope.create_local(Elem::Bool),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

fn launch_binary<K, KinplaceLhs, KernelInplaceRhs, R: JitRuntime, E, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
    kernel: K,
    kernel_inplace_lhs: KinplaceLhs,
    kernel_inplace_rhs: KernelInplaceRhs,
) -> JitTensor<R, u32, D>
where
    K: Kernel,
    KinplaceLhs: Kernel,
    KernelInplaceRhs: Kernel,
    E: JitElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output = binary::<K, KinplaceLhs, KernelInplaceRhs, R, E, D>(
        lhs,
        rhs,
        can_be_used_as_bool,
        kernel,
        kernel_inplace_lhs,
        kernel_inplace_rhs,
    );

    // We recast the tensor type.
    JitTensor::new(output.client, output.device, output.shape, output.handle)
}

fn launch_unary<K, Kinplace, R: JitRuntime, E, const D: usize>(
    tensor: JitTensor<R, E, D>,
    scalars: E,
    kernel: K,
    kernel_inplace: Kinplace,
) -> JitTensor<R, u32, D>
where
    K: Kernel,
    Kinplace: Kernel,
    E: JitElement,
{
    let can_be_used_as_bool = mem::size_of::<E>() == mem::size_of::<u32>();

    let output = unary::<K, Kinplace, R, E, D>(
        tensor,
        Some(&[scalars]),
        can_be_used_as_bool,
        kernel,
        kernel_inplace,
    );

    // We recast the tensor type.
    JitTensor::new(output.client, output.device, output.shape, output.handle)
}
