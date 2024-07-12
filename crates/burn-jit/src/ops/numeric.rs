use crate::kernel::{launch_unary, unary_op, UnaryOp};
use crate::{binary, JitRuntime};
use crate::{element::JitElement, tensor::JitTensor};
use cubecl::client::ComputeClient;
use burn_tensor::{ElementConversion, Shape};
use cubecl::ir::{BinaryOperator, Elem, Operator, Scope, Variable};
use cubecl::{calculate_cube_count_elemwise, prelude::*, SUBCUBE_DIM_APPROX};
use cubecl::{tensor_vectorization_factor, Runtime};

pub fn full<R: JitRuntime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    full_device::<R, E, D>(client, shape, device.clone(), value)
}

pub fn full_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    shape: Shape<D>,
    device: R::Device,
    value: E,
) -> JitTensor<R, E, D> {
    let empty = empty_device(client, device, shape);

    #[cube(launch)]
    pub fn full_kernel<C: Numeric + Vectorized>(tensor: &mut Tensor<C>, value: C) {
        if ABSOLUTE_POS >= tensor.len() {
            return;
        }

        tensor[ABSOLUTE_POS] = value;
    }

    let num_elems = empty.shape.num_elements();
    let vectorization_factor =
        tensor_vectorization_factor(&[4, 2], &empty.shape.dims, &empty.strides, D - 1);
    let cube_count = calculate_cube_count_elemwise(
        num_elems / vectorization_factor as usize,
        SUBCUBE_DIM_APPROX,
    );

    full_kernel::launch::<E::Primitive, R>(
        empty.client.clone(),
        cube_count,
        CubeDim::default(),
        TensorArg::vectorized(
            vectorization_factor,
            &empty.handle,
            &empty.strides,
            &empty.shape.dims,
        ),
        ScalarArg::new(value),
    );

    empty
}

pub fn zeros<R: JitRuntime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    zeros_device(client, device.clone(), shape)
}

pub fn zeros_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 0.elem())
}

pub fn ones<R: JitRuntime, E: JitElement, const D: usize>(
    shape: Shape<D>,
    device: &R::Device,
) -> JitTensor<R, E, D> {
    let client = R::client(device);

    ones_device::<R, E, D>(client, device.clone(), shape)
}

pub fn ones_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    full_device::<R, E, D>(client, shape, device, 1.elem())
}

pub fn empty_device<R: JitRuntime, E: JitElement, const D: usize>(
    client: ComputeClient<R::Server, R::Channel>,
    device: R::Device,
    shape: Shape<D>,
) -> JitTensor<R, E, D> {
    let buffer = client.empty(shape.num_elements() * core::mem::size_of::<E>());

    JitTensor::new_contiguous(client, device, shape, buffer)
}

pub fn add<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Add(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn add_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary_op!(numeric(lhs, rhs) => |context, lhs, rhs| {
        #[cube]
        fn execute<C: Numeric>(lhs: C, rhs: C) -> C {
            lhs + rhs
        }
        execute::__expand::<C>(context, lhs, rhs)
    })
}

pub fn sub<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Sub(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn sub_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary_op!(numeric(lhs, rhs) => |context, lhs, rhs| {
        #[cube]
        fn execute<C: Numeric>(lhs: C, rhs: C) -> C {
            lhs - rhs
        }
        execute::__expand::<C>(context, lhs, rhs)
    })
}

pub fn mul<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Mul(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn mul_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary_op!(numeric(lhs, rhs) => |context, lhs, rhs| {
        #[cube]
        fn execute<C: Numeric>(lhs: C, rhs: C) -> C {
            lhs * rhs
        }
        execute::__expand::<C>(context, lhs, rhs)
    })
}

pub fn div<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Div(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}

pub fn div_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    unary_op!(numeric(lhs, rhs) => |context, lhs, rhs| {
        #[cube]
        fn execute<C: Numeric>(lhs: C, rhs: C) -> C {
            lhs / rhs
        }
        execute::__expand::<C>(context, lhs, rhs)
    })
}

pub fn remainder_scalar<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: E,
) -> JitTensor<R, E, D> {
    let shape = lhs.shape.clone();
    let device = lhs.device.clone();

    let rhs_tensor = full::<R, E, D>(shape, &device, rhs);

    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Remainder(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs_tensor,
        elem: E
    )
}

pub fn pow<R: JitRuntime, E: JitElement, const D: usize>(
    lhs: JitTensor<R, E, D>,
    rhs: JitTensor<R, E, D>,
) -> JitTensor<R, E, D> {
    binary!(
        operation: |scope: &mut Scope, elem: Elem, position: Variable| Operator::Powf(BinaryOperator {
            lhs: scope.read_array(0, elem, position),
            rhs: scope.read_array(1, elem, position),
            out: scope.create_local(elem),
        }),
        runtime: R,
        input: lhs; rhs,
        elem: E
    )
}
