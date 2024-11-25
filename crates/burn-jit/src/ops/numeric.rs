use crate::kernel::{
    launch_binop, launch_scalar_binop, AddOp, DivOp, MulOp, PowOp, RemainderOp, SubOp,
};
use crate::{element::JitElement, tensor::JitTensor};
use crate::{FloatElement, JitRuntime};

pub fn add<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, AddOp>(lhs, rhs)
}

pub fn add_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, AddOp>(lhs, rhs)
}

pub fn sub<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, SubOp>(lhs, rhs)
}

pub fn sub_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, SubOp>(lhs, rhs)
}

pub fn mul<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, MulOp>(lhs, rhs)
}

pub fn mul_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, MulOp>(lhs, rhs)
}

pub fn div<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, DivOp>(lhs, rhs)
}

pub fn div_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, DivOp>(lhs, rhs)
}

pub fn remainder<R: JitRuntime, E: JitElement>(
    lhs: JitTensor<R>,
    rhs: JitTensor<R>,
) -> JitTensor<R> {
    launch_binop::<R, E, RemainderOp>(lhs, rhs)
}

pub fn remainder_scalar<R: JitRuntime, E: JitElement>(lhs: JitTensor<R>, rhs: E) -> JitTensor<R> {
    launch_scalar_binop::<R, E, RemainderOp>(lhs, rhs)
}

pub fn pow<R: JitRuntime, E: FloatElement>(lhs: JitTensor<R>, rhs: JitTensor<R>) -> JitTensor<R> {
    launch_binop::<R, E, PowOp>(lhs, rhs)
}
