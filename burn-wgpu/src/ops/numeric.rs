use std::marker::PhantomData;
use std::sync::Arc;

use burn_tensor::{Element, ElementConversion, Shape};

use crate::kernel::{
    binary_elemwise, binary_elemwise_inplace, reduction_mean_dim, reduction_sum, reduction_sum_dim,
    unary_scalar, unary_scalar_inplace,
};
use crate::pool::get_context;
use crate::{
    binary_elemwise, binary_elemwise_inplace, element::WgpuElement, tensor::WgpuTensor,
    unary_scalar, unary_scalar_inplace,
};
use crate::{GraphicsApi, WgpuDevice};

pub struct NumericOps<G: GraphicsApi> {
    _g: PhantomData<G>,
}

impl<G: GraphicsApi> NumericOps<G> {
    pub fn zeros<E: WgpuElement, const D: usize>(
        shape: Shape<D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        let context = get_context::<G>(device);

        let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

        WgpuTensor::new(context, shape, Arc::new(buffer))
    }

    pub fn ones<E: WgpuElement + Element, const D: usize>(
        shape: Shape<D>,
        device: &WgpuDevice,
    ) -> WgpuTensor<E, D> {
        let context = get_context::<G>(device);

        let buffer = context.create_buffer(shape.num_elements() * core::mem::size_of::<E>());

        Self::add_scalar(
            WgpuTensor::new(context, shape, Arc::new(buffer)),
            1i32.elem::<E>(),
        )
    }

    pub fn add<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        binary_elemwise!(Add, "+");
        binary_elemwise_inplace!(AddInplace, "+");

        if lhs.can_mut_broadcast(&rhs) {
            return binary_elemwise_inplace::<AddInplace, E, D>(lhs, rhs);
        }

        if rhs.can_mut_broadcast(&lhs) {
            return binary_elemwise_inplace::<AddInplace, E, D>(rhs, lhs);
        }

        binary_elemwise::<Add, E, D>(lhs, rhs)
    }

    pub fn add_scalar<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<E, D> {
        unary_scalar!(AddScalar, ops "+");
        unary_scalar_inplace!(AddScalarInplace, ops "+");

        if lhs.can_mut() {
            return unary_scalar_inplace::<AddScalarInplace, E, D>(lhs, rhs);
        }

        unary_scalar::<AddScalar, E, D>(lhs, rhs)
    }

    pub fn sub<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        binary_elemwise!(Sub, "-");
        binary_elemwise_inplace!(SubInplace, "-");

        if lhs.can_mut_broadcast(&rhs) {
            return binary_elemwise_inplace::<SubInplace, E, D>(lhs, rhs);
        }

        binary_elemwise::<Sub, E, D>(lhs, rhs)
    }

    pub fn sub_scalar<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<E, D> {
        unary_scalar!(SubScalar, ops "-");
        unary_scalar_inplace!(SubScalarInplace, ops "-");

        if lhs.can_mut() {
            return unary_scalar_inplace::<SubScalarInplace, E, D>(lhs, rhs);
        }

        unary_scalar::<SubScalar, E, D>(lhs, rhs)
    }

    pub fn mul<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        binary_elemwise!(Mul, "*");
        binary_elemwise_inplace!(MulInplace, "*");

        if lhs.can_mut_broadcast(&rhs) {
            return binary_elemwise_inplace::<MulInplace, E, D>(lhs, rhs);
        }

        if rhs.can_mut_broadcast(&lhs) {
            return binary_elemwise_inplace::<MulInplace, E, D>(rhs, lhs);
        }

        binary_elemwise::<Mul, E, D>(lhs, rhs)
    }

    pub fn mul_scalar<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<E, D> {
        unary_scalar!(MulScalar, ops "*");
        unary_scalar_inplace!(MulScalarInplace, ops "*");

        if lhs.can_mut() {
            return unary_scalar_inplace::<MulScalarInplace, E, D>(lhs, rhs);
        }

        unary_scalar::<MulScalar, E, D>(lhs, rhs)
    }

    pub fn div<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, D> {
        binary_elemwise!(Div, "/");
        binary_elemwise_inplace!(DivInplace, "/");

        if lhs.can_mut_broadcast(&rhs) {
            return binary_elemwise_inplace::<DivInplace, E, D>(lhs, rhs);
        }

        binary_elemwise::<Div, E, D>(lhs, rhs)
    }

    pub fn div_scalar<E: WgpuElement, const D: usize>(
        lhs: WgpuTensor<E, D>,
        rhs: E,
    ) -> WgpuTensor<E, D> {
        unary_scalar!(DivScalar, ops "/");
        unary_scalar_inplace!(DivScalarInplace, ops "/");

        if lhs.can_mut() {
            return unary_scalar_inplace::<DivScalarInplace, E, D>(lhs, rhs);
        }

        unary_scalar::<DivScalar, E, D>(lhs, rhs)
    }

    pub fn sum<E: WgpuElement + Element, const D: usize>(
        tensor: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, 1> {
        reduction_sum(tensor)
    }

    pub fn sum_dim<E: WgpuElement + Element, const D: usize>(
        tensor: WgpuTensor<E, D>,
        dim: usize,
    ) -> WgpuTensor<E, D> {
        reduction_sum_dim(tensor, dim)
    }

    pub fn mean<E: WgpuElement + Element, const D: usize>(
        tensor: WgpuTensor<E, D>,
    ) -> WgpuTensor<E, 1> {
        let num_elems = tensor.shape.num_elements();
        Self::div_scalar(reduction_sum(tensor), (num_elems as f32).elem())
    }

    pub fn mean_dim<E: WgpuElement + Element, const D: usize>(
        tensor: WgpuTensor<E, D>,
        dim: usize,
    ) -> WgpuTensor<E, D> {
        reduction_mean_dim(tensor, dim)
    }
}
