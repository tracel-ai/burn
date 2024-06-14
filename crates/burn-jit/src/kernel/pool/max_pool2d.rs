use burn_cube::{
    cpa,
    frontend::TensorHandle,
    ir::{Elem, Item, Scope, Variable},
    CubeCountSettings, Execution,
};
use std::{fmt::Debug, marker::PhantomData};

use crate::{element::JitElement, ops::numeric::empty_device, tensor::JitTensor, JitRuntime};
use burn_tensor::{ops::conv::calculate_pool_output_size, Shape};

use super::{Pool2dEagerKernel, PoolStrategy};

#[derive(Default, Debug, Clone)]
struct MaxPool<E: JitElement> {
    _elem: PhantomData<E>,
}

impl<E: JitElement> PoolStrategy for MaxPool<E> {
    type Accumulator = Variable;

    fn initialize(&self, scope: &mut Scope, item: Item) -> Self::Accumulator {
        let max_val = scope.create_local(item);
        let max_initial = Variable::ConstantScalar(E::minimum_value().to_f64(), item.elem());
        cpa!(scope, max_val = max_initial);
        max_val
    }

    fn process_result(
        &self,
        scope: &mut Scope,
        accumulator: Self::Accumulator,
        result: Variable,
        _idx: Variable,
    ) -> Self::Accumulator {
        let is_max = scope.create_local(Elem::Bool);
        cpa!(scope, is_max = result > accumulator);
        cpa!(scope, if(is_max).then(|scope|{
            cpa!(scope, accumulator = result);
        }));
        accumulator
    }

    fn assign(
        &self,
        scope: &mut Scope,
        id: Variable,
        output: Variable,
        _indices: Option<Variable>,
        accumulator: Self::Accumulator,
    ) {
        cpa!(scope, output[id] = accumulator);
    }

    fn with_indices() -> bool {
        false
    }
}

#[derive(Default, Debug, Clone)]
struct MaxPoolWithIndices<E: JitElement> {
    _elem: PhantomData<E>,
}

impl<E: JitElement> PoolStrategy for MaxPoolWithIndices<E> {
    type Accumulator = (Variable, Variable);

    fn initialize(&self, scope: &mut Scope, item: Item) -> Self::Accumulator {
        let max_val = scope.create_local(item);
        let max_initial = Variable::ConstantScalar(E::minimum_value().to_f64(), item.elem());
        cpa!(scope, max_val = max_initial);
        let max_index = scope.create_local(Elem::UInt);
        (max_val, max_index)
    }

    fn process_result(
        &self,
        scope: &mut Scope,
        (max_val, max_index): Self::Accumulator,
        result: Variable,
        idx: Variable,
    ) -> Self::Accumulator {
        let is_max = scope.create_local(Elem::Bool);
        cpa!(scope, is_max = result > max_val);
        cpa!(scope, if(is_max).then(|scope|{
            cpa!(scope, max_val = result);
            cpa!(scope, max_index = idx);
        }));
        (max_val, max_index)
    }

    fn assign(
        &self,
        scope: &mut Scope,
        id: Variable,
        output: Variable,
        indices: Option<Variable>,
        (max_val, max_index): Self::Accumulator,
    ) {
        let indices = indices.unwrap();
        cpa!(scope, output[id] = max_val);
        cpa!(scope, indices[id] = max_index);
    }

    fn with_indices() -> bool {
        true
    }
}

pub(crate) fn max_pool2d<R: JitRuntime, E: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> JitTensor<R, E, 4> {
    let [batch_size, channels, _, _] = x.shape.dims;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape.dims[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape.dims[3],
    );

    let shape_out = Shape::new([batch_size, channels, size_0, size_1]);
    let output = empty_device(x.client.clone(), x.device.clone(), shape_out);

    let kernel = Pool2dEagerKernel::<MaxPool<E>, R, E>::new(kernel_size, MaxPool::default());

    Execution::start(kernel, x.client)
        .inputs(&[TensorHandle::<R>::new(&x.handle, &x.strides, &x.shape.dims)])
        .outputs(&[TensorHandle::new(
            &output.handle,
            &output.strides,
            &output.shape.dims,
        )])
        .with_scalars(&[
            stride[0] as u32,
            stride[1] as u32,
            dilation[0] as u32,
            dilation[1] as u32,
            padding[0] as u32,
            padding[1] as u32,
        ])
        .execute(CubeCountSettings::Output { pos: 0 });

    output
}

pub(crate) fn max_pool2d_with_indices<R: JitRuntime, E: JitElement, I: JitElement>(
    x: JitTensor<R, E, 4>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> (JitTensor<R, E, 4>, JitTensor<R, I, 4>) {
    let [batch_size, channels, _, _] = x.shape.dims;

    let size_0 = calculate_pool_output_size(
        kernel_size[0],
        stride[0],
        padding[0],
        dilation[0],
        x.shape.dims[2],
    );
    let size_1 = calculate_pool_output_size(
        kernel_size[1],
        stride[1],
        padding[1],
        dilation[1],
        x.shape.dims[3],
    );

    let shape_out = Shape::new([batch_size, channels, size_0, size_1]);
    let output = empty_device(x.client.clone(), x.device.clone(), shape_out.clone());
    let indices = empty_device(x.client.clone(), x.device.clone(), shape_out);

    let kernel = Pool2dEagerKernel::<MaxPoolWithIndices<E>, R, E>::new(
        kernel_size,
        MaxPoolWithIndices::default(),
    );

    Execution::start(kernel, x.client)
        .inputs(&[TensorHandle::<R>::new(&x.handle, &x.strides, &x.shape.dims)])
        .outputs(&[
            TensorHandle::new(&output.handle, &output.strides, &output.shape.dims),
            TensorHandle::new(&indices.handle, &indices.strides, &indices.shape.dims),
        ])
        .with_scalars(&[
            stride[0] as i32,
            stride[1] as i32,
            dilation[0] as i32,
            dilation[1] as i32,
            padding[0] as i32,
            padding[1] as i32,
        ])
        .execute(CubeCountSettings::Output { pos: 0 });

    (output, indices)
}
