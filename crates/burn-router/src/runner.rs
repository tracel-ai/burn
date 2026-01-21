use super::{RouterTensor, RunnerClient};
use crate::{
    binary_bool_ops, binary_float_cmp_ops, binary_float_ops, binary_int_cmp_ops, binary_int_ops,
    reduce_float_dim_ops, reduce_float2int_dim_ops, reduce_int_dim_ops, scalar_float_cmp_ops,
    scalar_float_ops, scalar_int_cmp_ops, scalar_int_ops, unary_float_ops, unary_int_ops,
};
use alloc::sync::Arc;
use burn_backend::{Backend, DType, ExecutionError, Shape, TensorData, tensor::IndexingUpdateOp};
use burn_ir::{
    BackendIr, BaseOperationIr, BoolOperationIr, FloatOperationIr, HandleContainer, IntOperationIr,
    ModuleOperationIr, NumericOperationIr, OperationIr, TensorId, TensorIr, TensorStatus,
};
use burn_std::{future::DynFut, stub::Mutex};

/// A runner's context contains a [handle container](HandleContainer) to manage
/// (i.e., fetch and update) existing tensors.
pub struct RunnerContext<B: BackendIr> {
    /// Handle container to retrieve tensors based on their intermediate representation.
    handles: HandleContainer<B::Handle>,
}

impl<B: BackendIr> RunnerContext<B> {
    /// Create a new (uninitialized) empty tensor and returns its corresponding [tensor id](TensorId).
    fn create_empty_handle(&mut self) -> TensorId {
        self.handles.create_tensor_uninit()
    }
}

/// A runner is responsible for executing tensor operations for a given [intermediate backend](BackendIr).
#[derive(Clone)]
pub struct Runner<B: BackendIr> {
    // Mutex for the mutable handles
    context: Arc<Mutex<RunnerContext<B>>>,
    device: B::Device,
}

impl<B: BackendIr> Runner<B> {
    /// Create a new runner.
    pub fn new(device: B::Device) -> Self {
        Self {
            context: Arc::new(Mutex::new(RunnerContext {
                handles: HandleContainer::new(),
            })),
            device,
        }
    }

    /// Get the tensor handle for the given [tensor representation](TensorIr).
    pub(crate) fn get_tensor_handle(&self, tensor: &TensorIr) -> B::Handle {
        let handles = &mut self.context.lock().unwrap().handles;
        handles.get_tensor_handle(tensor).handle
    }

    /// Create a tensor with the given handle and shape.
    pub(crate) fn register_tensor<C: RunnerClient>(
        &self,
        handle: B::Handle,
        shape: Shape,
        dtype: DType,
        client: C,
    ) -> RouterTensor<C> {
        let mut ctx = self.context.lock().unwrap();
        let id = ctx.create_empty_handle();

        ctx.handles.register_handle(id, handle);
        core::mem::drop(ctx);

        RouterTensor::new(id, shape, dtype, client)
    }

    /// Register a tensor from its data and id.
    pub fn register_tensor_data_id(&self, id: TensorId, data: TensorData) {
        let mut ctx = self.context.lock().unwrap();
        let dtype = data.dtype;

        if dtype.is_float() {
            let tensor = B::float_from_data(data, &self.device);
            ctx.handles.register_float_tensor::<B>(&id, tensor)
        } else if dtype.is_int() {
            let tensor = B::int_from_data(data, &self.device);
            ctx.handles.register_int_tensor::<B>(&id, tensor)
        } else if dtype.is_bool() {
            let tensor = B::bool_from_data(data, &self.device);
            ctx.handles.register_bool_tensor::<B>(&id, tensor)
        } else if let DType::QFloat(_) = dtype {
            todo!();
        }

        core::mem::drop(ctx);
    }

    /// Register a tensor and returns its intermediate representation.
    pub fn register_tensor_data_desc(&self, data: TensorData) -> TensorIr {
        let mut ctx = self.context.lock().unwrap();
        let id = ctx.create_empty_handle();
        let shape = data.shape.clone();
        let dtype = data.dtype;

        if dtype.is_float() {
            let tensor = B::float_from_data(data, &self.device);
            ctx.handles.register_float_tensor::<B>(&id, tensor)
        } else if dtype.is_int() {
            let tensor = B::int_from_data(data, &self.device);
            ctx.handles.register_int_tensor::<B>(&id, tensor)
        } else if dtype.is_bool() {
            let tensor = B::bool_from_data(data, &self.device);
            ctx.handles.register_bool_tensor::<B>(&id, tensor)
        } else if let DType::QFloat(_) = dtype {
            todo!();
        }

        core::mem::drop(ctx);

        TensorIr {
            id,
            shape: Shape::from(shape),
            status: TensorStatus::ReadWrite,
            dtype,
        }
    }
}

impl<B: BackendIr> RunnerClient for Runner<B> {
    type Device = B::Device;

    /// Execute a tensor operation.
    fn register_op(&self, op: OperationIr) {
        // Remove unused tensor handles
        let mut ctx = self.context.lock().unwrap();

        let handles = &mut ctx.handles;
        match &op {
            // For every op: get the input(s), execute the operation and register the output(s)
            OperationIr::BaseFloat(op) => match op {
                BaseOperationIr::Reshape(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_reshape(tensor, desc.out.shape.clone());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SwapDims(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_swap_dims(tensor, desc.dim1, desc.dim2);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Permute(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_permute(tensor, &desc.axes);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Flip(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_flip(tensor, &desc.axes);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Expand(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_expand(tensor, desc.out.shape.clone());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Unfold(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_unfold(tensor, desc.dim, desc.size, desc.step);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Slice(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let output = B::float_slice(tensor, &desc.ranges);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SliceAssign(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output = B::float_slice_assign(tensor, &desc.ranges, value);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Gather(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::float_gather(desc.dim, tensor, indices);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Scatter(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output = match desc.update {
                        IndexingUpdateOp::Add => {
                            B::float_scatter_add(desc.dim, tensor, indices, value)
                        }
                    };
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Select(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::float_select(tensor, desc.dim, indices);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SelectAssign(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output = match desc.update {
                        IndexingUpdateOp::Add => {
                            B::float_select_add(tensor, desc.dim, indices, value)
                        }
                    };
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::MaskWhere(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let mask = handles.get_bool_tensor::<B>(&desc.mask);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output = B::float_mask_where(tensor, mask, value);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::MaskFill(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let mask = handles.get_bool_tensor::<B>(&desc.mask);

                    let output = B::float_mask_fill(tensor, mask, desc.value.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Equal(desc) => {
                    binary_float_cmp_ops!(handles, desc, B::float_equal)
                }
                BaseOperationIr::EqualElem(desc) => {
                    scalar_float_cmp_ops!(handles, desc, B::float_equal_elem)
                }
                BaseOperationIr::RepeatDim(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let output = B::float_repeat_dim(tensor, desc.dim, desc.times);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Cat(desc) => {
                    let tensors = desc
                        .tensors
                        .iter()
                        .map(|tensor| handles.get_float_tensor::<B>(tensor))
                        .collect();

                    let output = B::float_cat(tensors, desc.dim);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Cast(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_cast(tensor, desc.out.dtype.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Empty(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::float_empty(shape, &self.device, desc.out.dtype.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Ones(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::float_ones(shape, &self.device, desc.out.dtype.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Zeros(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::float_zeros(shape, &self.device, desc.out.dtype.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::BaseInt(op) => match op {
                BaseOperationIr::Reshape(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_reshape(tensor, desc.out.shape.clone());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SwapDims(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_swap_dims(tensor, desc.dim1, desc.dim2);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Permute(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_permute(tensor, &desc.axes);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Flip(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_flip(tensor, &desc.axes);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Expand(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_expand(tensor, desc.out.shape.clone());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Unfold(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_unfold(tensor, desc.dim, desc.size, desc.step);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Slice(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let output = B::int_slice(tensor, &desc.ranges);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SliceAssign(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let value = handles.get_int_tensor::<B>(&desc.value);

                    let output = B::int_slice_assign(tensor, &desc.ranges, value);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Gather(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::int_gather(desc.dim, tensor, indices);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Scatter(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_int_tensor::<B>(&desc.value);

                    let output = match desc.update {
                        IndexingUpdateOp::Add => {
                            B::int_scatter_add(desc.dim, tensor, indices, value)
                        }
                    };
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Select(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::int_select(tensor, desc.dim, indices);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SelectAssign(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_int_tensor::<B>(&desc.value);

                    let output = match desc.update {
                        IndexingUpdateOp::Add => {
                            B::int_select_add(tensor, desc.dim, indices, value)
                        }
                    };
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::MaskWhere(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let mask = handles.get_bool_tensor::<B>(&desc.mask);
                    let value = handles.get_int_tensor::<B>(&desc.value);

                    let output = B::int_mask_where(tensor, mask, value);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::MaskFill(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);
                    let mask = handles.get_bool_tensor::<B>(&desc.mask);

                    let output = B::int_mask_fill(tensor, mask, desc.value.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Equal(desc) => {
                    binary_int_cmp_ops!(handles, desc, B::int_equal)
                }
                BaseOperationIr::EqualElem(desc) => {
                    scalar_int_cmp_ops!(handles, desc, B::int_equal_elem)
                }
                BaseOperationIr::RepeatDim(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let output = B::int_repeat_dim(tensor, desc.dim, desc.times);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Cat(desc) => {
                    let tensors = desc
                        .tensors
                        .iter()
                        .map(|tensor| handles.get_int_tensor::<B>(tensor))
                        .collect();

                    let output = B::int_cat(tensors, desc.dim);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Cast(_) => unreachable!(),
                BaseOperationIr::Empty(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::int_empty(shape, &self.device, desc.out.dtype.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Ones(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::int_ones(shape, &self.device, desc.out.dtype.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Zeros(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::int_zeros(shape, &self.device, desc.out.dtype.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::BaseBool(op) => match op {
                BaseOperationIr::Reshape(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_reshape(tensor, desc.out.shape.clone());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SwapDims(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_swap_dims(tensor, desc.dim1, desc.dim2);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Permute(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_permute(tensor, &desc.axes);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Flip(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_flip(tensor, &desc.axes);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Expand(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_expand(tensor, desc.out.shape.clone());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Unfold(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_unfold(tensor, desc.dim, desc.size, desc.step);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Slice(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);

                    let output = B::bool_slice(tensor, &desc.ranges);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SliceAssign(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let value = handles.get_bool_tensor::<B>(&desc.value);

                    let output = B::bool_slice_assign(tensor, &desc.ranges, value);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Gather(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::bool_gather(desc.dim, tensor, indices);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Scatter(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_bool_tensor::<B>(&desc.value);

                    let output = match desc.update {
                        IndexingUpdateOp::Add => {
                            B::bool_scatter_or(desc.dim, tensor, indices, value)
                        }
                    };
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Select(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::bool_select(tensor, desc.dim, indices);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SelectAssign(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_bool_tensor::<B>(&desc.value);

                    let output = match desc.update {
                        IndexingUpdateOp::Add => {
                            B::bool_select_or(tensor, desc.dim, indices, value)
                        }
                    };
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::MaskWhere(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let mask = handles.get_bool_tensor::<B>(&desc.mask);
                    let value = handles.get_bool_tensor::<B>(&desc.value);

                    let output = B::bool_mask_where(tensor, mask, value);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::MaskFill(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);
                    let mask = handles.get_bool_tensor::<B>(&desc.mask);

                    let output = B::bool_mask_fill(tensor, mask, desc.value.elem());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Equal(desc) => {
                    let lhs = handles.get_bool_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_bool_tensor::<B>(&desc.rhs);

                    let output = B::bool_equal(lhs, rhs);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::EqualElem(desc) => {
                    let lhs = handles.get_bool_tensor::<B>(&desc.lhs);

                    let output = B::bool_equal_elem(lhs, desc.rhs.elem());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::RepeatDim(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.tensor);

                    let output = B::bool_repeat_dim(tensor, desc.dim, desc.times);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Cat(desc) => {
                    let tensors = desc
                        .tensors
                        .iter()
                        .map(|tensor| handles.get_bool_tensor::<B>(tensor))
                        .collect();

                    let output = B::bool_cat(tensors, desc.dim);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Cast(_) => unreachable!(),
                BaseOperationIr::Empty(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::bool_empty(shape, &self.device);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Zeros(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::bool_zeros(shape, &self.device);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Ones(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::bool_ones(shape, &self.device);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::NumericFloat(_dtype, op) => match op {
                NumericOperationIr::Add(desc) => {
                    binary_float_ops!(handles, desc, B::float_add)
                }
                NumericOperationIr::AddScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_add_scalar)
                }
                NumericOperationIr::Sub(desc) => {
                    binary_float_ops!(handles, desc, B::float_sub)
                }
                NumericOperationIr::SubScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_sub_scalar)
                }
                NumericOperationIr::Div(desc) => {
                    binary_float_ops!(handles, desc, B::float_div)
                }
                NumericOperationIr::DivScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_div_scalar)
                }
                NumericOperationIr::Rem(desc) => {
                    binary_float_ops!(handles, desc, B::float_remainder)
                }
                NumericOperationIr::RemScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_remainder_scalar)
                }
                NumericOperationIr::Mul(desc) => {
                    binary_float_ops!(handles, desc, B::float_mul)
                }
                NumericOperationIr::MulScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_mul_scalar)
                }
                NumericOperationIr::Abs(desc) => {
                    unary_float_ops!(handles, desc, B::float_abs)
                }
                NumericOperationIr::Full(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::float_full(
                        shape,
                        desc.value.into(),
                        &self.device,
                        desc.out.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::MeanDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_mean_dim)
                }
                NumericOperationIr::Mean(desc) => {
                    unary_float_ops!(handles, desc, B::float_mean)
                }
                NumericOperationIr::Sum(desc) => {
                    unary_float_ops!(handles, desc, B::float_sum)
                }
                NumericOperationIr::SumDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_sum_dim)
                }
                NumericOperationIr::Prod(desc) => {
                    unary_float_ops!(handles, desc, B::float_prod)
                }
                NumericOperationIr::ProdDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_prod_dim)
                }
                NumericOperationIr::Greater(desc) => {
                    binary_float_cmp_ops!(handles, desc, B::float_greater)
                }
                NumericOperationIr::GreaterElem(desc) => {
                    scalar_float_cmp_ops!(handles, desc, B::float_greater_elem)
                }
                NumericOperationIr::GreaterEqual(desc) => {
                    binary_float_cmp_ops!(handles, desc, B::float_greater_equal)
                }
                NumericOperationIr::GreaterEqualElem(desc) => {
                    scalar_float_cmp_ops!(handles, desc, B::float_greater_equal_elem)
                }
                NumericOperationIr::Lower(desc) => {
                    binary_float_cmp_ops!(handles, desc, B::float_lower)
                }
                NumericOperationIr::LowerElem(desc) => {
                    scalar_float_cmp_ops!(handles, desc, B::float_lower_elem)
                }
                NumericOperationIr::LowerEqual(desc) => {
                    binary_float_cmp_ops!(handles, desc, B::float_lower_equal)
                }
                NumericOperationIr::LowerEqualElem(desc) => {
                    scalar_float_cmp_ops!(handles, desc, B::float_lower_equal_elem)
                }
                NumericOperationIr::ArgMax(desc) => {
                    reduce_float2int_dim_ops!(handles, desc, B::float_argmax)
                }
                NumericOperationIr::ArgMin(desc) => {
                    reduce_float2int_dim_ops!(handles, desc, B::float_argmin)
                }
                NumericOperationIr::Max(desc) => {
                    unary_float_ops!(handles, desc, B::float_max)
                }
                NumericOperationIr::MaxDimWithIndices(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::float_max_dim_with_indices(tensor, desc.dim);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::MinDimWithIndices(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::float_min_dim_with_indices(tensor, desc.dim);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::Min(desc) => {
                    unary_float_ops!(handles, desc, B::float_min)
                }
                NumericOperationIr::MaxDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_max_dim)
                }
                NumericOperationIr::MinDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_min_dim)
                }
                NumericOperationIr::MaxAbs(desc) => {
                    unary_float_ops!(handles, desc, B::float_max_abs)
                }
                NumericOperationIr::MaxAbsDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_max_abs_dim)
                }
                NumericOperationIr::Clamp(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let output = B::float_clamp(tensor, desc.min.into(), desc.max.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::IntRandom(_) => unreachable!(),
                NumericOperationIr::Powf(desc) => {
                    binary_float_ops!(handles, desc, B::float_powf)
                }
                NumericOperationIr::CumSum(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_cumsum(tensor, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumProd(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_cumprod(tensor, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumMin(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_cummin(tensor, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumMax(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_cummax(tensor, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::NumericInt(_dtype, op) => match op {
                NumericOperationIr::Add(desc) => {
                    binary_int_ops!(handles, desc, B::int_add)
                }
                NumericOperationIr::AddScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::int_add_scalar)
                }
                NumericOperationIr::Sub(desc) => {
                    binary_int_ops!(handles, desc, B::int_sub)
                }
                NumericOperationIr::SubScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::int_sub_scalar)
                }
                NumericOperationIr::Div(desc) => {
                    binary_int_ops!(handles, desc, B::int_div)
                }
                NumericOperationIr::DivScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::int_div_scalar)
                }
                NumericOperationIr::Rem(desc) => {
                    binary_int_ops!(handles, desc, B::int_remainder)
                }
                NumericOperationIr::RemScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::int_remainder_scalar)
                }
                NumericOperationIr::Mul(desc) => {
                    binary_int_ops!(handles, desc, B::int_mul)
                }
                NumericOperationIr::MulScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::int_mul_scalar)
                }
                NumericOperationIr::Abs(desc) => {
                    unary_int_ops!(handles, desc, B::int_abs)
                }
                NumericOperationIr::Full(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::int_full(
                        shape,
                        desc.value.into(),
                        &self.device,
                        desc.out.dtype.into(),
                    );
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::MeanDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_mean_dim)
                }
                NumericOperationIr::Mean(desc) => {
                    unary_int_ops!(handles, desc, B::int_mean)
                }
                NumericOperationIr::Sum(desc) => {
                    unary_int_ops!(handles, desc, B::int_sum)
                }
                NumericOperationIr::SumDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_sum_dim)
                }
                NumericOperationIr::Prod(desc) => {
                    unary_int_ops!(handles, desc, B::int_prod)
                }
                NumericOperationIr::ProdDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_prod_dim)
                }
                NumericOperationIr::Greater(desc) => {
                    binary_int_cmp_ops!(handles, desc, B::int_greater)
                }
                NumericOperationIr::GreaterElem(desc) => {
                    scalar_int_cmp_ops!(handles, desc, B::int_greater_elem)
                }
                NumericOperationIr::GreaterEqual(desc) => {
                    binary_int_cmp_ops!(handles, desc, B::int_greater_equal)
                }
                NumericOperationIr::GreaterEqualElem(desc) => {
                    scalar_int_cmp_ops!(handles, desc, B::int_greater_equal_elem)
                }
                NumericOperationIr::Lower(desc) => {
                    binary_int_cmp_ops!(handles, desc, B::int_lower)
                }
                NumericOperationIr::LowerElem(desc) => {
                    scalar_int_cmp_ops!(handles, desc, B::int_lower_elem)
                }
                NumericOperationIr::LowerEqual(desc) => {
                    binary_int_cmp_ops!(handles, desc, B::int_lower_equal)
                }
                NumericOperationIr::LowerEqualElem(desc) => {
                    scalar_int_cmp_ops!(handles, desc, B::int_lower_equal_elem)
                }
                NumericOperationIr::ArgMax(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_argmax)
                }
                NumericOperationIr::ArgMin(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_argmin)
                }
                NumericOperationIr::Max(desc) => {
                    unary_int_ops!(handles, desc, B::int_max)
                }
                NumericOperationIr::MaxDimWithIndices(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::int_max_dim_with_indices(tensor, desc.dim);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::MinDimWithIndices(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::int_min_dim_with_indices(tensor, desc.dim);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::Min(desc) => {
                    unary_int_ops!(handles, desc, B::int_min)
                }
                NumericOperationIr::MaxDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_max_dim)
                }
                NumericOperationIr::MinDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_min_dim)
                }
                NumericOperationIr::MaxAbs(desc) => {
                    unary_int_ops!(handles, desc, B::int_max_abs)
                }
                NumericOperationIr::MaxAbsDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_max_abs_dim)
                }
                NumericOperationIr::Clamp(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let output = B::int_clamp(tensor, desc.min.into(), desc.max.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::IntRandom(desc) => {
                    let shape = desc.out.shape.clone();

                    let output = B::int_random(shape, desc.distribution, &self.device);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::Powf(desc) => {
                    let lhs = handles.get_int_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_float_tensor::<B>(&desc.rhs);

                    let output = B::int_powf(lhs, rhs);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumSum(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_cumsum(tensor, desc.axis);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumProd(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_cumprod(tensor, desc.axis);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumMin(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_cummin(tensor, desc.axis);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::CumMax(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_cummax(tensor, desc.axis);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::Bool(op) => match op {
                BoolOperationIr::IntoFloat(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_into_float(tensor);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BoolOperationIr::IntoInt(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_into_int(tensor);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BoolOperationIr::Not(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_not(tensor);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BoolOperationIr::And(desc) => {
                    binary_bool_ops!(handles, desc, B::bool_and)
                }
                BoolOperationIr::Or(desc) => {
                    binary_bool_ops!(handles, desc, B::bool_or)
                }
            },
            OperationIr::Int(op) => match op {
                IntOperationIr::IntoFloat(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_into_float(tensor);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                IntOperationIr::Matmul(desc) => {
                    binary_int_ops!(handles, desc, B::int_matmul)
                }
                IntOperationIr::BitwiseAnd(desc) => {
                    binary_int_ops!(handles, desc, B::bitwise_and)
                }
                IntOperationIr::BitwiseAndScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::bitwise_and_scalar)
                }
                IntOperationIr::BitwiseOr(desc) => {
                    binary_int_ops!(handles, desc, B::bitwise_or)
                }
                IntOperationIr::BitwiseOrScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::bitwise_or_scalar)
                }
                IntOperationIr::BitwiseXor(desc) => {
                    binary_int_ops!(handles, desc, B::bitwise_xor)
                }
                IntOperationIr::BitwiseXorScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::bitwise_xor_scalar)
                }
                IntOperationIr::BitwiseNot(desc) => {
                    unary_int_ops!(handles, desc, B::bitwise_not)
                }
                IntOperationIr::BitwiseLeftShift(desc) => {
                    binary_int_ops!(handles, desc, B::bitwise_left_shift)
                }
                IntOperationIr::BitwiseRightShift(desc) => {
                    binary_int_ops!(handles, desc, B::bitwise_right_shift)
                }
                IntOperationIr::BitwiseLeftShiftScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::bitwise_left_shift_scalar)
                }
                IntOperationIr::BitwiseRightShiftScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::bitwise_right_shift_scalar)
                }
            },
            OperationIr::Float(_dtype, op) => match op {
                FloatOperationIr::Exp(desc) => {
                    unary_float_ops!(handles, desc, B::float_exp)
                }
                FloatOperationIr::Log(desc) => {
                    unary_float_ops!(handles, desc, B::float_log)
                }
                FloatOperationIr::Log1p(desc) => {
                    unary_float_ops!(handles, desc, B::float_log1p)
                }
                FloatOperationIr::Erf(desc) => {
                    unary_float_ops!(handles, desc, B::float_erf)
                }
                FloatOperationIr::PowfScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_powf_scalar)
                }
                FloatOperationIr::Sqrt(desc) => {
                    unary_float_ops!(handles, desc, B::float_sqrt)
                }
                FloatOperationIr::Cos(desc) => {
                    unary_float_ops!(handles, desc, B::float_cos)
                }
                FloatOperationIr::Sin(desc) => {
                    unary_float_ops!(handles, desc, B::float_sin)
                }
                FloatOperationIr::Tanh(desc) => {
                    unary_float_ops!(handles, desc, B::float_tanh)
                }
                FloatOperationIr::Tan(desc) => unary_float_ops!(handles, desc, B::float_tan),
                FloatOperationIr::Cosh(desc) => unary_float_ops!(handles, desc, B::float_cosh),
                FloatOperationIr::Sinh(desc) => unary_float_ops!(handles, desc, B::float_sinh),
                FloatOperationIr::ArcCos(desc) => unary_float_ops!(handles, desc, B::float_acos),
                FloatOperationIr::ArcCosh(desc) => unary_float_ops!(handles, desc, B::float_acosh),
                FloatOperationIr::ArcSin(desc) => unary_float_ops!(handles, desc, B::float_asin),
                FloatOperationIr::ArcSinh(desc) => unary_float_ops!(handles, desc, B::float_asinh),
                FloatOperationIr::ArcTan(desc) => unary_float_ops!(handles, desc, B::float_atan),
                FloatOperationIr::ArcTanh(desc) => unary_float_ops!(handles, desc, B::float_atanh),
                FloatOperationIr::ArcTan2(desc) => binary_float_ops!(handles, desc, B::float_atan2),
                FloatOperationIr::Round(desc) => {
                    unary_float_ops!(handles, desc, B::float_round)
                }
                FloatOperationIr::Floor(desc) => {
                    unary_float_ops!(handles, desc, B::float_floor)
                }
                FloatOperationIr::Ceil(desc) => {
                    unary_float_ops!(handles, desc, B::float_ceil)
                }
                FloatOperationIr::Trunc(desc) => {
                    unary_float_ops!(handles, desc, B::float_trunc)
                }
                FloatOperationIr::IntoInt(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_into_int(tensor);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::Matmul(desc) => {
                    binary_float_ops!(handles, desc, B::float_matmul)
                }
                FloatOperationIr::Cross(desc) => {
                    let lhs = handles.get_float_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_float_tensor::<B>(&desc.rhs);
                    let output = B::float_cross(lhs, rhs, desc.dim);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::Random(desc) => {
                    let shape = desc.out.shape.clone();

                    let output = B::float_random(shape, desc.distribution, &self.device);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::Recip(desc) => {
                    unary_float_ops!(handles, desc, B::float_recip)
                }
                FloatOperationIr::Quantize(_) => todo!(),
                FloatOperationIr::Dequantize(_) => todo!(),
                FloatOperationIr::IsNan(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_is_nan(tensor);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::IsInf(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_is_inf(tensor);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::GridSample2d(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let grid = handles.get_float_tensor::<B>(&desc.grid);

                    let output = B::float_grid_sample_2d(tensor, grid, desc.options.clone().into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::Module(op) => match op {
                ModuleOperationIr::Embedding(desc) => {
                    let weights = handles.get_float_tensor::<B>(&desc.weights);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::embedding(weights, indices);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::EmbeddingBackward(desc) => {
                    let weights = handles.get_float_tensor::<B>(&desc.weights);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let output_grad = handles.get_float_tensor::<B>(&desc.out_grad);

                    let output = B::embedding_backward(weights, output_grad, indices);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv1d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::conv1d(x, weight, bias, desc.clone().options.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv1dXBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output =
                        B::conv1d_x_backward(x, weight, output_grad, desc.clone().options.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv1dWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::conv1d_weight_backward(
                        x,
                        weight,
                        output_grad,
                        desc.clone().options.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv1dBiasBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let bias = handles.get_float_tensor::<B>(&desc.bias);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::conv1d_bias_backward(x, bias, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv2d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::conv2d(x, weight, bias, desc.clone().options.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv2dXBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output =
                        B::conv2d_x_backward(x, weight, output_grad, desc.clone().options.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv2dWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::conv2d_weight_backward(
                        x,
                        weight,
                        output_grad,
                        desc.clone().options.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv2dBiasBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let bias = handles.get_float_tensor::<B>(&desc.bias);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::conv2d_bias_backward(x, bias, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv3d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::conv3d(x, weight, bias, desc.options.clone().into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv3dXBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output =
                        B::conv3d_x_backward(x, weight, output_grad, desc.clone().options.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv3dWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::conv3d_weight_backward(
                        x,
                        weight,
                        output_grad,
                        desc.clone().options.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Conv3dBiasBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let bias = handles.get_float_tensor::<B>(&desc.bias);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::conv3d_bias_backward(x, bias, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::DeformableConv2d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let offset = handles.get_float_tensor::<B>(&desc.offset);
                    let mask = desc
                        .mask
                        .as_ref()
                        .map(|mask| handles.get_float_tensor::<B>(mask));
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::deform_conv2d(
                        x,
                        offset,
                        weight,
                        mask,
                        bias,
                        desc.options.clone().into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::DeformableConv2dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let offset = handles.get_float_tensor::<B>(&desc.offset);
                    let mask = desc
                        .mask
                        .as_ref()
                        .map(|mask| handles.get_float_tensor::<B>(mask));
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));
                    let output_grad = handles.get_float_tensor::<B>(&desc.out_grad);

                    let output = B::deform_conv2d_backward(
                        x,
                        offset,
                        weight,
                        mask,
                        bias,
                        output_grad,
                        desc.options.clone().into(),
                    );

                    handles.register_float_tensor::<B>(&desc.input_grad.id, output.x_grad);
                    handles.register_float_tensor::<B>(&desc.offset_grad.id, output.offset_grad);
                    handles.register_float_tensor::<B>(&desc.weight_grad.id, output.weight_grad);
                    if let Some((mask_grad, field)) = output.mask_grad.zip(desc.mask_grad.as_ref())
                    {
                        handles.register_float_tensor::<B>(&field.id, mask_grad);
                    }
                    if let Some((bias_grad, field)) = output.bias_grad.zip(desc.bias_grad.as_ref())
                    {
                        handles.register_float_tensor::<B>(&field.id, bias_grad);
                    }
                }
                ModuleOperationIr::ConvTranspose1d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::conv_transpose1d(x, weight, bias, desc.options.clone().into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose2d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::conv_transpose2d(x, weight, bias, desc.options.clone().into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose3d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::conv_transpose3d(x, weight, bias, desc.options.clone().into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AvgPool1d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::avg_pool1d(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.count_include_pad,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AvgPool2d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::avg_pool2d(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.count_include_pad,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AvgPool1dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::avg_pool1d_backward(
                        x,
                        grad,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.count_include_pad,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AvgPool2dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::avg_pool2d_backward(
                        x,
                        grad,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.count_include_pad,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AdaptiveAvgPool1d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::adaptive_avg_pool1d(x, desc.output_size);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AdaptiveAvgPool2d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::adaptive_avg_pool2d(x, desc.output_size);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AdaptiveAvgPool1dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::adaptive_avg_pool1d_backward(x, grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AdaptiveAvgPool2dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::adaptive_avg_pool2d_backward(x, grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::MaxPool1d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::max_pool1d(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::MaxPool1dWithIndices(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::max_pool1d_with_indices(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output.output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output.indices);
                }
                ModuleOperationIr::MaxPool1dWithIndicesBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let output_grad = handles.get_float_tensor::<B>(&desc.grad);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::max_pool1d_with_indices_backward(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                        output_grad,
                        indices,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output.x_grad);
                }
                ModuleOperationIr::MaxPool2d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::max_pool2d(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::MaxPool2dWithIndices(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::max_pool2d_with_indices(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output.output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output.indices);
                }
                ModuleOperationIr::MaxPool2dWithIndicesBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let output_grad = handles.get_float_tensor::<B>(&desc.grad);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::max_pool2d_with_indices_backward(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                        output_grad,
                        indices,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output.x_grad);
                }
                ModuleOperationIr::Interpolate(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::interpolate(x, desc.output_size, desc.options.clone().into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::InterpolateBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::interpolate_backward(
                        x,
                        grad,
                        desc.output_size,
                        desc.options.clone().into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::Custom(_) => {
                panic!("Can't execute custom operation here")
            }
            OperationIr::Init(_) => {
                // Nothing to do.
            }
            OperationIr::Drop(repr) => {
                handles.remove_handle(repr.id);
            }
        }
    }

    fn read_tensor_async(&self, tensor: TensorIr) -> DynFut<Result<TensorData, ExecutionError>> {
        let mut ctx = self.context.lock().unwrap();

        enum Output<B: Backend> {
            Float(B::FloatTensorPrimitive),
            Int(B::IntTensorPrimitive),
            Bool(B::BoolTensorPrimitive),
        }

        let tensor = if tensor.dtype.is_float() {
            let tensor = ctx.handles.get_float_tensor::<B>(&tensor);
            Output::<B>::Float(tensor)
        } else if tensor.dtype.is_int() {
            let tensor = ctx.handles.get_int_tensor::<B>(&tensor);
            Output::Int(tensor)
        } else if tensor.dtype.is_bool() {
            let tensor = ctx.handles.get_bool_tensor::<B>(&tensor);
            Output::Bool(tensor)
        } else if let DType::QFloat(_) = tensor.dtype {
            todo!()
        } else {
            unimplemented!()
        };

        match tensor {
            Output::Float(val) => Box::pin(B::float_into_data(val)),
            Output::Int(val) => Box::pin(B::int_into_data(val)),
            Output::Bool(val) => Box::pin(B::bool_into_data(val)),
        }
    }

    fn register_tensor_data(&self, data: TensorData) -> RouterTensor<Self> {
        let desc = self.register_tensor_data_desc(data);
        RouterTensor::new(desc.id, desc.shape, desc.dtype, self.clone())
    }

    fn device(&self) -> Self::Device {
        self.device.clone()
    }

    fn sync(&self) -> Result<(), ExecutionError> {
        B::sync(&self.device)
    }

    fn seed(&self, seed: u64) {
        B::seed(&self.device, seed)
    }

    fn create_empty_handle(&self) -> TensorId {
        let mut ctx = self.context.lock().unwrap();
        ctx.create_empty_handle()
    }

    fn supports_dtype(&self, dtype: DType) -> bool {
        B::supports_dtype(&self.device, dtype)
    }
}
