use super::{RouterClient, RouterTensor};
use crate::CustomOpRegistry;
use crate::{
    binary_bool_ops, binary_float_cmp_ops, binary_float_ops, binary_int_cmp_ops, binary_int_ops,
    reduce_float_dim_ops, reduce_float2int_dim_ops, reduce_int_dim_ops, scalar_float_cmp_ops,
    scalar_float_ops, scalar_int_cmp_ops, scalar_int_ops, unary_float_ops, unary_int_ops,
};
use alloc::{
    boxed::Box,
    format,
    string::{String, ToString},
    vec::Vec,
};
use burn_backend::{
    Backend, BackendGraph, DType, DeviceOps, ExecutionError, FloatDType, ProfileDuration,
    ProfileOptions, ProfileToken, Shape, TensorData, TensorMetadata, TensorPrimitive,
    ops::TransactionPrimitive,
};
use burn_ir::{
    ActivationOperationIr, BackendIr, BaseOperationIr, BoolOperationIr, FloatOperationIr,
    HandleContainer, HandleKind, IntOperationIr, ModuleOperationIr, NumericOperationIr,
    OperationIr, TensorError, TensorId, TensorIr, TensorStatus,
};
use burn_std::{DeviceSettings, future::DynFut};
use portable_atomic::{AtomicU64, Ordering};

/// `$float` on a float tensor and `$quantized` on a quantized one, each with the tensor bound to
/// `$tensor`, so an op that has a quantized kernel keeps its tensor quantized.
macro_rules! float_or_quantized {
    ($tensor:ident, $float:expr, $quantized:expr) => {
        match $tensor {
            TensorPrimitive::Float($tensor) => TensorPrimitive::Float($float),
            TensorPrimitive::QFloat($tensor) => TensorPrimitive::QFloat($quantized),
        }
    };
}

/// An interpreter's context contains a [handle container](HandleContainer) to manage
/// (i.e., fetch and update) existing tensors.
pub struct InterpreterContext<B: BackendIr> {
    /// Handle container to retrieve tensors based on their intermediate representation.
    handles: HandleContainer<B::Handle>,
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

impl<B: BackendIr> InterpreterContext<B> {
    /// Create a new (uninitialized) empty tensor and returns its corresponding [tensor id](TensorId).
    fn create_empty_handle(&mut self) -> TensorId {
        let value = COUNTER.fetch_add(1, Ordering::Relaxed);
        TensorId::new(value)
    }
}

/// A tensor interpreter is responsible for executing tensor operations for a given [intermediate backend](BackendIr).
///
/// Single-owner and **not** `Clone`: the interpreter is driven by exactly one thread (the remote
/// server's per-session worker), so the handle container is owned directly rather than behind a
/// lock — every op takes `&mut self`, with no per-op `Mutex` on the hot path.
pub struct TensorInterpreter<B: BackendIr> {
    context: InterpreterContext<B>,
    device: B::Device,
    /// Handlers for [custom operations](OperationIr::Custom), keyed by id. Shared read-only across
    /// every session, so executing a custom op is a map lookup plus a call.
    custom_ops: CustomOpRegistry<B>,
}

impl<B: BackendIr> core::fmt::Debug for TensorInterpreter<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorInterpreter")
            .field("device", &self.device)
            .finish()
    }
}

impl<B: BackendIr> TensorInterpreter<B> {
    /// Create a new interpreter without any custom operation handlers.
    pub fn new(device: B::Device) -> Self {
        Self::with_custom_ops(device, CustomOpRegistry::default())
    }

    /// Create a new interpreter with the given [custom operation handlers](CustomOpRegistry).
    pub fn with_custom_ops(device: B::Device, custom_ops: CustomOpRegistry<B>) -> Self {
        Self {
            context: InterpreterContext {
                handles: HandleContainer::new(),
            },
            device,
            custom_ops,
        }
    }

    /// Get the tensor handle for the given [tensor representation](TensorIr).
    pub fn get_tensor_handle(&mut self, tensor: &TensorIr) -> B::Handle {
        let handles = &mut self.context.handles;
        handles.get_tensor_handle(tensor).handle
    }

    /// Take the typed backend primitive for `tensor`, dispatching on its dtype.
    ///
    /// Unlike [`get_tensor_handle`](Self::get_tensor_handle) (which returns the opaque
    /// `B::Handle`), this returns the concrete float/int/bool/quantized primitive so the caller
    /// can hand it to `B::*_to_device`.
    pub fn get_tensor(&mut self, tensor: &TensorIr) -> HandleKind<B> {
        let handles = &mut self.context.handles;
        match tensor.dtype {
            DType::QFloat(_) => HandleKind::Quantized(handles.get_quantized_tensor::<B>(tensor)),
            dtype if dtype.is_float() => HandleKind::Float(handles.get_float_tensor::<B>(tensor)),
            dtype if dtype.is_int() || dtype.is_uint() => {
                HandleKind::Int(handles.get_int_tensor::<B>(tensor))
            }
            dtype if dtype.is_bool() => HandleKind::Bool(handles.get_bool_tensor::<B>(tensor)),
            dtype => unreachable!("{dtype:?} is neither float, int, bool nor quantized"),
        }
    }

    /// Move a primitive produced on another interpreter's device onto this interpreter's device
    /// and register it under `id`.
    ///
    /// The counterpart of [`get_tensor`](Self::get_tensor): the source interpreter hands over its
    /// primitive, and the destination calls `B::*_to_device` onto its own device. When both
    /// interpreters share the same device, the backend's `to_device` is a cheap no-op.
    pub fn register_tensor_to_device(&mut self, id: TensorId, tensor: HandleKind<B>) {
        let ctx = &mut self.context;
        match tensor {
            HandleKind::Float(tensor) => {
                let tensor = B::float_to_device(tensor, &self.device);
                ctx.handles.register_float_tensor::<B>(&id, tensor);
            }
            HandleKind::Int(tensor) => {
                let tensor = B::int_to_device(tensor, &self.device);
                ctx.handles.register_int_tensor::<B>(&id, tensor);
            }
            HandleKind::Bool(tensor) => {
                let tensor = B::bool_to_device(tensor, &self.device);
                ctx.handles.register_bool_tensor::<B>(&id, tensor);
            }
            HandleKind::Quantized(tensor) => {
                let tensor = B::q_to_device(tensor, &self.device);
                ctx.handles.register_quantized_tensor::<B>(&id, tensor);
            }
        }
    }

    /// Register `id` as failed by `error`, so a read of it reports why instead of finding nothing.
    pub fn set_error(&mut self, id: TensorId, error: TensorError) {
        self.context.handles.set_error(id, error);
    }

    /// Register `id` as failed by `error`, unless something wrote it or a failure already claims
    /// it: for a failure that cannot name its write set.
    pub fn claim_unwritten(&mut self, id: TensorId, error: TensorError) {
        self.context.handles.claim_unwritten(id, error);
    }

    /// The failure `tensor` carries, taken the way [`get_tensor`](Self::get_tensor) takes its
    /// handle: a read-write use releases it.
    pub fn take_error(&mut self, tensor: &TensorIr) -> Option<TensorError> {
        self.context.handles.take_error(tensor)
    }

    /// Why `id` cannot be read, if it cannot: a failure claimed it, or nothing produced it.
    pub fn read_failure(&self, id: &TensorId) -> Option<String> {
        let handles = &self.context.handles;
        match handles.error(id) {
            Some(error) => Some(error.to_string()),
            None => (!handles.has_handle(id)).then(|| format!("tensor {id:?} was never produced")),
        }
    }

    /// Create a tensor with the given handle and shape.
    pub fn register_tensor<C: RouterClient>(
        &mut self,
        handle: B::Handle,
        shape: Shape,
        dtype: DType,
        client: C,
    ) -> RouterTensor<C> {
        let ctx = &mut self.context;
        let id = ctx.create_empty_handle();

        ctx.handles.register_handle(id, handle);

        RouterTensor::new(id, shape, dtype, client)
    }

    /// Register `new_id` as an alias of `src_id`: a second handle over the same backing buffer.
    ///
    /// Used for the fusion layer's cross-stream sharing (see
    /// [`RouterClient::register_alias`](crate::RouterClient::register_alias)). Cloning the backend
    /// handle is a cheap `Arc`-style refcount bump, so the buffer survives until *both* ids are
    /// freed — consuming the alias on one stream can't pull it out from under the other.
    ///
    /// An alias of a tensor a failure claimed carries the same failure.
    pub fn register_alias(&mut self, new_id: TensorId, src_id: TensorId) {
        let ctx = &mut self.context;
        if let Some(error) = ctx.handles.error(&src_id) {
            let error = error.clone();
            ctx.handles.set_error(new_id, error);
            return;
        }
        let handle = ctx
            .handles
            .get_handle_ref(&src_id)
            .expect("alias source tensor must be materialized before it is aliased")
            .clone();
        ctx.handles.register_handle(new_id, handle);
    }

    /// Register a tensor from its data and id.
    pub fn register_tensor_data_id(&mut self, id: TensorId, data: TensorData) {
        let ctx = &mut self.context;
        let dtype = data.dtype();

        match dtype {
            DType::QFloat(_) => {
                let tensor = B::q_from_data(data, &self.device);
                ctx.handles.register_quantized_tensor::<B>(&id, tensor)
            }
            dtype if dtype.is_float() => {
                let tensor = B::float_from_data(data, &self.device);
                ctx.handles.register_float_tensor::<B>(&id, tensor)
            }
            dtype if dtype.is_int() || dtype.is_uint() => {
                let tensor = B::int_from_data(data, &self.device);
                ctx.handles.register_int_tensor::<B>(&id, tensor)
            }
            dtype if dtype.is_bool() => {
                let tensor = B::bool_from_data(data, &self.device);
                ctx.handles.register_bool_tensor::<B>(&id, tensor)
            }
            dtype => unreachable!("{dtype:?} is neither float, int, bool nor quantized"),
        }
    }

    /// Register a tensor and returns its intermediate representation.
    pub fn register_tensor_data_desc(&mut self, data: TensorData) -> TensorIr {
        let id = self.context.create_empty_handle();
        let shape = data.shape().clone();
        let dtype = data.dtype();
        self.register_tensor_data_id(id, data);

        TensorIr {
            id,
            shape,
            status: TensorStatus::ReadWrite,
            dtype,
        }
    }

    /// Get the device default settings.
    pub fn device_settings(&self) -> DeviceSettings {
        self.device.defaults()
    }
}

/// The message inside a caught panic payload: what `panic!` produces.
#[cfg(feature = "std")]
fn panic_message(panic: &(dyn core::any::Any + Send)) -> &str {
    panic
        .downcast_ref::<&'static str>()
        .copied()
        .or_else(|| panic.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

impl<B: BackendIr> TensorInterpreter<B> {
    /// Execute a tensor operation.
    ///
    /// Failures are deferred to the tensors the operation was going to write,
    /// as on fusion and CubeCL backends: an operation reading a tensor a
    /// failure claimed is skipped and its outputs carry that failure, and one
    /// that panics claims its outputs with the panic's message. Either is
    /// reported by the read of one of those outputs, while work sharing no
    /// tensor with the failure runs as usual.
    pub fn register_op(&mut self, op: OperationIr) {
        let handles = &mut self.context.handles;
        if let Some(error) = handles.input_error(&op) {
            let error = error.propagated();
            handles.set_output_errors(&op, &error);
            return;
        }

        #[cfg(feature = "std")]
        {
            let executed =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute_op(&op)));
            if let Err(panic) = executed {
                let error = TensorError::new(ExecutionError::generic(panic_message(&*panic)));
                self.context.handles.set_output_errors(&op, &error);
            }
        }
        #[cfg(not(feature = "std"))]
        self.execute_op(&op);
    }

    fn execute_op(&mut self, op: &OperationIr) {
        let handles = &mut self.context.handles;
        match op {
            // For every op: get the input(s), execute the operation and register the output(s)
            OperationIr::BaseFloat(op) => match op {
                BaseOperationIr::Reshape(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.input);
                    let shape = desc.out.shape.clone();

                    let output = float_or_quantized!(
                        tensor,
                        B::float_reshape(tensor, shape),
                        B::q_reshape(tensor, shape)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SwapDims(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.input);

                    let output = float_or_quantized!(
                        tensor,
                        B::float_swap_dims(tensor, desc.dim1, desc.dim2),
                        B::q_swap_dims(tensor, desc.dim1, desc.dim2)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Permute(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.input);

                    let output = float_or_quantized!(
                        tensor,
                        B::float_permute(tensor, &desc.axes),
                        B::q_permute(tensor, &desc.axes)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Flip(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.input);

                    let output = float_or_quantized!(
                        tensor,
                        B::float_flip(tensor, &desc.axes),
                        B::q_flip(tensor, &desc.axes)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Expand(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.input);
                    let shape = desc.out.shape.clone();

                    let output = float_or_quantized!(
                        tensor,
                        B::float_expand(tensor, shape),
                        B::q_expand(tensor, shape)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Unfold(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_unfold(tensor, desc.dim, desc.size, desc.step);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Slice(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.tensor);

                    let output = float_or_quantized!(
                        tensor,
                        B::float_slice(tensor, &desc.ranges),
                        B::q_slice(tensor, &desc.ranges)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SliceAssign(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output = B::float_slice_assign(tensor, &desc.ranges, value);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Gather(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = float_or_quantized!(
                        tensor,
                        B::float_gather(desc.dim, tensor, indices),
                        B::q_gather(desc.dim, tensor, indices)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Scatter(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output = B::float_scatter(desc.dim, tensor, indices, value, desc.update);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::ScatterNd(desc) => {
                    let data = handles.get_float_tensor::<B>(&desc.data);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let values = handles.get_float_tensor::<B>(&desc.values);

                    let output = B::float_scatter_nd(data, indices, values, desc.reduction);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::GatherNd(desc) => {
                    let data = handles.get_float_tensor::<B>(&desc.data);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::float_gather_nd(data, indices);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Select(desc) => {
                    let tensor = handles.get_float_primitive::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = float_or_quantized!(
                        tensor,
                        B::float_select(tensor, desc.dim, indices),
                        B::q_select(tensor, desc.dim, indices)
                    );
                    handles.register_float_primitive::<B>(&desc.out.id, output);
                }
                BaseOperationIr::SelectAssign(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let output =
                        B::float_select_assign(tensor, desc.dim, indices, value, desc.update);
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
                BaseOperationIr::NotEqual(desc) => {
                    binary_float_cmp_ops!(handles, desc, B::float_not_equal)
                }
                BaseOperationIr::NotEqualElem(desc) => {
                    scalar_float_cmp_ops!(handles, desc, B::float_not_equal_elem)
                }
                BaseOperationIr::All(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_all(tensor, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Any(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_any(tensor, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::AllDim(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_all_dim(tensor, desc.axis, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::AnyDim(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_any_dim(tensor, desc.axis, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
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

                    let output = B::int_scatter(desc.dim, tensor, indices, value, desc.update);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::ScatterNd(desc) => {
                    let data = handles.get_int_tensor::<B>(&desc.data);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let values = handles.get_int_tensor::<B>(&desc.values);

                    let output = B::int_scatter_nd(data, indices, values, desc.reduction);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::GatherNd(desc) => {
                    let data = handles.get_int_tensor::<B>(&desc.data);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::int_gather_nd(data, indices);
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

                    let output =
                        B::int_select_assign(tensor, desc.dim, indices, value, desc.update);
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
                BaseOperationIr::Cast(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_cast(tensor, desc.out.dtype.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
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
                BaseOperationIr::NotEqual(desc) => {
                    binary_int_cmp_ops!(handles, desc, B::int_not_equal)
                }
                BaseOperationIr::NotEqualElem(desc) => {
                    scalar_int_cmp_ops!(handles, desc, B::int_not_equal_elem)
                }
                BaseOperationIr::All(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_all(tensor, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Any(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_any(tensor, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::AllDim(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_all_dim(tensor, desc.axis, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::AnyDim(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_any_dim(tensor, desc.axis, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
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

                    let output = B::bool_scatter(desc.dim, tensor, indices, value, desc.update);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::ScatterNd(desc) => {
                    let data = handles.get_bool_tensor::<B>(&desc.data);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);
                    let values = handles.get_bool_tensor::<B>(&desc.values);

                    let output = B::bool_scatter_nd(data, indices, values, desc.reduction);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::GatherNd(desc) => {
                    let data = handles.get_bool_tensor::<B>(&desc.data);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::bool_gather_nd(data, indices);
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

                    let output =
                        B::bool_select_assign(tensor, desc.dim, indices, value, desc.update);
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

                    let output = B::bool_mask_fill(tensor, mask, desc.value.into());
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

                    let output = B::bool_equal_elem(lhs, desc.rhs.into());
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
                    let output = B::bool_empty(shape, &self.device, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Zeros(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::bool_zeros(shape, &self.device, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Ones(desc) => {
                    let shape = desc.out.shape.clone();
                    let output = B::bool_ones(shape, &self.device, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::NotEqual(desc) => {
                    let lhs = handles.get_bool_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_bool_tensor::<B>(&desc.rhs);
                    let output = B::bool_not_equal(lhs, rhs);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::NotEqualElem(desc) => {
                    let lhs = handles.get_bool_tensor::<B>(&desc.lhs);
                    let output = B::bool_not_equal_elem(lhs, desc.rhs.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::All(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);
                    let output = B::bool_all(tensor);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::Any(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);
                    let output = B::bool_any(tensor);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::AllDim(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);
                    let output = B::bool_all_dim(tensor, desc.axis);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                BaseOperationIr::AnyDim(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);
                    let output = B::bool_any_dim(tensor, desc.axis);
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::NumericFloat(_dtype, op) => match op {
                NumericOperationIr::Pad(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_pad(tensor, &desc.padding, desc.mode.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
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
                    reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_mean_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::Mean(desc) => {
                    unary_float_ops!(handles, desc, B::float_mean)
                }
                NumericOperationIr::Sum(desc) => {
                    unary_float_ops!(handles, desc, B::float_sum)
                }
                NumericOperationIr::SumDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_sum_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::SumDims(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_sum_dims(input, &desc.axes);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::Prod(desc) => {
                    unary_float_ops!(handles, desc, B::float_prod)
                }
                NumericOperationIr::ProdDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_prod_dim(
                        tensor, axis
                    ))
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
                    reduce_float2int_dim_ops!(handles, desc, |tensor, axis, _, dtype| {
                        B::float_argmax(tensor, axis, dtype)
                    })
                }
                NumericOperationIr::ArgTopK(desc) => {
                    reduce_float2int_dim_ops!(handles, desc, B::float_argtopk)
                }
                NumericOperationIr::ArgMin(desc) => {
                    reduce_float2int_dim_ops!(handles, desc, |tensor, axis, _, dtype| {
                        B::float_argmin(tensor, axis, dtype)
                    })
                }
                NumericOperationIr::Max(desc) => {
                    unary_float_ops!(handles, desc, B::float_max)
                }
                NumericOperationIr::MaxDimWithIndices(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::float_max_dim_with_indices(
                        tensor,
                        desc.dim,
                        desc.out_indices.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::TopKWithIndices(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::float_topk_with_indices(
                        tensor,
                        desc.dim,
                        desc.k,
                        desc.out_indices.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::MinDimWithIndices(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::float_min_dim_with_indices(
                        tensor,
                        desc.dim,
                        desc.out_indices.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output_idx);
                }
                NumericOperationIr::Min(desc) => {
                    unary_float_ops!(handles, desc, B::float_min)
                }
                NumericOperationIr::MaxDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_max_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::TopK(desc) => {
                    reduce_float_dim_ops!(handles, desc, B::float_topk)
                }
                NumericOperationIr::MinDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_min_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::MaxAbs(desc) => {
                    unary_float_ops!(handles, desc, B::float_max_abs)
                }
                NumericOperationIr::MaxAbsDim(desc) => {
                    reduce_float_dim_ops!(handles, desc, |tensor, axis, _| B::float_max_abs_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::Clamp(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);

                    let output = B::float_clamp(tensor, desc.min.into(), desc.max.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::IntRandom(_) => unreachable!(),
                NumericOperationIr::Powi(desc) => {
                    let lhs = handles.get_float_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_int_tensor::<B>(&desc.rhs);
                    let output = (B::float_powi)(lhs, rhs);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::PowiScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_powi_scalar)
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
                NumericOperationIr::Neg(desc) => {
                    unary_float_ops!(handles, desc, B::float_neg)
                }
                NumericOperationIr::Sign(desc) => {
                    unary_float_ops!(handles, desc, B::float_sign)
                }
                NumericOperationIr::ClampMin(desc) => {
                    scalar_float_ops!(handles, desc, B::float_clamp_min)
                }
                NumericOperationIr::ClampMax(desc) => {
                    scalar_float_ops!(handles, desc, B::float_clamp_max)
                }
                NumericOperationIr::Sort(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::float_sort(tensor, desc.dim, desc.descending);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::SortWithIndices(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let (values, indices) = B::float_sort_with_indices(
                        tensor,
                        desc.dim,
                        desc.descending,
                        desc.out_indices.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, values);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, indices);
                }
                NumericOperationIr::ArgSort(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);
                    let output =
                        B::float_argsort(tensor, desc.dim, desc.descending, desc.out.dtype.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::NumericInt(_dtype, op) => match op {
                NumericOperationIr::Pad(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_pad(tensor, &desc.padding, desc.mode.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
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
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_mean_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::Mean(desc) => {
                    unary_int_ops!(handles, desc, B::int_mean)
                }
                NumericOperationIr::Sum(desc) => {
                    unary_int_ops!(handles, desc, B::int_sum)
                }
                NumericOperationIr::SumDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_sum_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::SumDims(desc) => {
                    let input = handles.get_int_tensor::<B>(&desc.input);
                    let output = desc
                        .axes
                        .iter()
                        .fold(input, |tensor, &axis| B::int_sum_dim(tensor, axis));
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::Prod(desc) => {
                    unary_int_ops!(handles, desc, B::int_prod)
                }
                NumericOperationIr::ProdDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_prod_dim(
                        tensor, axis
                    ))
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
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_argmax(
                        tensor, axis
                    ))
                }
                NumericOperationIr::ArgTopK(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_argtopk)
                }
                NumericOperationIr::ArgMin(desc) => {
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_argmin(
                        tensor, axis
                    ))
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
                NumericOperationIr::TopKWithIndices(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let (output, output_idx) = B::int_topk_with_indices(tensor, desc.dim, desc.k);
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
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_max_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::TopK(desc) => {
                    reduce_int_dim_ops!(handles, desc, B::int_topk)
                }
                NumericOperationIr::MinDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_min_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::MaxAbs(desc) => {
                    unary_int_ops!(handles, desc, B::int_max_abs)
                }
                NumericOperationIr::MaxAbsDim(desc) => {
                    reduce_int_dim_ops!(handles, desc, |tensor, axis, _| B::int_max_abs_dim(
                        tensor, axis
                    ))
                }
                NumericOperationIr::Clamp(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.tensor);

                    let output = B::int_clamp(tensor, desc.min.into(), desc.max.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::IntRandom(desc) => {
                    let shape = desc.out.shape.clone();

                    let output = B::int_random(
                        shape,
                        desc.distribution,
                        &self.device,
                        desc.out.dtype.into(),
                    );
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::Powi(desc) => {
                    let lhs = handles.get_int_tensor::<B>(&desc.lhs);
                    let rhs = handles.get_int_tensor::<B>(&desc.rhs);

                    let output = B::int_powi(lhs, rhs);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::PowiScalar(desc) => {
                    scalar_int_ops!(handles, desc, B::int_powi_scalar)
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
                NumericOperationIr::Neg(desc) => {
                    unary_int_ops!(handles, desc, B::int_neg)
                }
                NumericOperationIr::Sign(desc) => {
                    unary_int_ops!(handles, desc, B::int_sign)
                }
                NumericOperationIr::ClampMin(desc) => {
                    scalar_int_ops!(handles, desc, B::int_clamp_min)
                }
                NumericOperationIr::ClampMax(desc) => {
                    scalar_int_ops!(handles, desc, B::int_clamp_max)
                }
                NumericOperationIr::Sort(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_sort(tensor, desc.dim, desc.descending);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                NumericOperationIr::SortWithIndices(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let (values, indices) =
                        B::int_sort_with_indices(tensor, desc.dim, desc.descending);
                    handles.register_int_tensor::<B>(&desc.out.id, values);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, indices);
                }
                NumericOperationIr::ArgSort(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);
                    let output = B::int_argsort(tensor, desc.dim, desc.descending);
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::Bool(op) => match op {
                BoolOperationIr::IntoFloat(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_into_float(tensor, desc.out.dtype.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                BoolOperationIr::IntoInt(desc) => {
                    let tensor = handles.get_bool_tensor::<B>(&desc.input);

                    let output = B::bool_into_int(tensor, desc.out.dtype.into());
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
                BoolOperationIr::Xor(desc) => {
                    binary_bool_ops!(handles, desc, B::bool_xor)
                }
            },
            OperationIr::Int(op) => match op {
                IntOperationIr::IntoFloat(desc) => {
                    let tensor = handles.get_int_tensor::<B>(&desc.input);

                    let output = B::int_into_float(tensor, desc.out.dtype.into());
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
                FloatOperationIr::Powf(desc) => {
                    binary_float_ops!(handles, desc, B::float_powf)
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
                FloatOperationIr::Hypot(desc) => binary_float_ops!(handles, desc, B::float_hypot),
                FloatOperationIr::Fmod(desc) => binary_float_ops!(handles, desc, B::float_fmod),
                FloatOperationIr::FmodScalar(desc) => {
                    scalar_float_ops!(handles, desc, B::float_fmod_scalar)
                }
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

                    let output = B::float_into_int(tensor, desc.out.dtype.into());
                    handles.register_int_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::Matmul(desc)
                    if matches!(desc.lhs.dtype, DType::QFloat(_))
                        || matches!(desc.rhs.dtype, DType::QFloat(_)) =>
                {
                    let lhs = handles.get_float_primitive::<B>(&desc.lhs);
                    let rhs = handles.get_float_primitive::<B>(&desc.rhs);

                    let output = Self::as_recorded(B::q_matmul(lhs, rhs), desc.out.dtype);
                    handles.register_float_primitive::<B>(&desc.out.id, output);
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

                    let output = B::float_random(
                        shape,
                        desc.distribution,
                        &self.device,
                        desc.out.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::Recip(desc) => {
                    unary_float_ops!(handles, desc, B::float_recip)
                }
                FloatOperationIr::Quantize(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.tensor);
                    let qparams = handles.get_quantization_parameters::<B>(&desc.qparams);

                    let output = B::quantize(tensor, &desc.scheme, qparams);
                    handles.register_quantized_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::Dequantize(desc) => {
                    let tensor = handles.get_quantized_tensor::<B>(&desc.input);

                    let output = B::dequantize(tensor, desc.out.dtype.into());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::IsNan(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_is_nan(tensor, desc.out.dtype.into());
                    handles.register_bool_tensor::<B>(&desc.out.id, output);
                }
                FloatOperationIr::IsInf(desc) => {
                    let tensor = handles.get_float_tensor::<B>(&desc.input);

                    let output = B::float_is_inf(tensor, desc.out.dtype.into());
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
                ModuleOperationIr::BatchNorm(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let gamma = handles.get_float_tensor::<B>(&desc.gamma);
                    let beta = handles.get_float_tensor::<B>(&desc.beta);
                    let mean = handles.get_float_tensor::<B>(&desc.mean);
                    let variance = handles.get_float_tensor::<B>(&desc.variance);
                    let output = B::batch_norm(x, gamma, beta, mean, variance, desc.epsilon.elem());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
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
                ModuleOperationIr::Linear(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let bias = desc
                        .bias
                        .as_ref()
                        .map(|bias| handles.get_float_tensor::<B>(bias));

                    let output = B::linear(x, weight, bias);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::LinearXBackward(desc) => {
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::linear_x_backward(weight, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::LinearWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::linear_weight_backward(x, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::LinearBiasBackward(desc) => {
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);

                    let output = B::linear_bias_backward(output_grad);
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
                        desc.dilation,
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
                        desc.dilation,
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
                        desc.dilation,
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
                        desc.dilation,
                        desc.count_include_pad,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AvgPool3d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::avg_pool3d(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.count_include_pad,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AvgPool3dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::avg_pool3d_backward(
                        x,
                        grad,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
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
                ModuleOperationIr::AdaptiveAvgPool3d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::adaptive_avg_pool3d(x, desc.output_size);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AdaptiveAvgPool3dBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let grad = handles.get_float_tensor::<B>(&desc.grad);

                    let output = B::adaptive_avg_pool3d_backward(x, grad);
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
                        desc.out_indices.dtype.into(),
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
                        desc.out_indices.dtype.into(),
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
                ModuleOperationIr::MaxPool3d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::max_pool3d(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::MaxPool3dWithIndices(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);

                    let output = B::max_pool3d_with_indices(
                        x,
                        desc.kernel_size,
                        desc.stride,
                        desc.padding,
                        desc.dilation,
                        desc.ceil_mode,
                        desc.out_indices.dtype.into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output.output);
                    handles.register_int_tensor::<B>(&desc.out_indices.id, output.indices);
                }
                ModuleOperationIr::MaxPool3dWithIndicesBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let output_grad = handles.get_float_tensor::<B>(&desc.grad);
                    let indices = handles.get_int_tensor::<B>(&desc.indices);

                    let output = B::max_pool3d_with_indices_backward(
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
                ModuleOperationIr::Attention(desc) => {
                    let query = handles.get_float_tensor::<B>(&desc.query);
                    let key = handles.get_float_tensor::<B>(&desc.key);
                    let value = handles.get_float_tensor::<B>(&desc.value);
                    let mask = desc.mask.as_ref().map(|m| handles.get_bool_tensor::<B>(m));
                    let attn_bias = desc
                        .attn_bias
                        .as_ref()
                        .map(|ab| handles.get_float_tensor::<B>(ab));

                    let output = B::attention(
                        query,
                        key,
                        value,
                        mask,
                        attn_bias,
                        desc.options.clone().into(),
                    );

                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::AttentionWithLogsumexp(desc) => {
                    let query = handles.get_float_tensor::<B>(&desc.query);
                    let key = handles.get_float_tensor::<B>(&desc.key);
                    let value = handles.get_float_tensor::<B>(&desc.value);

                    let (out, logsumexp) =
                        B::attention_with_logsumexp(query, key, value, desc.options.clone().into());

                    handles.register_float_tensor::<B>(&desc.out.id, out);
                    handles.register_float_tensor::<B>(&desc.logsumexp.id, logsumexp);
                }
                ModuleOperationIr::AttentionBackward(desc) => {
                    let query = handles.get_float_tensor::<B>(&desc.query);
                    let key = handles.get_float_tensor::<B>(&desc.key);
                    let value = handles.get_float_tensor::<B>(&desc.value);
                    let out = handles.get_float_tensor::<B>(&desc.out);
                    let logsumexp = handles.get_float_tensor::<B>(&desc.logsumexp);
                    let out_grad = handles.get_float_tensor::<B>(&desc.out_grad);

                    let gradients = B::attention_backward(
                        query,
                        key,
                        value,
                        out,
                        logsumexp,
                        out_grad,
                        desc.options.clone().into(),
                    );

                    handles.register_float_tensor::<B>(&desc.query_grad.id, gradients.query_grad);
                    handles.register_float_tensor::<B>(&desc.key_grad.id, gradients.key_grad);
                    handles.register_float_tensor::<B>(&desc.value_grad.id, gradients.value_grad);
                }
                ModuleOperationIr::CtcLoss(desc) => {
                    let log_probs = handles.get_float_tensor::<B>(&desc.log_probs);
                    let targets = handles.get_int_tensor::<B>(&desc.targets);
                    let input_lengths = handles.get_int_tensor::<B>(&desc.input_lengths);
                    let target_lengths = handles.get_int_tensor::<B>(&desc.target_lengths);

                    let output = B::ctc_loss(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths,
                        desc.blank,
                    );

                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::CtcLossBackward(desc) => {
                    let log_probs = handles.get_float_tensor::<B>(&desc.log_probs);
                    let targets = handles.get_int_tensor::<B>(&desc.targets);
                    let input_lengths = handles.get_int_tensor::<B>(&desc.input_lengths);
                    let target_lengths = handles.get_int_tensor::<B>(&desc.target_lengths);
                    let grad_loss = handles.get_float_tensor::<B>(&desc.grad_loss);

                    let output = B::ctc_loss_backward(
                        log_probs,
                        targets,
                        input_lengths,
                        target_lengths,
                        grad_loss,
                        desc.blank,
                    );

                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::LayerNorm(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let gamma = handles.get_float_tensor::<B>(&desc.gamma);
                    let beta = desc.beta.as_ref().map(|b| handles.get_float_tensor::<B>(b));
                    let output = B::layer_norm(input, gamma, beta, desc.epsilon.elem());
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::Unfold4d(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let options = burn_backend::ops::UnfoldOptions::new(
                        desc.options.stride,
                        desc.options.padding,
                        desc.options.dilation,
                    );
                    let output = B::unfold4d(x, desc.kernel_size, options);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose1dWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);
                    let output = B::conv_transpose1d_weight_backward(
                        x,
                        weight,
                        output_grad,
                        desc.options.clone().into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose1dBiasBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let bias = handles.get_float_tensor::<B>(&desc.bias);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);
                    let output = B::conv_transpose1d_bias_backward(x, bias, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose2dWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);
                    let output = B::conv_transpose2d_weight_backward(
                        x,
                        weight,
                        output_grad,
                        desc.options.clone().into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose2dBiasBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let bias = handles.get_float_tensor::<B>(&desc.bias);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);
                    let output = B::conv_transpose2d_bias_backward(x, bias, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose3dWeightBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let weight = handles.get_float_tensor::<B>(&desc.weight);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);
                    let output = B::conv_transpose3d_weight_backward(
                        x,
                        weight,
                        output_grad,
                        desc.options.clone().into(),
                    );
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ModuleOperationIr::ConvTranspose3dBiasBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.x);
                    let bias = handles.get_float_tensor::<B>(&desc.bias);
                    let output_grad = handles.get_float_tensor::<B>(&desc.output_grad);
                    let output = B::conv_transpose3d_bias_backward(x, bias, output_grad);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
            },
            OperationIr::Activation(op) => match op {
                ActivationOperationIr::Relu(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let output = B::relu(input);
                    handles.register_float_tensor::<B>(&desc.out.id, output);
                }
                ActivationOperationIr::ReluBackward(desc) => {
                    let output = handles.get_float_tensor::<B>(&desc.lhs);
                    let grad = handles.get_float_tensor::<B>(&desc.rhs);
                    let result = B::relu_backward(output, grad);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::LeakyRelu(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.lhs);
                    let result = B::leaky_relu(input, desc.rhs.into());
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::PRelu(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.lhs);
                    let alpha = handles.get_float_tensor::<B>(&desc.rhs);
                    let result = B::prelu(input, alpha);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::Gelu(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let result = B::gelu(input);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::GeluBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.lhs);
                    let grad = handles.get_float_tensor::<B>(&desc.rhs);
                    let result = B::gelu_backward(x, grad);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::Sigmoid(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let result = B::sigmoid(input);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::SigmoidBackward(desc) => {
                    let output = handles.get_float_tensor::<B>(&desc.lhs);
                    let grad = handles.get_float_tensor::<B>(&desc.rhs);
                    let result = B::sigmoid_backward(output, grad);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::HardSigmoid(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.tensor);
                    let result = B::hard_sigmoid(input, desc.alpha.into(), desc.beta.into());
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::LogSigmoid(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let result = B::log_sigmoid(input);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::LogSigmoidBackward(desc) => {
                    let x = handles.get_float_tensor::<B>(&desc.lhs);
                    let grad = handles.get_float_tensor::<B>(&desc.rhs);
                    let result = B::log_sigmoid_backward(x, grad);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::Softmax(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let result = B::softmax(input, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::LogSoftmax(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let result = B::log_softmax(input, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
                ActivationOperationIr::Softmin(desc) => {
                    let input = handles.get_float_tensor::<B>(&desc.input);
                    let result = B::softmin(input, desc.axis);
                    handles.register_float_tensor::<B>(&desc.out.id, result);
                }
            },
            OperationIr::Custom(desc) => match self.custom_ops.get(&desc.id) {
                Some(handler) => handler(handles, desc, &self.device),
                None => panic!(
                    "No custom-op handler registered for `{}`. Register one on the server via \
                     `CustomOpRegistry`/the server builder before starting it.",
                    desc.id
                ),
            },
            OperationIr::Init(_) => {
                // Nothing to do.
            }
            OperationIr::Drop(repr) => {
                handles.remove_handle(repr.id);
            }
        }
    }

    /// Read a tensor's data, returning a future for the host readback.
    ///
    /// A tensor a failure claimed reads as that failure.
    pub fn read_tensor_async(
        &mut self,
        tensor: TensorIr,
    ) -> DynFut<Result<TensorData, ExecutionError>> {
        match self.context.handles.take_error(&tensor) {
            Some(error) => {
                let error = ExecutionError::from(error);
                Box::pin(async move { Err(error) })
            }
            None => self.take_for_read(&tensor).into_data(),
        }
    }

    /// Read several tensors in the order given, those that are not quantized as one backend
    /// transaction.
    ///
    /// When a failure claimed one of them, the read is that failure; the others are still taken.
    pub fn read_tensors_async(
        &mut self,
        tensors: &[TensorIr],
    ) -> DynFut<Result<Vec<TensorData>, ExecutionError>> {
        let mut batch = ReadBatch::default();
        let mut failure = None;
        for tensor in tensors {
            match self.context.handles.take_error(tensor) {
                Some(error) => {
                    failure.get_or_insert(error);
                }
                None => batch.push(self.take_for_read(tensor)),
            }
        }
        match failure {
            Some(error) => {
                let error = ExecutionError::from(error);
                Box::pin(async move { Err(error) })
            }
            None => Box::pin(batch.execute()),
        }
    }

    fn take_for_read(&mut self, tensor: &TensorIr) -> ReadPrimitive<B> {
        let handles = &mut self.context.handles;
        match tensor.dtype {
            DType::QFloat(_) => ReadPrimitive::Quantized(handles.get_quantized_tensor::<B>(tensor)),
            dtype if dtype.is_float() => {
                ReadPrimitive::Float(handles.get_float_tensor::<B>(tensor))
            }
            dtype if dtype.is_int() || dtype.is_uint() => {
                ReadPrimitive::Int(handles.get_int_tensor::<B>(tensor))
            }
            dtype if dtype.is_bool() => ReadPrimitive::Bool(handles.get_bool_tensor::<B>(tensor)),
            dtype => unreachable!("{dtype:?} is neither float, int, bool nor quantized"),
        }
    }

    /// `output` in the dtype the client recorded for it, which a backend picking a quantized
    /// matmul's output from its own settings may not have produced.
    fn as_recorded(output: TensorPrimitive<B>, dtype: DType) -> TensorPrimitive<B> {
        if output.dtype() == dtype {
            return output;
        }
        let output = match output {
            TensorPrimitive::Float(tensor) => tensor,
            TensorPrimitive::QFloat(tensor) => B::dequantize(tensor, FloatDType::F32),
        };
        match dtype {
            DType::QFloat(scheme) => TensorPrimitive::QFloat(B::quantize_dynamic(output, &scheme)),
            dtype => TensorPrimitive::Float(B::float_cast(output, dtype.into())),
        }
    }

    /// The device this interpreter executes on.
    pub fn device(&self) -> B::Device {
        self.device.clone()
    }

    /// Block until all queued backend work has completed.
    ///
    /// # Errors
    ///
    /// Returns an [`ExecutionError`] when the queued work failed.
    pub fn sync(&self) -> Result<(), ExecutionError> {
        B::sync(&self.device)
    }

    /// Seed the backend's RNG.
    pub fn seed(&self, seed: u64) {
        B::seed(&self.device, seed)
    }

    /// The set of supported usages for `dtype` on this backend.
    pub fn dtype_usage(&self, dtype: DType) -> burn_backend::DTypeUsageSet {
        B::dtype_usage(&self.device, dtype)
    }

    /// Open a profiling window on the backend where the calling stream stands.
    pub fn profile_start(&self) -> Result<Option<ProfileToken>, ExecutionError> {
        B::profile_start(&self.device)
    }

    /// Close the window `token` where the calling stream stands.
    pub fn profile_end(
        &self,
        token: ProfileToken,
        options: ProfileOptions,
    ) -> Result<ProfileDuration, ExecutionError> {
        B::profile_end(&self.device, token, options)
    }

    /// Drop the window `token` where the calling stream stands without
    /// measuring it.
    pub fn profile_abandon(&self, token: ProfileToken) {
        B::profile_abandon(&self.device, token)
    }

    /// Prepare the backend for a graph capture.
    pub fn graph_prepare(&self) -> Result<(), ExecutionError> {
        B::graph_prepare(&self.device)
    }

    /// Start recording the backend's launches on the calling stream into a graph.
    pub fn graph_start_capture(&self) -> Result<(), ExecutionError> {
        B::graph_start_capture(&self.device)
    }

    /// Stop the recording on the calling stream and return the backend's graph.
    pub fn graph_stop_capture(&self) -> Result<BackendGraph<B>, ExecutionError> {
        B::graph_stop_capture(&self.device)
    }

    /// Replay `graph` on the calling stream.
    ///
    /// # Safety
    ///
    /// The contract of [`Backend::graph_replay`].
    pub unsafe fn graph_replay(&self, graph: &BackendGraph<B>) -> Result<(), ExecutionError> {
        // Safety: forwarded verbatim from this method's own contract.
        unsafe { B::graph_replay(&self.device, graph) }
    }
}

/// A tensor taken from the handle container to be read.
enum ReadPrimitive<B: Backend> {
    Float(B::FloatTensorPrimitive),
    Int(B::IntTensorPrimitive),
    Bool(B::BoolTensorPrimitive),
    Quantized(B::QuantizedTensorPrimitive),
}

impl<B: Backend> ReadPrimitive<B> {
    fn into_data(self) -> DynFut<Result<TensorData, ExecutionError>> {
        match self {
            Self::Float(tensor) => Box::pin(B::float_into_data(tensor)),
            Self::Int(tensor) => Box::pin(B::int_into_data(tensor)),
            Self::Bool(tensor) => Box::pin(B::bool_into_data(tensor)),
            Self::Quantized(tensor) => Box::pin(B::q_into_data(tensor)),
        }
    }
}

/// The tensors of one read, the quantized ones kept out of the backend transaction because a
/// fused backend's transaction cannot read them.
#[derive(Default)]
struct ReadBatch<B: Backend> {
    transaction: TransactionPrimitive<B>,
    quantized: Vec<QuantizedRead<B>>,
    len: usize,
}

/// A quantized tensor of a [`ReadBatch`], and where its data goes in the batch's answer.
struct QuantizedRead<B: Backend> {
    position: usize,
    tensor: B::QuantizedTensorPrimitive,
}

impl<B: Backend> ReadBatch<B> {
    fn transaction_holds_tensors(&self) -> bool {
        self.quantized.len() < self.len
    }

    fn push(&mut self, tensor: ReadPrimitive<B>) {
        match tensor {
            ReadPrimitive::Float(tensor) => self
                .transaction
                .register_float(TensorPrimitive::Float(tensor)),
            ReadPrimitive::Int(tensor) => self.transaction.register_int(tensor),
            ReadPrimitive::Bool(tensor) => self.transaction.register_bool(tensor),
            ReadPrimitive::Quantized(tensor) => self.quantized.push(QuantizedRead {
                position: self.len,
                tensor,
            }),
        }
        self.len += 1;
    }

    async fn execute(self) -> Result<Vec<TensorData>, ExecutionError> {
        // Some backends' transactions cannot run without a tensor to find their device from.
        let mut data = if self.transaction_holds_tensors() {
            self.transaction.execute_async().await?
        } else {
            Vec::with_capacity(self.len)
        };
        for read in self.quantized {
            data.insert(read.position, B::q_into_data(read.tensor).await?);
        }
        Ok(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_backend::quantization::{QuantScheme, QuantStore, QuantValue};
    use burn_flex::Flex;
    use burn_ir::{CastOpIr, MatmulOpIr};

    #[test]
    fn a_quantized_matmul_yields_the_dtype_the_client_recorded() {
        let mut interpreter = TensorInterpreter::<Flex>::new(Default::default());
        let scheme = QuantScheme::default()
            .with_value(QuantValue::Q8S)
            .with_store(QuantStore::Native);
        let lhs = TensorData::quantized(vec![1i8, 2, 3, 4], [2, 2], scheme, &[0.5], None);
        let lhs = interpreter.register_tensor_data_desc(lhs);
        let rhs = TensorData::new(vec![1.0f32, 0.0, 0.0, 1.0], [2, 2]);
        let rhs = interpreter.register_tensor_data_desc(rhs);
        let mut out = TensorIr::uninit(TensorId::new(u64::MAX), Shape::new([2, 2]), DType::F16);

        interpreter.register_op(OperationIr::Float(
            DType::F32,
            FloatOperationIr::Matmul(MatmulOpIr {
                lhs,
                rhs,
                out: out.clone(),
            }),
        ));

        out.status = TensorStatus::ReadWrite;
        let data = burn_std::reader::try_read_sync(interpreter.read_tensor_async(out))
            .expect("flex reads synchronously")
            .unwrap();
        assert_eq!(data.dtype(), DType::F16);
    }

    /// A tensor a failure claimed reads as that failure, rather than panicking the worker that
    /// serves the read, and a read-write read releases it as it would release a handle.
    #[test]
    fn a_failed_tensor_reads_as_its_failure() {
        let mut interpreter = TensorInterpreter::<Flex>::new(Default::default());
        let written = interpreter.register_tensor_data_desc(TensorData::from([1.0f32]));
        let failed = TensorIr::uninit(TensorId::new(u64::MAX), Shape::new([1]), DType::F32);
        interpreter.set_error(
            failed.id,
            TensorError::new(ExecutionError::with_context("the work failed")),
        );

        let read = TensorIr {
            status: TensorStatus::ReadOnly,
            ..failed
        };
        let error = burn_std::reader::try_read_sync(
            interpreter.read_tensors_async(&[written, read.clone()]),
        )
        .expect("flex reads synchronously")
        .unwrap_err();
        assert!(error.to_string().contains("the work failed"), "{error}");

        // A read-write read is the tensor's last use, so it releases the failure with it.
        let last = TensorIr {
            status: TensorStatus::ReadWrite,
            ..read
        };
        let read = burn_std::reader::try_read_sync(interpreter.read_tensor_async(last.clone()))
            .expect("flex reads synchronously");
        assert!(read.is_err());
        assert_eq!(
            interpreter.read_failure(&last.id),
            Some(format!("tensor {:?} was never produced", last.id))
        );
    }

    fn read(
        interpreter: &mut TensorInterpreter<Flex>,
        tensor: &TensorIr,
    ) -> Result<TensorData, ExecutionError> {
        let tensor = TensorIr {
            status: TensorStatus::ReadOnly,
            ..tensor.clone()
        };
        burn_std::reader::try_read_sync(interpreter.read_tensor_async(tensor))
            .expect("flex reads synchronously")
    }

    fn exp(input: &TensorIr, out: u64) -> (OperationIr, TensorIr) {
        let out = TensorIr::uninit(TensorId::new(out), input.shape.clone(), DType::F32);
        let input = TensorIr {
            status: TensorStatus::ReadOnly,
            ..input.clone()
        };
        let op = OperationIr::Float(
            DType::F32,
            FloatOperationIr::Exp(burn_ir::UnaryOpIr {
                input,
                out: out.clone(),
            }),
        );
        (op, out)
    }

    /// An operation reading a tensor a failure claimed does not run: its output reports the root
    /// failure, and an operation sharing no tensor with it still runs.
    #[test]
    fn an_op_on_a_failed_tensor_is_skipped_and_its_output_reports_the_root() {
        let mut interpreter = TensorInterpreter::<Flex>::new(Default::default());
        let healthy = interpreter.register_tensor_data_desc(TensorData::from([0.0f32]));
        let failed = TensorIr::uninit(TensorId::new(u64::MAX), Shape::new([1]), DType::F32);
        interpreter.set_error(
            failed.id,
            TensorError::new(ExecutionError::with_context("the root failure")),
        );

        let (skipped, skipped_out) = exp(&failed, u64::MAX - 1);
        let (downstream, downstream_out) = exp(&skipped_out, u64::MAX - 2);
        let (independent, independent_out) = exp(&healthy, u64::MAX - 3);
        interpreter.register_op(skipped);
        interpreter.register_op(downstream);
        interpreter.register_op(independent);

        let error = read(&mut interpreter, &downstream_out).unwrap_err();
        assert!(error.to_string().contains("the root failure"), "{error}");
        read(&mut interpreter, &independent_out)
            .unwrap()
            .assert_eq(&TensorData::from([1.0f32]), true);
    }

    /// An alias of a tensor a failure claimed reads as that failure.
    #[test]
    fn an_alias_of_a_failed_tensor_carries_its_failure() {
        let mut interpreter = TensorInterpreter::<Flex>::new(Default::default());
        let failed = TensorId::new(u64::MAX);
        interpreter.set_error(
            failed,
            TensorError::new(ExecutionError::with_context("the root failure")),
        );

        let alias = TensorId::new(u64::MAX - 1);
        interpreter.register_alias(alias, failed);

        assert!(
            interpreter
                .read_failure(&alias)
                .is_some_and(|reason| reason.contains("the root failure"))
        );
    }

    #[test]
    fn an_int_cast_yields_the_element_the_client_recorded() {
        let mut interpreter = TensorInterpreter::<Flex>::new(Default::default());
        let input = interpreter.register_tensor_data_desc(TensorData::from([1i32, -2, 3]));
        let mut out = TensorIr::uninit(TensorId::new(u64::MAX), Shape::new([3]), DType::I64);

        interpreter.register_op(OperationIr::BaseInt(BaseOperationIr::Cast(CastOpIr {
            input,
            out: out.clone(),
        })));

        out.status = TensorStatus::ReadWrite;
        let data = burn_std::reader::try_read_sync(interpreter.read_tensor_async(out))
            .expect("flex reads synchronously")
            .unwrap();
        data.assert_eq(&TensorData::from([1i64, -2, 3]), true);
    }
}
