use alloc::string::{String, ToString};
use alloc::sync::Arc;
use burn_backend::backend::BackendTypes;
use burn_ir::{BackendIr, CustomOpIr, HandleContainer};
use core::any::TypeId;
use hashbrown::HashMap;

/// A handler that executes a single custom operation against a backend's tensor handles.
///
/// It reads its input tensors (and any scalar arguments) out of `handles`, runs the actual backend
/// computation on the interpreter's `device`, and registers the resulting output tensor(s) back
/// into `handles` under the ids declared by the [`CustomOpIr`]. The `device` lets a handler create
/// brand-new tensors (e.g. a custom op with no tensor inputs, like a server-side data loader). This
/// is the server-side counterpart of the client building an `OperationIr::Custom` — see
/// [`CustomOpRegistry`].
pub type CustomOpHandler<B> = Arc<
    dyn Fn(
            &mut HandleContainer<<B as BackendIr>::Handle>,
            &CustomOpIr,
            &<B as BackendTypes>::Device,
        ) + Send
        + Sync,
>;

/// A set of [custom operation handlers](CustomOpHandler), keyed by the custom op's id.
///
/// The remote backend ships custom ops to the server as `OperationIr::Custom(CustomOpIr)`; the
/// server's [`TensorInterpreter`](crate::TensorInterpreter) can't know how to execute an arbitrary
/// custom op on its own, so it looks the id up in this registry and calls the matching handler.
///
/// The registry is *owned* state (cheap to clone — handlers live behind an [`Arc`]), built once
/// before the server starts and shared read-only across every session's interpreter. There is no
/// global/static registry to manage.
#[derive(Clone)]
pub struct CustomOpRegistry<B: BackendIr> {
    handlers: HashMap<String, RegisteredHandler<B>>,
}

/// A handler with the type of the function it was registered from, which tells a repeated
/// registration of the same function apart from a different function under the same id.
#[derive(Clone)]
struct RegisteredHandler<B: BackendIr> {
    function: TypeId,
    handler: CustomOpHandler<B>,
}

/// A different handler is already registered under the custom operation's id.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CustomOpConflict {
    /// The id both handlers claim.
    pub id: String,
}

impl core::fmt::Display for CustomOpConflict {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "a different handler is already registered for custom operation `{}`",
            self.id
        )
    }
}

impl core::error::Error for CustomOpConflict {}

impl<B: BackendIr> Default for CustomOpRegistry<B> {
    fn default() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }
}

impl<B: BackendIr> CustomOpRegistry<B> {
    /// Create a new, empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the handler executed when a custom op with the given `id` reaches the interpreter.
    ///
    /// The `id` must match the one the client puts in its [`CustomOpIr`]. Registering the same id
    /// twice replaces the previous handler.
    pub fn register<F>(&mut self, id: &str, handler: F)
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        self.handlers
            .insert(id.to_string(), RegisteredHandler::new(handler));
    }

    /// Register the handler executed for `id`, unless a different handler already claims it.
    ///
    /// Registering the same function again, such as a closure from the same source location,
    /// replaces it and succeeds, so registering a set of handlers twice is harmless. A different
    /// function under the same id is refused and the registry keeps the first.
    ///
    /// Functions are told apart by their type, so handlers whose types are the same pass as one
    /// function: every `Box<dyn Fn>` and every `fn` pointer of the handler signature, and one
    /// closure registered with different captured values, which replaces the first.
    pub fn try_register<F>(&mut self, id: &str, handler: F) -> Result<(), CustomOpConflict>
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        if let Some(registered) = self.handlers.get(id)
            && registered.function != TypeId::of::<F>()
        {
            return Err(CustomOpConflict { id: id.to_string() });
        }
        self.register(id, handler);
        Ok(())
    }

    /// Get the handler registered for `id`, if any.
    pub(crate) fn get(&self, id: &str) -> Option<&CustomOpHandler<B>> {
        self.handlers.get(id).map(|registered| &registered.handler)
    }
}

impl<B: BackendIr> RegisteredHandler<B> {
    fn new<F>(handler: F) -> Self
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        Self {
            function: TypeId::of::<F>(),
            handler: Arc::new(handler),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorInterpreter;
    use burn_backend::{DType, Scalar, Shape, TensorData, ops::FloatTensorOps};
    use burn_flex::Flex;
    use burn_ir::{OperationIr, ScalarIr, TensorId, TensorIr};
    use core::slice;
    use std::sync::Mutex;

    #[test]
    fn custom_op_is_dispatched_with_scalars() {
        // A handler that scales a float tensor by a scalar argument, and records what it received so
        // the test can assert the interpreter routed the op (and its scalars) to it.
        let seen_scalars = Arc::new(Mutex::new(None));
        let seen_scalars_handler = seen_scalars.clone();

        let mut registry = CustomOpRegistry::<Flex>::new();
        registry.register("scale", move |handles, ir, _device| {
            let input = handles.get_float_tensor::<Flex>(&ir.inputs[0]);
            let factor: Scalar = ir.scalars[0].into();
            let output = Flex::float_mul_scalar(input, factor);
            handles.register_float_tensor::<Flex>(&ir.outputs[0].id, output);
            *seen_scalars_handler.lock().unwrap() = Some(ir.scalars.clone());
        });

        let mut interp = TensorInterpreter::<Flex>::with_custom_ops(Default::default(), registry);
        let input = interp.register_tensor_data_desc(TensorData::from([2.0f32, 4.0]));
        let output = TensorIr::uninit(TensorId::new(1_000_000), Shape::from([2]), DType::F32);

        let desc =
            CustomOpIr::with_scalars("scale", &[input], &[output], vec![ScalarIr::Float(3.0)]);
        interp.register_op(OperationIr::Custom(desc));

        // The interpreter routed the op to our handler, carrying the scalar we shipped. (The handler
        // also read its input and registered an output; either failing would panic before here.)
        assert_eq!(
            seen_scalars.lock().unwrap().clone(),
            Some(vec![ScalarIr::Float(3.0)])
        );
    }

    #[test]
    fn custom_op_can_create_tensors_from_the_device() {
        // A no-input handler that builds a tensor from scratch on the interpreter's device — the
        // shape of a server-side data loader. Exercises the `&device` passed to the handler.
        use burn_backend::ops::IntTensorOps;

        let mut registry = CustomOpRegistry::<Flex>::new();
        registry.register("load", |handles, ir, device| {
            let values: Vec<i64> = ir.scalars.iter().map(|s| s.elem::<i64>()).collect();
            let n = values.len();
            let tensor = Flex::int_from_data(TensorData::new(values, [n]), device);
            handles.register_int_tensor::<Flex>(&ir.outputs[0].id, tensor);
        });

        let mut interp = TensorInterpreter::<Flex>::with_custom_ops(Default::default(), registry);
        let out = TensorIr::uninit(TensorId::new(2_000_000), Shape::from([3]), DType::I64);
        let desc = CustomOpIr::with_scalars(
            "load",
            &[],
            slice::from_ref(&out),
            vec![ScalarIr::Int(10), ScalarIr::Int(20), ScalarIr::Int(30)],
        );
        interp.register_op(OperationIr::Custom(desc));

        // The handler created the tensor on the device and registered it under the output id.
        let data = interp.get_tensor(&TensorIr {
            status: burn_ir::TensorStatus::ReadOnly,
            ..out
        });
        match data {
            burn_ir::HandleKind::Int(_) => {}
            _ => panic!("expected an int tensor"),
        }
    }

    #[test]
    fn try_register_refuses_a_different_handler_under_a_taken_id() {
        fn noop(
            _: &mut HandleContainer<<Flex as BackendIr>::Handle>,
            _: &CustomOpIr,
            _: &<Flex as BackendTypes>::Device,
        ) {
        }
        let mut registry = CustomOpRegistry::<Flex>::new();
        registry.try_register("op", noop).unwrap();
        // The same function again is a repeated registration, not a conflict.
        registry.try_register("op", noop).unwrap();
        assert_eq!(
            registry.try_register("op", |_, _, _| {}),
            Err(CustomOpConflict { id: "op".into() })
        );
    }

    #[test]
    #[should_panic(expected = "No custom-op handler registered")]
    fn unregistered_custom_op_panics() {
        let mut interp = TensorInterpreter::<Flex>::new(Default::default());
        let desc = CustomOpIr::new("missing", &[], &[]);
        interp.register_op(OperationIr::Custom(desc));
    }
}
