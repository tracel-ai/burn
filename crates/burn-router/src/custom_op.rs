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

/// Where a handler is registered from: the type that registers it and the name it gives the
/// handler there, such as a trait's generated handler set and one of its methods.
///
/// [`CustomOpRegistry::try_register`] tells a repeated registration from a conflict by it: the
/// same source registering again replaces its handler, and another source under a taken id is
/// refused. The handler's own type cannot tell them apart, as every `fn` pointer and every
/// `Box<dyn Fn>` of the handler signature shares one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandlerSource {
    owner: TypeId,
    name: &'static str,
}

/// A handler with the source it was registered from, `None` for [`CustomOpRegistry::register`].
#[derive(Clone)]
struct RegisteredHandler<B: BackendIr> {
    source: Option<HandlerSource>,
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

impl HandlerSource {
    /// The handler `Owner` registers under `name`.
    pub fn new<Owner: ?Sized + 'static>(name: &'static str) -> Self {
        Self {
            owner: TypeId::of::<Owner>(),
            name,
        }
    }
}

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
        self.insert(id, None, handler);
    }

    /// Register the handler executed for `id` from `source`, unless another source already
    /// claims `id`.
    ///
    /// The same source registering again replaces its handler and succeeds, so registering a set
    /// of handlers twice is harmless. Another source under the same id, or a handler registered
    /// with [`register`](Self::register), is refused, and the registry keeps the first.
    ///
    /// # Errors
    ///
    /// [`CustomOpConflict`] when another source holds `id`.
    pub fn try_register<F>(
        &mut self,
        id: &str,
        source: HandlerSource,
        handler: F,
    ) -> Result<(), CustomOpConflict>
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        if let Some(registered) = self.handlers.get(id)
            && registered.source != Some(source)
        {
            return Err(CustomOpConflict { id: id.to_string() });
        }
        self.insert(id, Some(source), handler);
        Ok(())
    }

    /// Get the handler registered for `id`, if any.
    pub(crate) fn get(&self, id: &str) -> Option<&CustomOpHandler<B>> {
        self.handlers.get(id).map(|registered| &registered.handler)
    }

    fn insert<F>(&mut self, id: &str, source: Option<HandlerSource>, handler: F)
    where
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr, &B::Device) + Send + Sync + 'static,
    {
        let registered = RegisteredHandler {
            source,
            handler: Arc::new(handler),
        };
        self.handlers.insert(id.to_string(), registered);
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
    fn try_register_refuses_another_source_under_a_taken_id() {
        type Handler = fn(
            &mut HandleContainer<<Flex as BackendIr>::Handle>,
            &CustomOpIr,
            &<Flex as BackendTypes>::Device,
        );
        struct First;
        struct Second;
        // Two different functions of one `fn` pointer type, which their types cannot tell apart.
        let first: Handler = |_, _, _| {};
        let second: Handler = |_, _, _| {};
        let mut registry = CustomOpRegistry::<Flex>::new();

        registry
            .try_register("op", HandlerSource::new::<First>("op"), first)
            .unwrap();
        // The same source again is a repeated registration, not a conflict.
        registry
            .try_register("op", HandlerSource::new::<First>("op"), first)
            .unwrap();
        let conflict = Err(CustomOpConflict { id: "op".into() });
        assert_eq!(
            registry.try_register("op", HandlerSource::new::<Second>("op"), second),
            conflict
        );
        assert_eq!(
            registry.try_register("op", HandlerSource::new::<First>("other"), second),
            conflict
        );

        registry.register("plain", first);
        assert_eq!(
            registry.try_register("plain", HandlerSource::new::<First>("plain"), first),
            Err(CustomOpConflict { id: "plain".into() })
        );
    }

    /// An op with no handler claims its outputs, so the read of one reports the missing handler
    /// rather than the interpreter's caller panicking.
    #[test]
    fn an_unregistered_custom_op_claims_its_outputs() {
        let mut interp = TensorInterpreter::<Flex>::new(Default::default());
        let out = TensorIr::uninit(TensorId::new(u64::MAX), Shape::new([1]), DType::F32);
        let desc = CustomOpIr::new("missing", &[], slice::from_ref(&out));
        interp.register_op(OperationIr::Custom(desc));

        let reason = interp.read_failure(&out.id).expect("the output is claimed");
        assert!(
            reason.contains("No custom-op handler registered"),
            "{reason}"
        );
    }
}
