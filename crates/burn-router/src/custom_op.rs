use alloc::string::{String, ToString};
use alloc::sync::Arc;
use burn_ir::{BackendIr, CustomOpIr, HandleContainer};
use hashbrown::HashMap;

/// A handler that executes a single custom operation against a backend's tensor handles.
///
/// It reads its input tensors (and any scalar arguments) out of `handles`, runs the actual backend
/// computation, and registers the resulting output tensor(s) back into `handles` under the ids
/// declared by the [`CustomOpIr`]. This is the server-side counterpart of the client building an
/// `OperationIr::Custom` — see [`CustomOpRegistry`].
pub type CustomOpHandler<B> =
    Arc<dyn Fn(&mut HandleContainer<<B as BackendIr>::Handle>, &CustomOpIr) + Send + Sync>;

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
    handlers: HashMap<String, CustomOpHandler<B>>,
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
        F: Fn(&mut HandleContainer<B::Handle>, &CustomOpIr) + Send + Sync + 'static,
    {
        self.handlers.insert(id.to_string(), Arc::new(handler));
    }

    /// Get the handler registered for `id`, if any.
    pub(crate) fn get(&self, id: &str) -> Option<&CustomOpHandler<B>> {
        self.handlers.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorInterpreter;
    use burn_backend::{DType, Scalar, Shape, TensorData, ops::FloatTensorOps};
    use burn_flex::Flex;
    use burn_ir::{OperationIr, ScalarIr, TensorId, TensorIr};
    use std::sync::Mutex;

    #[test]
    fn custom_op_is_dispatched_with_scalars() {
        // A handler that scales a float tensor by a scalar argument, and records what it received so
        // the test can assert the interpreter routed the op (and its scalars) to it.
        let seen_scalars = Arc::new(Mutex::new(None));
        let seen_scalars_handler = seen_scalars.clone();

        let mut registry = CustomOpRegistry::<Flex>::new();
        registry.register("scale", move |handles, ir| {
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
    #[should_panic(expected = "No custom-op handler registered")]
    fn unregistered_custom_op_panics() {
        let mut interp = TensorInterpreter::<Flex>::new(Default::default());
        let desc = CustomOpIr::new("missing", &[], &[]);
        interp.register_op(OperationIr::Custom(desc));
    }
}
