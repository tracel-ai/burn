use super::ParamId;
use super::sync_once_cell::SyncOnceCell;
use alloc::format;

use alloc::boxed::Box;
use burn_std::stub::RwLock;
use burn_tensor::{Device, Shape};
use core::ops::Deref;

#[cfg(target_has_atomic = "ptr")]
use alloc::sync::Arc;

#[cfg(not(target_has_atomic = "ptr"))]
use portable_atomic_util::Arc;

#[cfg(target_has_atomic = "ptr")]
type Mapper<T> = Arc<dyn Fn(T) -> T + Send + Sync>;

#[cfg(not(target_has_atomic = "ptr"))]
type Mapper<T> = Arc<Box<dyn Fn(T) -> T + Send + Sync>>;

#[cfg(target_has_atomic = "ptr")]
fn new_mapper<T, F: Fn(T) -> T + Send + Sync + 'static>(func: F) -> Mapper<T> {
    Arc::new(func)
}

#[cfg(not(target_has_atomic = "ptr"))]
fn new_mapper<T, F: Fn(T) -> T + Send + Sync + 'static>(func: F) -> Mapper<T> {
    Arc::new(Box::new(func))
}

type InitFn<P> = Box<dyn FnOnce(&Device, bool) -> P + Send + Sync>;

fn new_init_fn<P: Parameter, F: FnOnce(&Device, bool) -> P + Send + Sync + 'static>(
    func: F,
) -> InitFn<P> {
    Box::new(func)
}

/// Coordinates lazy initialization across all clones of a [`Param`].
///
/// The sole purpose of this shared state is to ensure the initialization function runs at most
/// once: whichever clone first calls [`val`](Param::val) initializes the value, and all other
/// clones observe the same result.
///
/// # State Management
///
/// **Two logical states:**
///
/// 1. **Initialized**: `value` contains the parameter value and `initialization` is `None`.
/// 2. **Lazily Managed**: `initialization` contains `Some(RwLock<...>)`.
///    - *Before initialization*: `value` is empty, inner option is `Some(Uninitialized<T>)`.
///    - *After initialization*: `value` contains the parameter value, inner option is `None`.
///
/// The transition from uninitialized to initialized happens exactly once and is synchronized
/// across all clones.
pub(crate) struct LazyInitState<T: Parameter> {
    /// The SyncOnceCell holding the initialized parameter value.
    /// Empty for uninitialized parameters, populated after first access or explicit initialization.
    pub value: SyncOnceCell<T>,
    /// The deferred initialization state for lazy parameters.
    ///
    /// **State Transitions:**
    /// - Initialized params: `None`
    /// - Uninitialized params: `Some(RwLock<Some(Uninitialized<T>)>)`
    /// - After lazy init triggers: `Some(RwLock<None>)` (inner Option is taken)
    pub initialization: Option<RwLock<Option<Uninitialized<T>>>>,
}

impl<T: Parameter> LazyInitState<T> {
    /// Create a new parameter state that is already initialized.
    fn initialized(value: T) -> Arc<Self> {
        Arc::new(Self {
            value: SyncOnceCell::initialized(value),
            initialization: None,
        })
    }

    /// Create a new parameter state that is not already initialized.
    fn uninitialized(uninit: Uninitialized<T>) -> Arc<Self> {
        Arc::new(Self {
            value: SyncOnceCell::new(),
            initialization: Some(RwLock::new(Some(uninit))),
        })
    }

    /// Gets the parameter value, initializing it lazily if needed.
    fn val(&self) -> &T {
        self.value.get_or_init(|| {
            let mut init = self
                .initialization
                .as_ref()
                .expect("Should have an initialization when no state provided.")
                .write()
                .unwrap();
            let state = init.take().expect("Should exist when not initialized");
            state.initialize()
        })
    }
}

/// Parameters are the fundamental building blocks of [modules](crate::module::Module) where they
/// serve as containers for [tensors](crate::tensor::Tensor) that can be updated during
/// training, and loaded during inference. If you don't want to save the tensors
/// and/or don't want to update it during training, you don't need this type to wrap your tensor.
///
/// # Cloning
///
/// Cloning a parameter is always cheap; it never allocates or initializes tensors.
/// Clones share the same lazy initialization state, so initialization happens at most once and
/// all clones resolve to the same value regardless of which one triggered it.
///
/// This sharing is strictly scoped to lazy initialization. It only guarantees that all clones
/// observe the same initialization result. Subsequent transformations operate on independent
/// parameter values and never propagate across clones.
pub struct Param<T: Parameter> {
    /// The unique ID of this parameter. This is used by eg. optimizers to associate a gradient with a specific parameter.
    pub id: ParamId,
    /// Shared lazy initialization state across all clones of this parameter.
    /// The `Arc` exists solely to coordinate lazy initialization. It is not a general
    /// shared-ownership mechanism. Any mutation forks into a new `LazyInitState`.
    pub(crate) state: Arc<LazyInitState<T>>,
    pub(crate) param_mapper: ParamMapper<T>,
    // For stateful `module.valid()` <> `module.train()`
    pub(crate) require_grad: bool,
}

#[derive(Clone)]
/// Applies transformations when loading and saving parameters.
///
/// # Mapper System
///
/// `ParamMapper<T>` allows applying transformations during serialization and deserialization:
/// - `load: Option<Mapper<T>>` - transformation during deserialization (applied in `transform_for_load()`)
/// - `save: Option<Mapper<T>>` - transformation during serialization (applied in `transform_for_save()`)
///
/// These are commonly used for:
/// - Quantization/dequantization
/// - Precision conversion (e.g., FP32 ↔ FP16)
/// - Custom parameter transformations
pub struct ParamMapper<T: Parameter> {
    load: Option<Mapper<T>>,
    save: Option<Mapper<T>>,
}

impl<T: Parameter> core::fmt::Debug for ParamMapper<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "ParamMapper {{ load: {}, save: {} }}",
            self.load.is_some(),
            self.save.is_some(),
        ))
    }
}

impl<T: Parameter> ParamMapper<T> {
    /// Applies the transformation when loading the given parameter.
    pub fn on_load(&self, param: T) -> T {
        match &self.load {
            Some(mapper) => mapper(param),
            None => param,
        }
    }
    /// Applies the transformation when saving the given parameter.
    pub fn on_save(&self, param: T) -> T {
        match &self.save {
            Some(mapper) => mapper(param),
            None => param,
        }
    }
}

impl<T: Parameter> Default for ParamMapper<T> {
    fn default() -> Self {
        Self {
            load: None,
            save: None,
        }
    }
}

impl<T: Parameter> core::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T: Parameter> core::fmt::Debug for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {} - {:?}", self.id, self.param_mapper).as_str())
    }
}

pub(crate) mod sealed {
    pub trait Sealed {}
}

/// Trait that defines what is necessary for a type to be a parameter.
///
/// # Notes
/// This trait is intentionally sealed to keep the set of parameters closed.
///
/// Although exposed publicly, parameter types are not meant to be extensible:
/// the parameter loading/saving, module system and optimizers assume a fixed,
/// closed set of parameter types represented exclusively by [`Tensor`](crate::Tensor) instances.
pub trait Parameter: sealed::Sealed + Clone + core::fmt::Debug + Send {
    /// Fetch the device.
    fn device(&self) -> Device;

    /// Fetch the gradient requirement.
    fn is_require_grad(&self) -> bool;

    /// Set the gradient requirement.
    fn set_require_grad(self, require_grad: bool) -> Self;

    /// Fetch the shape of the parameter.
    fn shape(&self) -> Shape;

    /// Moves the parameter to the target device if it is not already on it,
    /// applying any kind-specific preparation required for the loading lifecycle (e.g. detach).
    fn load_to_device(self, device: &Device) -> Self;
}

/// The deferred initialization state for lazy parameters.
#[allow(clippy::type_complexity)]
pub(crate) struct Uninitialized<P: Parameter> {
    /// The initialization function. Called with `(device, is_require_grad) -> Parameter`.
    init: InitFn<P>,
    /// The target device on which the parameter should be initialized.
    /// Used by `lazy_device()` to provide device information without triggering initialization.
    pub(crate) device: Device,
    /// The gradient requirement for the parameter.
    /// Used by `lazy_is_require_grad()` to provide gradient settings without triggering initialization.
    pub(crate) is_require_grad: bool,
    /// The shape of the tensor parameter.
    /// Used by `lazy_shape()` to provide shape information without triggering initialization.
    pub(crate) shape: Shape,
}

impl<P: Parameter> Uninitialized<P> {
    /// Runs the initialization function.
    ///
    /// This is called by [Param::val] when accessing an uninitialized parameter for the first time.
    /// The function is given the stored device and gradient requirement, and returns the initialized parameter.
    fn initialize(self) -> P {
        (self.init)(&self.device, self.is_require_grad)
    }
}

impl<T: Parameter> Param<T> {
    /// Create a new parameter that is already initialized.
    pub fn initialized(id: ParamId, value: T) -> Self {
        let require_grad = value.is_require_grad();
        Self {
            id,
            state: LazyInitState::initialized(value),
            param_mapper: Default::default(),
            require_grad,
        }
    }

    /// Create a new parameter that is not already initialized.
    pub fn uninitialized<F>(
        id: ParamId,
        init: F,
        device: Device,
        is_require_grad: bool,
        shape: Shape,
    ) -> Self
    where
        F: FnOnce(&Device, bool) -> T + Send + Sync + 'static,
    {
        Self {
            id,
            state: LazyInitState::uninitialized(Uninitialized {
                init: new_init_fn(init),
                device,
                is_require_grad,
                shape,
            }),
            param_mapper: Default::default(),
            require_grad: is_require_grad,
        }
    }

    /// Gets the parameter value, initializing it lazily if needed.
    ///
    /// For initialized parameters, this returns a clone of the cached value.
    /// For uninitialized parameters, this triggers initialization:
    pub fn val(&self) -> T {
        self.deref().clone()
    }

    /// Check if the parameter has been initialized.
    ///
    /// Returns `true` if the parameter's value has been computed and cached,
    /// `false` if it's still lazy and will be initialized on first access.
    pub fn is_initialized(&self) -> bool {
        self.state.value.get().is_some()
    }

    /// Gets the parameter's value while consuming the parameter.
    pub fn into_value(self) -> T {
        self.consume().1
    }

    /// Gets the parameter id and value while consuming the parameter.
    pub fn consume(self) -> (ParamId, T, ParamMapper<T>) {
        let tensor = self.val();

        core::mem::drop(self.state);

        (self.id, tensor, self.param_mapper)
    }

    /// Execute the given function on the inner value.
    pub fn map<F: FnOnce(T) -> T>(self, func: F) -> Self {
        let (id, tensor, param_mapper) = self.consume();
        let tensor = func(tensor);
        let require_grad = tensor.is_require_grad();

        Self {
            id,
            state: LazyInitState::initialized(tensor),
            param_mapper,
            require_grad,
        }
    }

    /// Create an initialized parameter with the given id, value, and param mapper.
    ///
    /// This is a helper method for creating parameters while preserving the param mapper,
    /// typically used in ModuleMapper implementations.
    pub fn from_mapped_value(id: ParamId, value: T, param_mapper: ParamMapper<T>) -> Self {
        let require_grad = value.is_require_grad();
        Self {
            id,
            state: LazyInitState::initialized(value),
            param_mapper,
            require_grad,
        }
    }

    /// Runs a transformation on the parameter when loading.
    pub fn load_mapper<F: Fn(T) -> T + Send + Sync + 'static>(mut self, func: F) -> Self {
        self.param_mapper.load = Some(new_mapper(func));

        self
    }

    /// Runs a transformation on the parameter when saving.
    pub fn save_mapper<F: Fn(T) -> T + Send + Sync + 'static>(mut self, func: F) -> Self {
        self.param_mapper.save = Some(new_mapper(func));

        self
    }

    /// Returns a new parameter whose initialization value is transformed by the given function.
    ///
    /// If the parameter is still uninitialized (lazy), the transformation is chained onto the
    /// existing initialization without triggering evaluation. If the parameter is already
    /// initialized, it immediately applies the transformation to the current value.
    pub fn init_mapper<F: Fn(T) -> T + Send + Sync + 'static>(self, func: F) -> Self
    where
        T: Sync + 'static,
    {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => return self.map(func),
        };

        let mut init = initialization.write().unwrap();

        match init.as_mut() {
            Some(value) => {
                let device = value.device.clone();
                let is_require_grad = value.is_require_grad;
                let shape = value.shape.clone();
                core::mem::drop(init);

                let base = self;
                Self {
                    id: base.id,
                    param_mapper: base.param_mapper.clone(),
                    require_grad: base.require_grad,
                    state: LazyInitState::uninitialized(Uninitialized {
                        // (device, require_grad) are already encoded in `Uninitialized` state and
                        // applied when `base.val()` triggers initialization. The transformed tensor
                        // inherits those settings automatically, but since the mapper function
                        // `F: Fn(T) -> T` is applied on the tensor, we need to ensure the require
                        // grad setting is preserved.
                        init: new_init_fn(move |_a, b| func(base.val()).set_require_grad(b)),
                        device,
                        is_require_grad,
                        shape,
                    }),
                }
            }
            None => {
                core::mem::drop(init);
                self.map(func)
            }
        }
    }

    /// The device on which the parameter is or will be initialized, **without triggering initialization**.
    ///
    /// This is critical for the load optimization: when loading tensors into an uninitialized parameter,
    /// we need to know the target device to move the loaded tensor appropriately, but we don't want to
    /// trigger the initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::device] when you need the device but want to
    /// preserve lazy initialization.
    pub fn lazy_device(&self) -> Device {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => return self.device(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.device.clone(),
            None => self.device(),
        }
    }

    /// The gradient requirement on which the parameter is or will be initialized, **without triggering initialization**.
    ///
    /// Similar to [lazy_device](Self::lazy_device), this is critical for the load optimization.
    /// When loading tensors into an uninitialized parameter, we need to apply the correct gradient
    /// setting to the loaded tensor without triggering the initialization function.
    ///
    /// # Notes
    ///
    /// This is a crate-private function, since users are not expected to use `is_require_grad` of an
    /// uninitialized module to then override its value. All low-level functions should be provided
    /// by `burn` and should handle those details.
    pub(crate) fn lazy_is_require_grad(&self) -> bool {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => return self.is_require_grad(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.is_require_grad,
            None => self.is_require_grad(),
        }
    }

    /// Override the gradient requirement for the current parameter.
    pub fn set_require_grad(self, require_grad: bool) -> Self {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => return self.map(|tensor| tensor.set_require_grad(require_grad)),
        };

        let mut init = initialization.write().unwrap();
        let mut is_lazy = false;

        if let Some(value) = init.as_mut() {
            is_lazy = true;
            value.is_require_grad = require_grad;
        };

        core::mem::drop(init);

        if is_lazy {
            return self;
        }

        self.map(|tensor| tensor.set_require_grad(require_grad))
    }

    /// The shape of the parameter, **without triggering initialization**.
    ///
    /// This is critical for shape validation during loading: when applying tensors to an
    /// uninitialized parameter, we need to validate the shape without triggering the
    /// initialization function (which would allocate an unnecessary tensor).
    ///
    /// Use this instead of [crate::tensor::Tensor::shape] when you need the shape but want to
    /// preserve lazy initialization.
    pub fn lazy_shape(&self) -> burn_tensor::Shape {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => return self.shape(),
        };

        let init = initialization.read().unwrap();

        match init.as_ref() {
            Some(value) => value.shape.clone(),
            None => self.shape(),
        }
    }

    /// Transform a parameter for loading by applying load transformations.
    ///
    /// This method is used to restore a parameter from a tensor (typically during deserialization).
    /// It ensures the tensor is moved to the expected device, applies the param mapper's
    /// `on_load` transformation, and preserves the autodiff settings (require_grad).
    pub fn transform_for_load(self, tensor: T, param_id: ParamId) -> Self {
        let mut new_tensor = tensor;

        let mapper = self.param_mapper.clone();

        let expected_device = self.lazy_device();
        let expected_require_grad = self.lazy_is_require_grad();

        // Make sure we load the tensor into the same module device.
        new_tensor = new_tensor.load_to_device(&expected_device);

        new_tensor = mapper.on_load(new_tensor);

        // Make sure we load the tensor with the same autodiff setting.
        new_tensor = new_tensor.set_require_grad(expected_require_grad);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded
    }

    /// Transform a parameter for saving by applying save transformations.
    ///
    /// This method is used to prepare a parameter for saving (typically during serialization).
    /// It applies the param mapper's `on_save` transformation, which can be used
    /// to modify the tensor before serialization (e.g., quantization, precision conversion).
    pub fn transform_for_save(&self) -> Self {
        let mut tensor = self.val();
        let mapper = self.param_mapper.clone();

        tensor = mapper.on_save(tensor);

        Self::initialized(self.id, tensor)
    }
}

impl<T: Parameter> Clone for Param<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            state: self.state.clone(),
            param_mapper: self.param_mapper.clone(),
            require_grad: self.require_grad,
        }
    }
}

impl<T: Parameter> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.state.val()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn_tensor::Tensor;

    // Param<T> should be Sync so that models can be shared across threads
    // (e.g. parallel inference with rayon).
    fn _assert_sync<T: Sync>() {}

    #[test]
    fn param_is_sync() {
        fn check() {
            _assert_sync::<Param<Tensor<2>>>();
        }
        check();
    }

    /// Concurrent lazy initialization must not panic.
    ///
    /// Multiple threads call `val()` on an uninitialized `Param` simultaneously.
    /// `SyncOnceCell::get_or_init` guarantees only one thread runs the initializer;
    /// the others block and receive the same value.
    #[cfg(feature = "std")]
    #[test]
    fn param_concurrent_lazy_init() {
        use alloc::vec::Vec;

        let device = Default::default();

        let param: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, _require_grad| Tensor::random([2, 3], Default::default(), device),
            device,
            false,
            [2, 3].into(),
        );

        // Share across threads via &param (requires Sync).
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..4).map(|_| s.spawn(|| param.val())).collect();

            let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();

            // All threads must get the same value.
            let expected = results[0].to_data();
            for result in &results[1..] {
                assert_eq!(result.to_data(), expected);
            }
        });
    }

    #[test]
    fn param_clones_share_lazy_initialization() {
        let device = Default::default();

        // We use random values so that if it initializes twice, the data will mismatch.
        let param_original: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, _require_grad| Tensor::random([2, 3], Default::default(), device),
            device,
            false,
            [2, 3].into(),
        );

        // Regression: https://github.com/tracel-ai/burn/issues/5040
        // Clone the parameter while it is still uninitialized.
        // Previously, this would clone the init function only, leading to different parameter states.
        let param_clone = param_original.clone();

        let tensor_original = param_original.val();
        assert!(param_original.is_initialized());
        assert!(param_clone.is_initialized());

        let tensor_clone = param_clone.val();

        tensor_original
            .into_data()
            .assert_eq(&tensor_clone.into_data(), true);
    }

    #[test]
    fn param_set_require_grad_forks_from_shared_state() {
        let device = Default::default();

        let param1: Param<Tensor<2>> = Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 3], device).set_require_grad(require_grad),
            device,
            true,
            [2, 3].into(),
        );

        // Clone param; both now point to the exact same Arc<LazyInitState>
        let param2 = param1.clone();

        // Force initialization via the first clone.
        let _tensor1 = param1.val();
        assert!(param1.is_initialized());
        assert!(param2.is_initialized());

        // set_require_grad intentionally forks: param2 gets a new Arc with the mutated tensor.
        let param2 = param2.set_require_grad(false);

        // The fork produced the correct require_grad state.
        assert_eq!(param2.require_grad, false);
        assert_eq!(param1.require_grad, true); // param1 is unaffected

        // Values are still identical (same tensor data, different grad setting).
        param1
            .val()
            .into_data()
            .assert_eq(&param2.val().into_data(), true);
    }
}
