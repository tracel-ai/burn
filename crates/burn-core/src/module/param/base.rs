use super::ParamId;
use alloc::{boxed::Box, format};
use burn_std::DType;
use burn_std::stub::RwLock;
use burn_tensor::Shape;
use core::cell::OnceCell;
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

/// Parameters are the fundamental building blocks of [modules](crate::module::Module) where they
/// serve as containers for [tensors](crate::tensor::Tensor) that can be updated during
/// training, and loaded during inference. If you don't want to save the tensors
/// and/or don't want to update it during training, you don't need this type to wrap your tensor.
///
/// # Core Lazy Initialization Architecture
///
/// `Param<T>` has a dual-state design using `OnceCell<T>`:
///
/// ## State Management
///
/// **Two possible states:**
///
/// 1. **Initialized**: `state: OnceCell<T>` contains value, `initialization: None`
/// 2. **Uninitialized (Lazy)**: `state` is empty, `initialization: Some(RwLock<Option<Uninitialized<T>>>)`
pub struct Param<T: Parameter> {
    /// The unique ID of this parameter. This is used by eg. optimizers to associate a gradient with a specific parameter.
    pub id: ParamId,
    /// The OnceCell holding the initialized parameter value.
    /// Empty for uninitialized parameters, populated after first access or explicit initialization.
    pub(crate) state: OnceCell<T>,
    /// The deferred initialization state for lazy parameters.
    ///
    /// **State Transitions:**
    /// - Initialized params: `None`
    /// - Uninitialized params: `Some(RwLock<Some(Uninitialized<T>)>)`
    /// - After lazy init triggers: `Some(RwLock<None>)` (inner Option is taken)
    pub(crate) initialization: Option<RwLock<Option<Uninitialized<T>>>>,
    pub(crate) param_mapper: ParamMapper<T>,
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
            self.save.is_some()
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

/// Trait that defines what is necessary for a type to be a parameter.
pub trait Parameter: Clone + core::fmt::Debug + Send {
    /// The device type to be used.
    type Device: Clone;

    /// Fetch the device.
    fn device(&self) -> Self::Device;

    /// Fetch the gradient requirement.
    fn is_require_grad(&self) -> bool;

    /// Set the gradient requirement.
    fn set_require_grad(self, require_grad: bool) -> Self;
}

/// The deferred initialization state for lazy parameters.
#[allow(clippy::type_complexity)]
pub(crate) struct Uninitialized<P: Parameter> {
    /// The initialization function. Called with `(device, is_require_grad) -> Parameter`.
    /// This function is consumed during initialization via `FnOnce`.
    init: Box<dyn FnOnce(&P::Device, Option<DType>, bool) -> P + Send>,
    /// The target device on which the parameter should be initialized.
    /// Used by `lazy_device()` to provide device information without triggering initialization.
    pub(crate) device: P::Device,
    pub(crate) dtype: Option<DType>,
    /// The gradient requirement for the parameter.
    /// Used by `lazy_is_require_grad()` to provide gradient settings without triggering initialization.
    pub(crate) is_require_grad: bool,
    /// The shape of the tensor parameter.
    /// Used by `lazy_shape()` to provide shape information without triggering initialization.
    pub(crate) shape: Shape,
}

impl<P: Parameter> Uninitialized<P> {
    /// Consumes the uninitialized state and runs the initialization function.
    ///
    /// This is called by [Param::val] when accessing an uninitialized parameter for the first time.
    /// The function is given the stored device and gradient requirement, and returns the initialized parameter.
    fn initialize(self) -> P {
        let init = self.init;
        init(&self.device, self.dtype, self.is_require_grad)
    }
}

impl<T: Parameter> Param<T> {
    /// Create a new parameter that is already initialized.
    pub fn initialized(id: ParamId, value: T) -> Self {
        Self {
            id,
            state: OnceCell::from(value),
            initialization: None,
            param_mapper: Default::default(),
        }
    }

    /// Create a new parameter that is not already initialized.
    pub fn uninitialized<F>(
        id: ParamId,
        init: F,
        device: T::Device,
        is_require_grad: bool,
        shape: Shape,
    ) -> Self
    where
        F: FnOnce(&T::Device, Option<DType>, bool) -> T + Send + 'static,
    {
        Self {
            id,
            state: OnceCell::new(),
            initialization: Some(RwLock::new(Some(Uninitialized {
                init: Box::new(init),
                device,
                dtype: None,
                is_require_grad,
                shape,
            }))),
            param_mapper: Default::default(),
        }
    }

    /// Gets the parameter value, initializing it lazily if needed.
    ///
    /// For initialized parameters, this returns a clone of the cached value.
    /// For uninitialized parameters, this triggers initialization:
    pub fn val(&self) -> T {
        self.state
            .get_or_init(|| {
                let mut result = self
                    .initialization
                    .as_ref()
                    .expect("Should have an initialization when no state provided.")
                    .write()
                    .unwrap();
                let state = result.take().expect("Should exist when not initialized");
                state.initialize()
            })
            .clone()
    }

    /// Check if the parameter has been initialized.
    ///
    /// Returns `true` if the parameter's value has been computed and cached,
    /// `false` if it's still lazy and will be initialized on first access.
    pub fn is_initialized(&self) -> bool {
        self.state.get().is_some()
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

        Self {
            id,
            state: OnceCell::from(tensor),
            initialization: None,
            param_mapper,
        }
    }

    /// Create an initialized parameter with the given id, value, and param mapper.
    ///
    /// This is a helper method for creating parameters while preserving the param mapper,
    /// typically used in ModuleMapper implementations.
    pub fn from_mapped_value(id: ParamId, value: T, param_mapper: ParamMapper<T>) -> Self {
        Self {
            id,
            state: OnceCell::from(value),
            initialization: None,
            param_mapper,
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

    /// Execute the given function on the inner value.
    pub fn init_mapper<F: FnOnce(T) -> T + Send + 'static>(self, func: F) -> Self
    where
        T: 'static,
    {
        let initialization = match &self.initialization {
            Some(init) => init,
            None => return self.map(func),
        };

        let mut init = initialization.write().unwrap();

        match init.as_mut() {
            Some(value) => {
                #[allow(clippy::type_complexity)]
                let mut prev: Box<
                    dyn FnOnce(&T::Device, Option<DType>, bool) -> T + Send,
                > = Box::new(|_, _, _| panic!("Fake func to not have null ref."));
                core::mem::swap(&mut prev, &mut value.init);

                value.init = Box::new(|a, b, c| {
                    let tensor = prev(a, b, c);
                    func(tensor)
                });
                core::mem::drop(init);
                self
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
    pub fn lazy_device(&self) -> T::Device {
        let initialization = match &self.initialization {
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
        let initialization = match &self.initialization {
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
        let initialization = match &self.initialization {
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
}

impl<T: Parameter> Clone for Param<T> {
    fn clone(&self) -> Self {
        let mut param = Param::initialized(self.id, self.val());
        param.param_mapper = self.param_mapper.clone();
        param
    }
}

impl<T: Parameter> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.state.get_or_init(|| {
            let mut result = self
                .initialization
                .as_ref()
                .expect("Should have an initialization when no state provided.")
                .write()
                .unwrap();

            let state = result.take().expect("Should exist when not initialized");
            state.initialize()
        })
    }
}
