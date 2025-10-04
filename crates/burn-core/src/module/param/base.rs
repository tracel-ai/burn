use super::ParamId;
use alloc::{boxed::Box, format};
use burn_common::stub::RwLock;
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
/// training, and loaded during inference. If you don't want to save the tensors with a record
/// and/or don't want to update it during training, you don't need this type to wrap your tensor.
///
/// # Laziness
///
/// The initialization of parameters can be lazy when created using
/// [uninitialized](Self::uninitialized), which can be done using an [initializer](crate::module::Initializer).
///
/// This reduces the amount of allocations done when loading a model for inference without having
/// to create a custom initialization function only for inference.
///
/// ## Example
///
/// ```rust, ignore
/// let device = Device::default();
/// let config = ModuleConfig::default();
/// let record = Recorder::new().load("/path/to/module", &device);
///
/// // No tensor allocation
/// let module = config.init(device);
/// // Will use the tensor allocated for the record if the same device is used.
/// let module = module.load_record(record);
/// ```
pub struct Param<T: Parameter> {
    /// The unique ID of this parameter. This is used by eg. optimizers to associate a gradient with a specific parameter.
    pub id: ParamId,
    state: OnceCell<T>,
    /// The locking is only required because of `lazy_device` and `lazy_is_require_grad`.
    ///
    /// Because of once cell, we have a guarantee that the initialization will only be called once,
    /// but it may be called at the same time as `lazy_device` and `lazy_is_require_grad`, which is
    /// when the lock is actually useful, waiting for the initialization to be completed before
    /// returning the value.
    initialization: Option<RwLock<Option<Uninitialized<T>>>>,
    pub(crate) param_mapper: ParamMapper<T>,
}

#[derive(Clone)]
/// Applies functions when loading and saving parameters.
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

#[allow(clippy::type_complexity)]
struct Uninitialized<P: Parameter> {
    init: Box<dyn FnOnce(&P::Device, bool) -> P + Send>,
    device: P::Device,
    is_require_grad: bool,
}

impl<P: Parameter> Uninitialized<P> {
    fn initialize(self) -> P {
        let init = self.init;
        init(&self.device, self.is_require_grad)
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
    pub fn uninitialized<F>(id: ParamId, init: F, device: T::Device, is_require_grad: bool) -> Self
    where
        F: FnOnce(&T::Device, bool) -> T + Send + 'static,
    {
        Self {
            id,
            state: OnceCell::new(),
            initialization: Some(RwLock::new(Some(Uninitialized {
                init: Box::new(init),
                device,
                is_require_grad,
            }))),
            param_mapper: Default::default(),
        }
    }

    /// Gets the parameter value.
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
    pub fn into_initialized(id: ParamId, value: T, param_mapper: ParamMapper<T>) -> Self {
        Self {
            id,
            state: OnceCell::from(value),
            initialization: None,
            param_mapper,
        }
    }

    /// Runs a transformation on the parameter when loading a saved record.
    pub fn load_mapper<F: Fn(T) -> T + Send + Sync + 'static>(mut self, func: F) -> Self {
        self.param_mapper.load = Some(new_mapper(func));

        self
    }

    /// Runs a transformation on the parameter when saving the record.
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
                let mut prev: Box<dyn FnOnce(&T::Device, bool) -> T + Send> =
                    Box::new(|_, _| panic!("Fake func to not have null ref."));
                core::mem::swap(&mut prev, &mut value.init);

                value.init = Box::new(|a, b| {
                    let tensor = prev(a, b);
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

    /// The device on which the parameter is or will be initialized.
    ///
    /// This should be used instead of [crate::tensor::Tensor::device], since using the tensor
    /// function requires a dereference, which triggers the initialization. This is only useful
    /// when the device is used for updating the tensor value, which has potentially not been
    /// initialized yet, like loading a record.
    ///
    /// # Notes
    ///
    /// This is a crate-private function, since users are not expected to use the device of an
    /// uninitialized module to then override its value. All low-level functions should be provided
    /// by `burn` and should handle those details.
    pub(crate) fn lazy_device(&self) -> T::Device {
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

    /// The gradient requirement on which the parameter is or will be initialized.
    ///
    /// This should be used instead of [crate::tensor::Tensor::is_require_grad], since using the tensor
    /// function requires a dereference, which triggers the initialization. This is only useful
    /// when the boolean is used for updating the tensor value, which has potentially not been
    /// initialized yet, like loading a record.
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
