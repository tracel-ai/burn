use super::ParamId;
use alloc::boxed::Box;
use alloc::format;
use burn_common::stub::RwLock;
use core::cell::OnceCell;
use core::ops::Deref;

/// Parameters are the fundamental building blocks of [modules](crate::module::Module) where they
/// serve as containers for [tensors](crate::tensor::Tensor) that can be updated during
/// training, and loaded during inference. If you don't want to save the tensors with a record
/// and/or don't want to update it during training, you don't need this type to wrap your tensor.
///
/// # Laziness
///
/// The initialization of parameters can be lazy when created using
/// [uninitialized](Self::uninitialized), which can be done using an [initializer](crate::nn::Initializer).
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
}

impl<T: Parameter> core::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T: Parameter> core::fmt::Debug for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
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
    init: Box<dyn Fn(&P::Device, bool) -> P + Send>,
    device: P::Device,
    is_require_grad: bool,
}

impl<P: Parameter> Uninitialized<P> {
    fn initialize(&self) -> P {
        let init = &self.init;
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
        }
    }

    /// Create a new parameter that is not already initialized.
    pub fn uninitialized<F>(id: ParamId, init: F, device: T::Device, is_require_grad: bool) -> Self
    where
        F: Fn(&T::Device, bool) -> T + Send + 'static,
    {
        Self {
            id,
            state: OnceCell::new(),
            initialization: Some(RwLock::new(Some(Uninitialized {
                init: Box::new(init),
                device,
                is_require_grad,
            }))),
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
                let state = result.as_ref().expect("Should exist when not initialized");
                let tensor = state.initialize();

                *result = None;

                tensor
            })
            .clone()
    }

    /// Gets the parameter's value while consuming the parameter.
    pub fn into_value(self) -> T {
        self.consume().1
    }

    /// Gets the parameter id and value while consuming the parameter.
    pub fn consume(self) -> (ParamId, T) {
        let tensor = self.val();

        core::mem::drop(self.state);

        (self.id, tensor)
    }

    /// Execute the given function on the inner value.
    pub fn map<F: Fn(T) -> T>(self, func: F) -> Self {
        let (id, tensor) = self.consume();
        let tensor = func(tensor);

        Self {
            id,
            state: OnceCell::from(tensor),
            initialization: None,
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
        Param::initialized(self.id.clone(), self.val())
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

            let state = result.as_ref().expect("Should exist when not initialized");
            let tensor = state.initialize();

            *result = None;

            tensor
        })
    }
}
