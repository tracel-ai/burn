use super::ParamId;
use alloc::format;
use burn_common::stub::RwLock;
use core::ops::Deref;
use once_cell::sync::OnceCell;

/// Define a parameter.
pub struct Param<T: Parameter> {
    pub(crate) id: ParamId,
    state: OnceCell<T>,
    /// The locking is only required because of `lazy_device` and `lazy_is_require_grad`.
    ///
    /// Because of once cell, we have a garantie that the initialization will only be called once,
    /// but it may be called at the same time as `lazy_device` and `lazy_is_require_grad`, which is
    /// when the lock is actually useful, waiting the the initialization to be completed before
    /// returning the value.
    init: RwLock<Option<Uninitialized<T>>>,
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

/// Trait that defines that is necessary for a type to be a parameter.
pub trait Parameter: Clone + core::fmt::Debug + Send + Sync {
    /// The device type to be used.
    type Device: Clone;

    /// Fetch the device.
    fn device(&self) -> Self::Device;
    /// Fetch the gradient requirement.
    fn is_require_grad(&self) -> bool;
}

struct Uninitialized<P: Parameter> {
    init: Box<dyn Fn(&P::Device) -> P + Send + Sync>,
    device: P::Device,
    is_require_grad: bool,
}

impl<P: Parameter> Uninitialized<P> {
    fn initialize(&self) -> P {
        let init = &self.init;
        init(&self.device)
    }
}

impl<T: Parameter> Param<T> {
    /// Create a new parameter the is already initialized.
    pub fn initialized(id: ParamId, value: T) -> Self {
        Self {
            id,
            state: OnceCell::with_value(value),
            init: RwLock::new(None),
        }
    }

    /// Create a new parameter the is not initialized.
    pub fn uninitialized<F>(id: ParamId, init: F, device: T::Device, is_require_grad: bool) -> Self
    where
        F: Fn(&T::Device) -> T + Send + Sync + 'static,
    {
        Self {
            id,
            state: OnceCell::new(),
            init: RwLock::new(Some(Uninitialized {
                init: Box::new(init),
                device,
                is_require_grad,
            })),
        }
    }

    /// Gets the parameter value.
    ///
    /// # Returns
    ///
    /// The parameter value.
    pub fn val(&self) -> T {
        self.state
            .get_or_init(|| {
                let mut result = self.init.write().unwrap();
                let state = result.as_ref().expect("Should be something.");
                let tensor = state.initialize();

                *result = None;

                tensor
            })
            .clone()
    }

    /// Gets the parameter value while consuming the parameter.
    ///
    /// # Returns
    ///
    /// The parameter value.
    pub fn consume(self) -> (ParamId, T) {
        let state = self.state.into_inner();
        let tensor = match state {
            Some(tensor) => tensor,
            None => {
                let val = self.init.write();
                val.unwrap().as_ref().unwrap().initialize()
            }
        };

        (self.id, tensor)
    }

    /// Execute the given function on the inner value.
    pub fn map<F: Fn(T) -> T>(self, func: F) -> Self {
        let (id, tensor) = self.consume();
        let tensor = func(tensor);

        Self {
            id,
            state: OnceCell::with_value(tensor),
            init: RwLock::new(None),
        }
    }

    /// The device on which the parameter is or will be initialized.
    pub(crate) fn lazy_device(&self) -> T::Device {
        let init = self.init.read().unwrap();

        match init.as_ref() {
            Some(value) => value.device.clone(),
            None => self.device(),
        }
    }

    /// The gradient requirement on which the parameter is or will be initialized.
    pub(crate) fn lazy_is_require_grad(&self) -> bool {
        let init = self.init.read().unwrap();

        match init.as_ref() {
            Some(value) => value.is_require_grad,
            None => self.is_require_grad(),
        }
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
            let mut result = self.init.write().unwrap();
            let state = result.as_ref().expect("Should be something.");
            let tensor = state.initialize();

            *result = None;

            tensor
        })
    }
}
