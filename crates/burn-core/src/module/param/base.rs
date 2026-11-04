#![allow(clippy::bool_assert_comparison)]

use crate::module::LoraAdapter;
use crate::module::Reparameterization;

use super::ParamId;
use super::reparameterization_dyn::DynReparameterization;
use super::sync_once_cell::SyncOnceCell;
use alloc::format;

use alloc::boxed::Box;
use burn_std::sync::{Arc, RwLock};
use burn_tensor::{Device, Shape, Sharding};
use core::ops::Deref;

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

type InitFn<P> = Box<dyn FnOnce(InitRequest<'_>) -> P + Send + Sync>;

fn new_init_fn<P: ParameterValue, F: FnOnce(&Device, bool) -> P + Send + Sync + 'static>(
    func: F,
) -> InitFn<P> {
    Box::new(move |request: InitRequest<'_>| request.build(func))
}

/// What a lazy parameter's initialization is asked to make.
struct InitRequest<'a> {
    device: &'a Device,
    /// The effective gradient requirement, false on a device without autodiff.
    require_grad: bool,
    /// The split of the value the initializer makes, before any init mapper changes its layout.
    sharding: Option<Sharding>,
}

impl InitRequest<'_> {
    /// The value `init` makes on the requested device, split and tracked as requested.
    fn build<P: ParameterValue>(self, init: impl FnOnce(&Device, bool) -> P) -> P {
        if let Some(sharding) = self.sharding {
            return init(self.device, false).into_sharded_leaf(sharding, self.require_grad);
        }
        init(self.device, self.require_grad)
    }
}

/// Why a lazy parameter shared with a clone and wrapped by an init mapper is not split.
const SHARED_INIT_MAPPED_REFUSAL: &str = "A lazy parameter with an init mapper is split before it \
    is cloned: initializing it here would pass the mapper, which may change its layout";

/// Where a loaded value takes its parameter's split.
#[derive(Clone, Copy)]
enum LoadSplit {
    /// A parameter not initialized yet is split in the record's layout, so before the load
    /// mapper runs any op and each member uploads only its chunk.
    BeforeMapper(Sharding),
    /// An initialized parameter's split is of its value, which the load mapper produces.
    AfterMapper(Sharding),
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
pub(crate) struct LazyInitState<T: ParameterValue> {
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

impl<T: ParameterValue> LazyInitState<T> {
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
                .write();
            let state = init.take().expect("Should exist when not initialized");
            state.initialize()
        })
    }
}

/// Parameters are identified, traversable values owned by [modules](crate::module::Module).
///
/// Most parameters contain [tensors](crate::tensor::Tensor) that can be updated during training
/// and loaded during inference. Parameters can also contain module control state such as
/// [`Flag`](crate::module::Flag), allowing it to participate in module traversal and parameter
/// group selection without pretending to be a tensor.
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
pub struct Param<T: ParameterValue> {
    /// The unique ID of this parameter. This is used by eg. optimizers to associate a gradient with a specific parameter.
    pub id: ParamId,
    /// Shared lazy initialization state across all clones of this parameter.
    /// The `Arc` exists solely to coordinate lazy initialization. It is not a general
    /// shared-ownership mechanism. Any mutation forks into a new `LazyInitState`.
    pub(crate) state: Arc<LazyInitState<T>>,
    pub(crate) param_mapper: ParamMapper<T>,
    /// Whether this value is configured to participate in training behavior.
    ///
    /// This is authoritative and kept separately from the effective value so transformations and
    /// backends that cannot currently activate the value don't erase the state that
    /// [`Module::valid`](crate::module::Module::valid) temporarily suspends and
    /// `train` restores.
    pub(crate) is_active: bool,
    /// Optional transformation that materializes the effective value from the stored base.
    pub(crate) reparameterization: Option<Box<dyn DynReparameterization>>,
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
pub struct ParamMapper<T: ParameterValue> {
    load: Option<Mapper<T>>,
    save: Option<Mapper<T>>,
    /// Carries configured training state across `consume` / `from_mapped_value` reconstruction.
    mapped_is_active: Option<bool>,
}

impl<T: ParameterValue> core::fmt::Debug for ParamMapper<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_fmt(format_args!(
            "ParamMapper {{ load: {}, save: {} }}",
            self.load.is_some(),
            self.save.is_some(),
        ))
    }
}

impl<T: ParameterValue> ParamMapper<T> {
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

impl<T: ParameterValue> Default for ParamMapper<T> {
    fn default() -> Self {
        Self {
            load: None,
            save: None,
            mapped_is_active: None,
        }
    }
}

impl<T: ParameterValue> core::fmt::Display for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {}", self.id).as_str())
    }
}

impl<T: ParameterValue> core::fmt::Debug for Param<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("Param: {} - {:?}", self.id, self.param_mapper).as_str())
    }
}

pub(crate) mod sealed {
    use super::DynReparameterization;
    use burn_tensor::Sharding;

    pub trait Sealed: Sized {
        /// Whether this value currently participates in the module's training behavior.
        fn is_active(&self) -> bool;

        /// Materialize a parameter with an attached reparameterization.
        ///
        /// # Notes
        /// This is part of the sealed trait to avoid [`DynReparameterization`] from showing up in the
        /// public `ParameterValue` trait.
        fn apply_reparameterization(self, reparameterization: &dyn DynReparameterization) -> Self {
            let _ = reparameterization;
            self
        }

        /// The value as a new leaf split as `sharding` says, tracked as `require_grad` says. A
        /// quantized value the split would cut through a quantization block or packed word stays
        /// replicated.
        fn into_sharded_leaf(self, sharding: Sharding, require_grad: bool) -> Self {
            let _ = (sharding, require_grad);
            self
        }
    }
}

/// A value that can be identified and traversed inside a [`Param`].
///
/// # Notes
/// This trait is intentionally sealed to keep the set of parameters closed.
///
/// Although exposed publicly, parameter values are not meant to be extensible. The closed set
/// includes device-backed values such as [`Tensor`](crate::Tensor) and device-independent control
/// values such as [`Flag`](crate::module::Flag).
pub trait ParameterValue: sealed::Sealed + Clone + core::fmt::Debug + Send {}

/// A tensor-backed [`ParameterValue`] with autodiff, device and loading capabilities.
///
/// General module-owned values can implement [`ParameterValue`] without pretending to require
/// gradients, reside on a device or have a tensor shape.
pub trait Parameter: ParameterValue {
    /// Fetch the gradient requirement.
    fn is_require_grad(&self) -> bool;

    /// Set the gradient requirement.
    ///
    /// Float tensors without autodiff remain unchanged. [`Param::set_require_grad`] stores
    /// the configured training state so it can be applied when entering training mode.
    fn set_require_grad(self, require_grad: bool) -> Self;

    /// Fetch the device.
    fn device(&self) -> Device;

    /// Fetch the shape of the parameter.
    fn shape(&self) -> Shape;

    /// How the value is split over its sharded compute device, or `None` when it is not on one.
    fn sharding(&self) -> Option<Sharding>;

    /// Moves the parameter to the target device if it is not already on it,
    /// applying any kind-specific preparation required for the loading lifecycle (e.g. detach).
    fn load_to_device(self, device: &Device) -> Self;
}

/// The deferred initialization state for lazy parameters.
#[allow(clippy::type_complexity)]
pub(crate) struct Uninitialized<P: ParameterValue> {
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
    /// The split of the initializer's output, before any init mapper; `None` leaves it replicated.
    pub(crate) sharding: Option<Sharding>,
    /// Whether an init mapper wraps the initializer, so the value may not be in `shape`'s layout.
    pub(crate) init_mapped: bool,
}

impl<P: ParameterValue> Uninitialized<P> {
    /// Runs the initialization function.
    ///
    /// This is called by [Param::val] when accessing an uninitialized parameter for the first time.
    /// The function receives the stored device and effective gradient requirement. On a plain
    /// device the requirement is false; the configured training state is preserved for `train()`.
    fn initialize(self) -> P {
        (self.init)(InitRequest {
            device: &self.device,
            require_grad: self.is_require_grad && self.device.is_autodiff(),
            sharding: self.sharding,
        })
    }
}

impl<T: ParameterValue> Param<T> {
    /// Create a new parameter that is already initialized.
    pub fn initialized(id: ParamId, value: T) -> Self {
        let is_active = value.is_active();
        Self {
            id,
            state: LazyInitState::initialized(value),
            param_mapper: Default::default(),
            is_active,
            reparameterization: None,
        }
    }

    /// Gets the effective parameter value, initializing it lazily if needed.
    ///
    /// For initialized parameters, this returns a clone of the cached value.
    /// For uninitialized parameters, this triggers initialization.
    ///
    /// When a reparameterization is attached, this materializes its effective value. Use
    /// [`base`](Self::base) to access the raw stored value without materialization. Conceptually,
    /// materialization composes the structural base with the attached reparameterization state.
    pub fn val(&self) -> T {
        let base = self.deref().clone();
        match &self.reparameterization {
            Some(reparameterization) => base.apply_reparameterization(reparameterization.as_ref()),
            None => base,
        }
    }

    /// Gets the raw stored parameter value **without** applying its reparameterization.
    pub fn base(&self) -> T {
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

    /// Gets the parameter id and raw value while consuming the parameter.
    ///
    /// Any attached reparameterization is dropped. During module traversal,
    /// `Param<Tensor<D>>::map` detaches the reparameterization before calling
    /// `ModuleMapper::map_float`, traverses it separately, and reattaches it afterward.
    pub fn consume(self) -> (ParamId, T, ParamMapper<T>) {
        let tensor = self.deref().clone();
        let mut param_mapper = self.param_mapper;
        param_mapper.mapped_is_active = Some(self.is_active);

        core::mem::drop(self.state);

        (self.id, tensor, param_mapper)
    }

    /// Execute the given function on the inner value while preserving its configured training
    /// state.
    ///
    /// Transforming the effective value must not implicitly change what
    /// [`Module::train`](crate::module::Module::train) restores. Use the explicit activation APIs,
    /// such as [`Param::set_require_grad`] for tensor parameters, to change that state.
    pub fn map<F: FnOnce(T) -> T>(self, func: F) -> Self {
        let (id, tensor, param_mapper) = self.consume();
        let tensor = func(tensor);
        Self::from_mapped_value(id, tensor, param_mapper)
    }

    /// Create an initialized parameter with the given id, value, and param mapper.
    ///
    /// This is a helper method for creating parameters while preserving the param mapper,
    /// typically used in ModuleMapper implementations.
    pub fn from_mapped_value(id: ParamId, value: T, mut param_mapper: ParamMapper<T>) -> Self {
        let is_active = param_mapper
            .mapped_is_active
            .take()
            .unwrap_or_else(|| value.is_active());
        Self {
            id,
            state: LazyInitState::initialized(value),
            param_mapper,
            is_active,
            reparameterization: None,
        }
    }
}

impl<T: Parameter> Param<T> {
    /// Create a new parameter that is not already initialized.
    ///
    /// The initializer receives the device and effective gradient requirement, which is false
    /// on devices without autodiff. The requested `is_require_grad` setting is preserved for
    /// [`Module::train`](crate::module::Module::train). The initializer must create the value on
    /// the device it receives, which is not `device` when the parameter moves before initializing.
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
                sharding: None,
                init_mapped: false,
            }),
            param_mapper: Default::default(),
            is_active: is_require_grad,
            reparameterization: None,
        }
    }

    pub(crate) fn reparameterization_dyn(&self) -> Option<&dyn DynReparameterization> {
        self.reparameterization.as_deref()
    }

    /// The concrete [reparameterization](Reparameterization) attached to this parameter, if any.
    pub fn reparameterization<R: Reparameterization>(&self) -> Option<&R> {
        self.reparameterization_dyn()?.as_any().downcast_ref()
    }

    /// The LoRA [adapter](LoraAdapter) attached to this parameter, if any.
    pub fn adapter(&self) -> Option<&LoraAdapter> {
        self.reparameterization()
    }

    /// Returns a cheap clone of this parameter with its reparameterization detached.
    ///
    /// The clone shares the same lazy-initialization state, so the raw base value is not
    /// duplicated. Used to route the optimizer/record traversal over the structural base.
    pub(crate) fn without_reparameterization(&self) -> Self {
        Self {
            id: self.id,
            state: self.state.clone(),
            param_mapper: self.param_mapper.clone(),
            is_active: self.is_active,
            reparameterization: None,
        }
    }

    pub(crate) fn with_dyn_reparameterization(
        mut self,
        reparameterization: Option<Box<dyn DynReparameterization>>,
    ) -> Self {
        self.reparameterization = reparameterization;
        self
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

        let mut init = initialization.write();

        match init.as_mut() {
            Some(value) => {
                let device = value.device.clone();
                let is_require_grad = value.is_require_grad;
                let shape = value.shape.clone();
                let sharding = value.sharding;
                core::mem::drop(init);

                let mut base = self;
                Self {
                    id: base.id,
                    param_mapper: base.param_mapper.clone(),
                    is_active: base.is_active,
                    reparameterization: None,
                    state: LazyInitState::uninitialized(Uninitialized {
                        // A clone sharing `base` still initializes it where it was built. `func`
                        // maps an untracked value: mapped from a tracked leaf, it would be a
                        // non-leaf that can't require grad.
                        init: Box::new(move |request: InitRequest<'_>| {
                            let device = request.device;
                            // Move the base and attached state together before materializing.
                            // Detaching here prevents the map_to_device fallback from dropping it.
                            let reparameterization = base.reparameterization.take();
                            let base = base
                                .placed_for_init(device, request.sharding)
                                .with_dyn_reparameterization(
                                    reparameterization.map(|state| state.to_device_dyn(device)),
                                );
                            let value = base.val().set_require_grad(false);
                            func(value).set_require_grad(request.require_grad)
                        }),
                        device,
                        is_require_grad,
                        shape,
                        sharding,
                        init_mapped: true,
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

        let init = initialization.read();

        match init.as_ref() {
            Some(value) => value.device.clone(),
            None => self.device(),
        }
    }

    /// The parameter split over its sharded compute device as `sharding` says, of the shape
    /// [`lazy_shape`](Self::lazy_shape) gives.
    ///
    /// One not initialized yet, and not shared with a clone, stays so, and is split as it
    /// initializes or loads, before any init or load mapper changes its layout: a
    /// [record](crate::module::Module::load_record) applied to it then uploads each member only
    /// its chunk of every weight. Moved before then to a device the sharding does not fit, it is
    /// left replicated. Any other parameter, a lazy one shared with a clone included, is
    /// initialized and its value resharded at once, as a new leaf, and a later load splits the
    /// loaded value after its load mapper. A quantized value the split would cut through a
    /// quantization block or packed word stays replicated.
    ///
    /// # Panics
    ///
    /// When the parameter is not on a sharded compute device, when
    /// [`Device::check_sharding`] refuses the sharding for its shape, or when it is not
    /// initialized yet, shares its state with a clone and has an init mapper: initializing it
    /// would pass the mapper, so split it before cloning it.
    pub fn with_sharding(mut self, sharding: Sharding) -> Self {
        let deferred = self.update_uninitialized(|uninitialized| {
            uninitialized
                .device
                .check_sharding(&uninitialized.shape, sharding)
                .map(|()| uninitialized.sharding = Some(sharding))
        });
        match deferred {
            Some(Ok(())) => return self,
            Some(Err(error)) if cfg!(feature = "shard") => panic!("{error}"),
            Some(Err(error)) => panic!("{error} (the `shard` feature is not enabled)"),
            None => {}
        }
        if self.lazy_init_mapped() {
            panic!("{SHARED_INIT_MAPPED_REFUSAL}");
        }
        let reparameterization = self.reparameterization.take();
        self.map(|value| {
            let require_grad = value.is_require_grad();
            value.into_sharded_leaf(sharding, require_grad)
        })
        .with_dyn_reparameterization(reparameterization)
    }

    /// How the parameter is or will be split over its sharded compute device, **without
    /// triggering initialization**.
    ///
    /// Of the shape [`lazy_shape`](Self::lazy_shape) gives, and `None` when it is not on one.
    pub fn lazy_sharding(&self) -> Option<Sharding> {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => return self.sharding(),
        };

        let init = initialization.read();

        match init.as_ref() {
            Some(value) => value.sharding.or_else(|| {
                value
                    .device
                    .check_sharding(&value.shape, Sharding::Replicated)
                    .ok()
                    .map(|()| Sharding::Replicated)
            }),
            None => self.sharding(),
        }
    }

    /// Whether the parameter is not initialized yet and an init mapper wraps its initializer.
    fn lazy_init_mapped(&self) -> bool {
        self.state
            .initialization
            .as_ref()
            .is_some_and(|initialization| {
                initialization
                    .read()
                    .as_ref()
                    .is_some_and(|uninitialized| uninitialized.init_mapped)
            })
    }

    /// Where a value loaded into this parameter takes its split, if it takes one.
    fn load_split(&self) -> Option<LoadSplit> {
        let declared = self
            .state
            .initialization
            .as_ref()
            .and_then(|initialization| initialization.read().as_ref().map(|value| value.sharding));
        // A loaded value is replicated already, and never a partial sum.
        let split = |sharding| matches!(sharding, Sharding::Split { .. }).then_some(sharding);
        match declared {
            Some(sharding) => sharding.and_then(split).map(LoadSplit::BeforeMapper),
            None => self.sharding().and_then(split).map(LoadSplit::AfterMapper),
        }
    }

    /// Put the parameter on `device`: one not initialized yet initializes there, unless a clone
    /// shares it, and any other is initialized if needed and moved with `move_value`. Either way it
    /// keeps the autodiff context it was built with, as moving a value does.
    pub(crate) fn map_to_device(
        mut self,
        device: &Device,
        move_value: impl FnOnce(T) -> T,
    ) -> Self {
        let retargeted = self.update_uninitialized(|uninitialized| {
            uninitialized.device = device
                .clone()
                .with_autodiff_context_from(&uninitialized.device);
            uninitialized.sharding = uninitialized.sharding.filter(|sharding| {
                device
                    .check_sharding(&uninitialized.shape, *sharding)
                    .is_ok()
            });
        });

        match retargeted {
            Some(()) => self,
            None => self.map(move_value),
        }
    }

    /// The parameter on `device`, split as `sharding` says in its layout before its own init
    /// mapper, as the base an init mapper wrapping it initializes.
    fn placed_for_init(mut self, device: &Device, sharding: Option<Sharding>) -> Self {
        let deferred = self.update_uninitialized(|uninitialized| {
            uninitialized.device = device
                .clone()
                .with_autodiff_context_from(&uninitialized.device);
            uninitialized.sharding = sharding;
        });
        if deferred.is_some() {
            return self;
        }
        if sharding.is_some() && self.lazy_init_mapped() {
            panic!("{SHARED_INIT_MAPPED_REFUSAL}");
        }
        let placed = self.map_to_device(device, |value| value.load_to_device(device));
        match sharding {
            Some(sharding) => placed.with_sharding(sharding),
            None => placed,
        }
    }

    /// Runs `update` on the parameter's deferred initialization, or returns `None` when it is
    /// initialized or shares its state with a clone, which `update` would then change too.
    fn update_uninitialized<R>(
        &mut self,
        update: impl FnOnce(&mut Uninitialized<T>) -> R,
    ) -> Option<R> {
        let initialization = Arc::get_mut(&mut self.state)?.initialization.as_ref()?;
        let mut uninitialized = initialization.write();
        uninitialized.as_mut().map(update)
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
            None => return self.is_active,
        };

        let init = initialization.read();

        match init.as_ref() {
            Some(value) => value.is_require_grad,
            None => self.is_active,
        }
    }

    /// Override the gradient-tracking setting for the current parameter.
    ///
    /// On a backend without autodiff the effective tensor remains detached, but the setting is
    /// preserved and takes effect through [`Module::train`](crate::module::Module::train).
    pub fn set_require_grad(mut self, require_grad: bool) -> Self {
        let initialization = match &self.state.initialization {
            Some(init) => init,
            None => {
                let mut param = self.map(|tensor| tensor.set_require_grad(require_grad));
                param.is_active = require_grad;
                return param;
            }
        };

        let mut init = initialization.write();
        let mut is_lazy = false;

        if let Some(value) = init.as_mut() {
            is_lazy = true;
            value.is_require_grad = require_grad;
        };

        core::mem::drop(init);

        if is_lazy {
            self.is_active = require_grad;
            return self;
        }

        let mut param = self.map(|tensor| tensor.set_require_grad(require_grad));
        param.is_active = require_grad;
        param
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

        let init = initialization.read();

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
        let split = self.load_split();

        // Make sure we load the tensor into the same module device.
        new_tensor = new_tensor.load_to_device(&expected_device);

        if let Some(LoadSplit::BeforeMapper(sharding)) = split {
            new_tensor = new_tensor.into_sharded_leaf(sharding, false);
        }

        new_tensor = mapper.on_load(new_tensor);

        if let Some(LoadSplit::AfterMapper(sharding)) = split {
            new_tensor = new_tensor.into_sharded_leaf(sharding, false);
        }

        // Make sure we load the tensor with the same autodiff setting.
        new_tensor = new_tensor.set_require_grad(expected_require_grad);

        let mut loaded = Self::initialized(param_id, new_tensor);
        loaded.param_mapper = mapper;
        loaded.is_active = expected_require_grad;
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

        let mut transformed = Self::initialized(self.id, tensor);
        transformed.is_active = self.is_active;
        transformed
    }
}

impl<T: ParameterValue> Clone for Param<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            state: self.state.clone(),
            param_mapper: self.param_mapper.clone(),
            is_active: self.is_active,
            reparameterization: self.reparameterization.clone(),
        }
    }
}

impl<T: ParameterValue> Deref for Param<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.state.val()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "shard")]
    use crate::{module::Module, test_sharded_device};
    use burn_tensor::Tensor;

    // Param<T> should be Sync so that models can be shared across threads
    // (e.g. parallel inference with rayon).
    fn _assert_sync<T: Sync>() {}
    fn _assert_parameter_value<T: ParameterValue>() {}
    fn _assert_parameter<T: Parameter>() {}

    #[test]
    fn param_is_sync() {
        fn check() {
            _assert_sync::<Param<Tensor<2>>>();
        }
        check();
    }

    #[test]
    fn parameter_capabilities_are_split_by_value_kind() {
        _assert_parameter_value::<crate::module::Flag>();
        _assert_parameter_value::<Tensor<1>>();
        _assert_parameter::<Tensor<1>>();
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
        assert_eq!(param2.is_active, false);
        assert_eq!(param1.is_active, true); // param1 is unaffected

        // Values are still identical (same tensor data, different grad setting).
        param1
            .val()
            .into_data()
            .assert_eq(&param2.val().into_data(), true);
    }

    fn lazy_ones(device: Device, require_grad: bool) -> Param<Tensor<2>> {
        Param::uninitialized(
            ParamId::new(),
            |device, require_grad| Tensor::ones([2, 4], device).set_require_grad(require_grad),
            device,
            require_grad,
            [2, 4].into(),
        )
    }

    #[cfg(all(feature = "shard", feature = "autodiff"))]
    #[test]
    fn a_lazy_param_given_a_sharding_initializes_split_as_a_leaf() {
        let param = lazy_ones(test_sharded_device().autodiff(), true)
            .with_sharding(Sharding::Split { dim: 1 });
        assert!(!param.is_initialized());
        assert_eq!(param.lazy_sharding(), Some(Sharding::Split { dim: 1 }));

        let value = param.val();

        assert_eq!(value.sharding(), Some(Sharding::Split { dim: 1 }));
        assert!(value.is_require_grad());
        value.into_data().assert_eq(
            &Tensor::<2>::ones([2, 4], &Device::flex()).into_data(),
            true,
        );
    }

    #[cfg(feature = "shard")]
    #[test]
    fn a_lazy_param_is_split_before_its_init_mapper() {
        let param = lazy_ones(test_sharded_device(), false)
            .init_mapper(|value: Tensor<2>| value.transpose())
            .with_sharding(Sharding::Split { dim: 1 });

        let value = param.val();

        assert_eq!(value.dims(), [4, 2]);
        assert_eq!(value.sharding(), Some(Sharding::Split { dim: 0 }));
    }

    #[cfg(feature = "shard")]
    #[test]
    #[should_panic(expected = "split before it is cloned")]
    fn a_lazy_param_with_an_init_mapper_shared_with_a_clone_is_refused() {
        let param = lazy_ones(test_sharded_device(), false)
            .init_mapper(|value: Tensor<2>| value.transpose());
        let _clone = param.clone();

        let _ = param.with_sharding(Sharding::Split { dim: 0 });
    }

    #[cfg(feature = "shard")]
    #[test]
    #[should_panic(expected = "split before it is cloned")]
    fn a_shared_init_mapped_base_under_another_init_mapper_is_refused() {
        let base = lazy_ones(test_sharded_device(), false)
            .init_mapper(|value: Tensor<2>| value.transpose());
        let _clone = base.clone();
        let param = base
            .init_mapper(|value: Tensor<2>| value)
            .with_sharding(Sharding::Split { dim: 1 });

        let _ = param.val();
    }

    #[cfg(feature = "shard")]
    #[test]
    fn a_lazy_param_on_a_sharded_compute_device_is_replicated_until_split() {
        let param = lazy_ones(test_sharded_device(), false);

        assert_eq!(param.lazy_sharding(), Some(Sharding::Replicated));
        assert!(!param.is_initialized());
    }

    #[cfg(not(feature = "shard"))]
    #[test]
    #[should_panic(expected = "the `shard` feature is not enabled")]
    fn splitting_a_param_without_the_shard_feature_says_so() {
        let _ = lazy_ones(Device::default(), false).with_sharding(Sharding::Split { dim: 0 });
    }

    #[cfg(all(feature = "shard", feature = "autodiff"))]
    #[test]
    fn an_initialized_tracked_param_is_resharded_as_a_leaf() {
        let device = test_sharded_device().autodiff();
        let param = Param::from_tensor(Tensor::<2>::ones([2, 4], &device))
            .with_sharding(Sharding::Split { dim: 1 });

        let value = param.val();

        assert_eq!(value.sharding(), Some(Sharding::Split { dim: 1 }));
        assert!(value.is_require_grad());
    }

    #[cfg(feature = "shard")]
    #[test]
    fn a_lazy_param_moved_off_its_sharded_compute_device_drops_its_sharding() {
        let param =
            lazy_ones(test_sharded_device(), false).with_sharding(Sharding::Split { dim: 1 });

        let moved = param.to_device(&Device::flex());

        assert_eq!(moved.lazy_sharding(), None);
        assert_eq!(moved.val().sharding(), None);
    }

    #[cfg(feature = "shard")]
    #[test]
    #[should_panic(expected = "no dim 2")]
    fn a_sharding_on_a_dim_the_param_lacks_is_refused_before_it_initializes() {
        let _ = lazy_ones(test_sharded_device(), false).with_sharding(Sharding::Split { dim: 2 });
    }
}
