use crate::CubeDevice;
use burn_cubecl_fusion::optim::elemwise::{self, ElementWiseFuser, ElemwiseOptimization};
use burn_cubecl_fusion::optim::matmul::{self, MatmulFuser, MatmulOptimization};
use burn_cubecl_fusion::optim::nhwc_relayout::{self, NHWCRelayoutFuser, NHWCRelayoutOptimization};
use burn_cubecl_fusion::optim::reduce::{self, ReduceFuser, ReduceOptimization, ReduceSettings};
use burn_cubecl_fusion::optim::reduce_broadcasted::{
    self, ReduceBroadcastedFuser, ReduceBroadcastedOptimization,
};
use burn_cubecl_fusion::optim::{CubeOptimization, CubeOptimizationState, FusedOperation};
use burn_fusion::OperationFuser;
use core::any::Any;
use std::sync::{Mutex, OnceLock};

/// A fuser competing for the operation segments of an execution stream,
/// [finishing](OperationFuser::finish) into a [`CubeOptimization`]. Wraps any
/// [`OperationFuser`].
pub struct CubeFuser {
    fuser: Box<dyn OperationFuser<CubeOptimization>>,
}

impl CubeFuser {
    /// Wrap the fuser.
    pub fn new(fuser: impl OperationFuser<CubeOptimization> + 'static) -> Self {
        Self {
            fuser: Box::new(fuser),
        }
    }
}

/// A user-provided fusion optimization: builds one [`CubeFuser`] per
/// execution stream, competing with the built-in fusers. The fuser's
/// [`finish`](OperationFuser::finish) wraps an implementation of
/// [`FusedOperation`] — normally [`Self::Operation`], which the provided
/// methods rely on.
///
/// Register a provider with [`register`] **at the start of the program**,
/// before the first tensor operation on the fusion backend.
pub trait OptimizationProvider: Send + Sync + 'static {
    /// The fused operation the [fusers](Self::fuser) finish.
    type Operation: FusedOperation;

    /// Name identifying the optimization — the handle [`remove`] takes, and
    /// the key serialized execution plans are restored by.
    fn name(&self) -> &str {
        Self::Operation::NAME
    }

    /// Build a fuser for a new execution stream on `device`.
    fn fuser(&self, device: &CubeDevice) -> CubeFuser;

    /// Recover an optimization produced by this provider's fuser from its
    /// serialized state — the counterpart of [`FusedOperation::to_state`].
    fn restore(&self, device: &CubeDevice, state: &CubeOptimizationState) -> CubeOptimization {
        CubeOptimization::new(Self::Operation::from_state(device, state.decode()))
    }
}

/// Object-safe view of an [`OptimizationProvider`], implemented for every one
/// of them below. Private on purpose: the erasure, like the box holding it,
/// is an implementation detail of the registry.
trait DynProvider: Send + Sync {
    fn fuser(&self, device: &CubeDevice) -> CubeFuser;
    fn restore(&self, device: &CubeDevice, state: &CubeOptimizationState) -> CubeOptimization;
}

impl<P: OptimizationProvider> DynProvider for P {
    fn fuser(&self, device: &CubeDevice) -> CubeFuser {
        OptimizationProvider::fuser(self, device)
    }

    fn restore(&self, device: &CubeDevice, state: &CubeOptimizationState) -> CubeOptimization {
        OptimizationProvider::restore(self, device, state)
    }
}

/// The built-in optimizations: providers registered by default for every
/// runtime, so removal and plan restoration treat them exactly like
/// user-provided ones.
struct ElemwiseProvider;
struct MatmulProvider;
struct ReduceProvider;
struct ReduceBroadcastedProvider;
struct NHWCRelayoutProvider;

impl OptimizationProvider for ElemwiseProvider {
    type Operation = ElemwiseOptimization;

    fn fuser(&self, device: &CubeDevice) -> CubeFuser {
        CubeFuser::new(ElementWiseFuser::new(device.clone()))
    }
}

impl OptimizationProvider for MatmulProvider {
    type Operation = MatmulOptimization;

    fn fuser(&self, device: &CubeDevice) -> CubeFuser {
        CubeFuser::new(MatmulFuser::new(device.clone()))
    }
}

impl OptimizationProvider for ReduceProvider {
    type Operation = ReduceOptimization;

    fn fuser(&self, device: &CubeDevice) -> CubeFuser {
        CubeFuser::new(ReduceFuser::new(device.clone(), ReduceSettings::Always))
    }
}

impl OptimizationProvider for ReduceBroadcastedProvider {
    type Operation = ReduceBroadcastedOptimization;

    fn fuser(&self, device: &CubeDevice) -> CubeFuser {
        CubeFuser::new(ReduceBroadcastedFuser::new(device.clone()))
    }
}

impl OptimizationProvider for NHWCRelayoutProvider {
    type Operation = NHWCRelayoutOptimization;

    fn fuser(&self, device: &CubeDevice) -> CubeFuser {
        CubeFuser::new(NHWCRelayoutFuser::new(device.clone()))
    }
}

/// Names of the built-in fusion optimizations, in the order streams try them —
/// the values [`remove`] accepts besides registered provider names.
pub const BUILTIN_NAMES: [&str; 5] = [
    elemwise::NAME,
    matmul::NAME,
    reduce::NAME,
    reduce_broadcasted::NAME,
    nhwc_relayout::NAME,
];

/// The default providers seeding a runtime's registry entry.
fn builtins() -> Vec<(String, Slot)> {
    vec![
        slot(ElemwiseProvider),
        slot(MatmulProvider),
        slot(ReduceProvider),
        slot(ReduceBroadcastedProvider),
        slot(NHWCRelayoutProvider),
    ]
}

fn slot(provider: impl OptimizationProvider) -> (String, Slot) {
    let name = provider.name().to_string();
    (name, Box::new(ProviderSlot(Box::new(provider))))
}

/// Error returned by [`register`] and [`remove`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The fusion backend service is already running, so the change could not
    /// be applied consistently across streams.
    ServiceRunning,
    /// A provider with the same name is already registered.
    DuplicateOptimization {
        /// The conflicting name.
        name: String,
    },
}

impl core::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ServiceRunning => write!(
                f,
                "the fusion backend service is already running; \
                 register or remove fusion optimizations at the start of your program, \
                 before the first tensor operation on the fusion backend"
            ),
            Self::DuplicateOptimization { name } => write!(
                f,
                "a fusion optimization named `{name}` is already registered"
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Register a user-defined fusion optimization.
///
/// Every execution stream created after this call includes the provider's
/// fuser alongside the built-in ones; the fusion search treats them all
/// equally, picking the best-scoring optimization per segment.
///
/// # Errors
///
/// Fails with [`RegistryError::ServiceRunning`] once the fusion backend
/// service has started — call this at the start of the program, before the
/// first tensor operation on the fusion backend — and with
/// [`RegistryError::DuplicateOptimization`] when a provider with the same name
/// is already registered, built-ins included.
pub fn register(provider: impl OptimizationProvider) -> Result<(), RegistryError> {
    let (name, slot) = slot(provider);
    registry().lock().unwrap().register(name, slot)
}

/// Remove the fusion optimization named `name` — one of the built-ins
/// ([`BUILTIN_NAMES`]) or a previously [`register`]ed provider.
///
/// Removing a name that matches nothing is not an error, so a removal stays
/// valid when a built-in is renamed or retired.
///
/// # Errors
///
/// Fails with [`RegistryError::ServiceRunning`] once the fusion backend
/// service has started; call this at the start of the program.
pub fn remove(name: &str) -> Result<(), RegistryError> {
    registry().lock().unwrap().remove(name)
}

/// Restore the optimization described by `state` through its provider's
/// [`restore`](OptimizationProvider::restore).
pub(crate) fn restore(device: &CubeDevice, state: CubeOptimizationState) -> CubeOptimization {
    let registry = registry().lock().unwrap();
    registry
        .provider(&state.name)
        .map(|slot| downcast(slot).restore(device, &state))
        .unwrap_or_else(|| {
            panic!(
                "no fusion optimization named `{}` is registered; register its \
                 provider before restoring serialized execution plans",
                state.name,
            )
        })
}

/// The fusers for a new execution stream: one per registered provider — the
/// built-ins minus the [`remove`]d ones, plus the user-registered ones. Seals
/// the registry — streams only exist once the fusion service runs, and later
/// registrations could not apply to the streams already built.
pub(crate) fn fusers(device: &CubeDevice) -> Vec<Box<dyn OperationFuser<CubeOptimization>>> {
    let mut registry = registry().lock().unwrap();
    registry
        .start()
        .providers
        .iter()
        .map(|(_, slot)| downcast(slot).fuser(device).fuser)
        .collect()
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::seeded(builtins)))
}

/// Recover the typed provider from a stored slot.
fn downcast(slot: &Slot) -> &dyn DynProvider {
    slot.downcast_ref::<ProviderSlot>()
        .expect("every slot in the registry holds a provider")
        .0
        .as_ref()
}

/// Wraps a provider so it can live in the type-erased registry.
struct ProviderSlot(Box<dyn DynProvider>);

/// A type-erased provider stored in the non-generic registry core.
type Slot = Box<dyn Any + Send + Sync>;

/// The providers and the fusion-service flag; sealed against changes once the
/// service starts.
struct Registry {
    providers: Vec<(String, Slot)>,
    started: bool,
}

impl Registry {
    fn seeded(defaults: impl FnOnce() -> Vec<(String, Slot)>) -> Self {
        Self {
            providers: defaults(),
            started: false,
        }
    }

    fn register(&mut self, name: String, slot: Slot) -> Result<(), RegistryError> {
        self.ensure_open()?;
        if self.providers.iter().any(|(other, _)| *other == name) {
            return Err(RegistryError::DuplicateOptimization { name });
        }
        self.providers.push((name, slot));
        Ok(())
    }

    fn remove(&mut self, name: &str) -> Result<(), RegistryError> {
        self.ensure_open()?;
        self.providers.retain(|(other, _)| other != name);
        Ok(())
    }

    fn start(&mut self) -> &Self {
        self.started = true;
        self
    }

    fn provider(&self, name: &str) -> Option<&Slot> {
        self.providers
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, slot)| slot)
    }

    /// Registrations and removals only apply before the fusion service starts.
    fn ensure_open(&self) -> Result<(), RegistryError> {
        if self.started {
            return Err(RegistryError::ServiceRunning);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot() -> Slot {
        Box::new(())
    }

    fn empty() -> Registry {
        Registry::seeded(Vec::new)
    }

    fn seeded() -> Vec<(String, Slot)> {
        vec![("builtin".into(), slot())]
    }

    #[test]
    fn register_then_remove_round_trips() {
        let mut registry = empty();

        registry
            .register("custom".into(), slot())
            .expect("first registration succeeds");
        registry.remove("custom").expect("removal succeeds");

        // The provider is gone, so the same name registers again.
        registry
            .register("custom".into(), slot())
            .expect("re-registration after removal succeeds");
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = empty();

        registry.register("custom".into(), slot()).unwrap();
        assert_eq!(
            registry.register("custom".into(), slot()),
            Err(RegistryError::DuplicateOptimization {
                name: "custom".into()
            })
        );
    }

    #[test]
    fn a_started_registry_is_sealed() {
        let mut registry = empty();
        registry.start();

        assert_eq!(
            registry.register("custom".into(), slot()),
            Err(RegistryError::ServiceRunning)
        );
        assert_eq!(
            registry.remove("builtin"),
            Err(RegistryError::ServiceRunning)
        );
    }

    #[test]
    fn provider_lookup_by_name() {
        let mut registry = empty();
        registry.register("custom".into(), slot()).unwrap();

        assert!(registry.provider("custom").is_some());
        assert!(registry.provider("unknown").is_none());
    }

    #[test]
    fn defaults_seed_the_registry() {
        let registry = Registry::seeded(seeded);

        assert!(registry.provider("builtin").is_some());
    }

    #[test]
    fn defaults_are_removable_and_reserve_their_name() {
        let mut registry = Registry::seeded(seeded);

        // A seeded name is held like any other, so it cannot be registered over...
        assert_eq!(
            registry.register("builtin".into(), slot()),
            Err(RegistryError::DuplicateOptimization {
                name: "builtin".into()
            })
        );

        // ...but it is not privileged either: removing it frees the name for a replacement.
        registry.remove("builtin").unwrap();
        assert!(registry.provider("builtin").is_none());
        registry
            .register("builtin".into(), slot())
            .expect("the name is free after removal");
    }
}
