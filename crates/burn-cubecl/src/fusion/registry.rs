use crate::CubeRuntime;
use burn_cubecl_fusion::optim::elemwise::{ElementWiseFuser, ElemwiseOptimization};
use burn_cubecl_fusion::optim::matmul::{MatmulFuser, MatmulOptimization};
use burn_cubecl_fusion::optim::reduce::{ReduceFuser, ReduceOptimization, ReduceSettings};
use burn_cubecl_fusion::optim::reduce_broadcasted::{
    ReduceBroadcastedFuser, ReduceBroadcastedOptimization,
};
use burn_cubecl_fusion::optim::{CubeOptimization, CubeOptimizationState, FusedOperation};
use burn_fusion::OperationFuser;
use core::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// The fuser type of the cubecl fusion runtime.
type CubeFuser<R> = Box<dyn OperationFuser<CubeOptimization<R>>>;

/// A user-provided fusion optimization: builds one
/// [`OperationFuser`] per execution stream, competing with the built-in
/// fusers. The fuser's [`finish`](OperationFuser::finish) wraps an
/// implementation of [`FusedOperation`] — normally [`Self::Operation`],
/// which the provided methods rely on.
///
/// Register a provider with [`register`] **at the start of the program**,
/// before the first tensor operation on the fusion backend.
pub trait OptimizationProvider<R: CubeRuntime>: Send + Sync + 'static {
    /// The fused operation the [fusers](Self::fuser) finish.
    type Operation: FusedOperation<R>;

    /// Name identifying the optimization — the handle [`remove`] takes, and
    /// the key serialized execution plans are restored by.
    fn name(&self) -> &str {
        Self::Operation::NAME
    }

    /// Build a fuser for a new execution stream on `device`.
    fn fuser(&self, device: &R::Device) -> CubeFuser<R>;

    /// Recover an optimization produced by this provider's fuser from its
    /// serialized state — the counterpart of [`FusedOperation::to_state`].
    fn restore(&self, device: &R::Device, state: &CubeOptimizationState) -> CubeOptimization<R> {
        CubeOptimization::new(Self::Operation::from_state(device, state.decode()))
    }
}

/// Object-safe view of an [`OptimizationProvider`], implemented for every one
/// of them below. Private on purpose: the erasure, like the box holding it,
/// is an implementation detail of the registry.
trait DynProvider<R: CubeRuntime>: Send + Sync {
    fn fuser(&self, device: &R::Device) -> CubeFuser<R>;
    fn restore(&self, device: &R::Device, state: &CubeOptimizationState) -> CubeOptimization<R>;
}

impl<R: CubeRuntime, P: OptimizationProvider<R>> DynProvider<R> for P {
    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        OptimizationProvider::fuser(self, device)
    }

    fn restore(&self, device: &R::Device, state: &CubeOptimizationState) -> CubeOptimization<R> {
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

impl<R: CubeRuntime> OptimizationProvider<R> for ElemwiseProvider {
    type Operation = ElemwiseOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        Box::new(ElementWiseFuser::new(device.clone()))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for MatmulProvider {
    type Operation = MatmulOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        Box::new(MatmulFuser::new(device.clone()))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for ReduceProvider {
    type Operation = ReduceOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        Box::new(ReduceFuser::new(device.clone(), ReduceSettings::Always))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for ReduceBroadcastedProvider {
    type Operation = ReduceBroadcastedOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        Box::new(ReduceBroadcastedFuser::new(device.clone()))
    }
}

/// Names of the built-in fusion optimizations, in the order streams try them —
/// the values [`remove`] accepts besides registered provider names. Matches
/// each built-in's [`FusedOperation::NAME`].
pub const BUILTIN_NAMES: [&str; 4] = ["ElementWise", "Matmul", "Reduce", "ReduceBroadcasted"];

/// The default providers seeding a runtime's registry entry.
fn builtins<R: CubeRuntime>() -> Vec<(String, Slot)> {
    vec![
        slot::<R>(ElemwiseProvider),
        slot::<R>(MatmulProvider),
        slot::<R>(ReduceProvider),
        slot::<R>(ReduceBroadcastedProvider),
    ]
}

fn slot<R: CubeRuntime>(provider: impl OptimizationProvider<R>) -> (String, Slot) {
    let name = provider.name().to_string();
    (name, Box::new(ProviderSlot::<R>(Box::new(provider))))
}

/// Error returned by [`register`] and [`remove`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The fusion backend service is already running for this runtime, so the
    /// change could not be applied consistently across streams.
    ServiceRunning {
        /// Type name of the runtime whose fusion service is running.
        runtime: &'static str,
    },
    /// A provider with the same name is already registered.
    DuplicateOptimization {
        /// The conflicting name.
        name: String,
    },
}

impl core::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ServiceRunning { runtime } => write!(
                f,
                "the fusion backend service for `{runtime}` is already running; \
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

/// Register a user-defined fusion optimization for the runtime `R`.
///
/// Every execution stream created after this call includes the provider's
/// fuser alongside the built-in ones; the fusion search treats them all
/// equally, picking the best-scoring optimization per segment.
///
/// # Errors
///
/// Fails with [`RegistryError::ServiceRunning`] once the fusion backend
/// service for `R` has started — call this at the start of the program, before
/// the first tensor operation on the fusion backend — and with
/// [`RegistryError::DuplicateOptimization`] when a provider with the same name
/// is already registered, built-ins included.
pub fn register<R: CubeRuntime>(
    provider: impl OptimizationProvider<R>,
) -> Result<(), RegistryError> {
    let (name, slot) = slot::<R>(provider);
    registry().lock().unwrap().register(
        TypeId::of::<R>(),
        runtime_name::<R>(),
        name,
        slot,
        builtins::<R>,
    )
}

/// Remove the fusion optimization named `name` for the runtime `R` — one of
/// the built-ins ([`BUILTIN_NAMES`]) or a previously [`register`]ed provider.
///
/// Removing a name that matches nothing is not an error, so a removal stays
/// valid when a built-in is renamed or retired.
///
/// # Errors
///
/// Fails with [`RegistryError::ServiceRunning`] once the fusion backend
/// service for `R` has started; call this at the start of the program.
pub fn remove<R: CubeRuntime>(name: &str) -> Result<(), RegistryError> {
    registry()
        .lock()
        .unwrap()
        .remove(TypeId::of::<R>(), runtime_name::<R>(), name, builtins::<R>)
}

/// Restore the optimization described by `state` through its provider's
/// [`restore`](OptimizationProvider::restore).
pub(crate) fn restore<R: CubeRuntime>(
    device: &R::Device,
    state: CubeOptimizationState,
) -> CubeOptimization<R> {
    registry()
        .lock()
        .unwrap()
        .provider(TypeId::of::<R>(), &state.name, builtins::<R>)
        .map(|slot| {
            slot.downcast_ref::<ProviderSlot<R>>()
                .expect("registry entries are keyed by runtime type")
                .0
                .restore(device, &state)
        })
        .unwrap_or_else(|| {
            panic!(
                "no fusion optimization named `{}` is registered for `{}`; register its \
                 provider before restoring serialized execution plans",
                state.name,
                runtime_name::<R>()
            )
        })
}

/// The fusers for a new execution stream: one per registered provider — the
/// built-ins minus the [`remove`]d ones, plus the user-registered ones. Seals
/// the registry for `R` — streams only exist once the fusion service runs, and
/// later registrations could not apply to the streams already built.
pub(crate) fn fusers<R: CubeRuntime>(device: &R::Device) -> Vec<CubeFuser<R>> {
    let mut registry = registry().lock().unwrap();
    registry
        .start(TypeId::of::<R>(), builtins::<R>)
        .providers
        .iter()
        .map(|(_, slot)| {
            slot.downcast_ref::<ProviderSlot<R>>()
                .expect("registry entries are keyed by runtime type")
                .0
                .fuser(device)
        })
        .collect()
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

fn runtime_name<R: CubeRuntime>() -> &'static str {
    core::any::type_name::<R>()
}

/// Wraps a provider so it can live in the type-erased registry; recovered by
/// downcasting on the runtime's own `TypeId` key.
struct ProviderSlot<R: CubeRuntime>(Box<dyn DynProvider<R>>);

/// A type-erased provider stored in the non-generic registry core.
type Slot = Box<dyn Any + Send + Sync>;

/// The non-generic registry core: per-runtime provider lists and the started
/// flag, keyed by the runtime's `TypeId`. Entries are seeded with the default
/// providers on first access.
#[derive(Default)]
struct Registry {
    entries: HashMap<TypeId, Entry>,
}

struct Entry {
    providers: Vec<(String, Slot)>,
    started: bool,
}

impl Registry {
    fn entry(
        &mut self,
        runtime: TypeId,
        defaults: impl FnOnce() -> Vec<(String, Slot)>,
    ) -> &mut Entry {
        self.entries.entry(runtime).or_insert_with(|| Entry {
            providers: defaults(),
            started: false,
        })
    }

    fn register(
        &mut self,
        runtime: TypeId,
        runtime_name: &'static str,
        name: String,
        slot: Slot,
        defaults: impl FnOnce() -> Vec<(String, Slot)>,
    ) -> Result<(), RegistryError> {
        let entry = self.entry(runtime, defaults);
        if entry.started {
            return Err(RegistryError::ServiceRunning {
                runtime: runtime_name,
            });
        }
        if entry.providers.iter().any(|(other, _)| *other == name) {
            return Err(RegistryError::DuplicateOptimization { name });
        }
        entry.providers.push((name, slot));
        Ok(())
    }

    fn remove(
        &mut self,
        runtime: TypeId,
        runtime_name: &'static str,
        name: &str,
        defaults: impl FnOnce() -> Vec<(String, Slot)>,
    ) -> Result<(), RegistryError> {
        let entry = self.entry(runtime, defaults);
        if entry.started {
            return Err(RegistryError::ServiceRunning {
                runtime: runtime_name,
            });
        }
        entry.providers.retain(|(other, _)| other != name);
        Ok(())
    }

    fn start(&mut self, runtime: TypeId, defaults: impl FnOnce() -> Vec<(String, Slot)>) -> &Entry {
        let entry = self.entry(runtime, defaults);
        entry.started = true;
        entry
    }

    fn provider(
        &mut self,
        runtime: TypeId,
        name: &str,
        defaults: impl FnOnce() -> Vec<(String, Slot)>,
    ) -> Option<&Slot> {
        self.entry(runtime, defaults)
            .providers
            .iter()
            .find(|(other, _)| other == name)
            .map(|(_, slot)| slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeA;
    struct RuntimeB;

    fn slot() -> Slot {
        Box::new(())
    }

    fn no_defaults() -> Vec<(String, Slot)> {
        Vec::new()
    }

    fn defaults() -> Vec<(String, Slot)> {
        vec![("builtin".into(), slot())]
    }

    #[test]
    fn register_then_remove_round_trips() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        registry
            .register(id, "A", "custom".into(), slot(), no_defaults)
            .expect("first registration succeeds");
        registry
            .remove(id, "A", "custom", no_defaults)
            .expect("removal succeeds");

        // The provider is gone, so the same name registers again.
        registry
            .register(id, "A", "custom".into(), slot(), no_defaults)
            .expect("re-registration after removal succeeds");
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        registry
            .register(id, "A", "custom".into(), slot(), no_defaults)
            .unwrap();
        assert_eq!(
            registry.register(id, "A", "custom".into(), slot(), no_defaults),
            Err(RegistryError::DuplicateOptimization {
                name: "custom".into()
            })
        );
    }

    #[test]
    fn started_service_seals_the_registry() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();
        registry.start(id, no_defaults);

        assert_eq!(
            registry.register(id, "A", "custom".into(), slot(), no_defaults),
            Err(RegistryError::ServiceRunning { runtime: "A" })
        );
        assert_eq!(
            registry.remove(id, "A", "builtin", no_defaults),
            Err(RegistryError::ServiceRunning { runtime: "A" })
        );
    }

    #[test]
    fn runtimes_are_independent() {
        let mut registry = Registry::default();
        registry.start(TypeId::of::<RuntimeA>(), no_defaults);

        // Runtime B is unaffected by A's running service.
        registry
            .register(
                TypeId::of::<RuntimeB>(),
                "B",
                "custom".into(),
                slot(),
                no_defaults,
            )
            .expect("other runtime still accepts registrations");
    }

    #[test]
    fn provider_lookup_by_name() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();
        registry
            .register(id, "A", "custom".into(), slot(), no_defaults)
            .unwrap();

        assert!(registry.provider(id, "custom", no_defaults).is_some());
        assert!(registry.provider(id, "unknown", no_defaults).is_none());
        assert!(
            registry
                .provider(TypeId::of::<RuntimeB>(), "custom", no_defaults)
                .is_none()
        );
    }

    #[test]
    fn defaults_seed_the_entry_once() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        assert!(registry.provider(id, "builtin", defaults).is_some());
        // The entry already exists; later defaults are not re-applied.
        assert_eq!(registry.start(id, defaults).providers.len(), 1);
    }

    #[test]
    fn defaults_are_removable_and_reserve_their_name() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        assert_eq!(
            registry.register(id, "A", "builtin".into(), slot(), defaults),
            Err(RegistryError::DuplicateOptimization {
                name: "builtin".into()
            })
        );

        registry.remove(id, "A", "builtin", defaults).unwrap();
        assert!(registry.provider(id, "builtin", defaults).is_none());
        registry
            .register(id, "A", "builtin".into(), slot(), defaults)
            .expect("the name is free after removal");
    }
}
