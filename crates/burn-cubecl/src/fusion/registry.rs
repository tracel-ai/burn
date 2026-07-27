use crate::CubeRuntime;
use burn_cubecl_fusion::optim::elemwise::{self, ElementWiseFuser, ElemwiseOptimization};
use burn_cubecl_fusion::optim::matmul::{self, MatmulFuser, MatmulOptimization};
use burn_cubecl_fusion::optim::nhwc_relayout::{self, NHWCRelayoutFuser, NHWCRelayoutOptimization};
use burn_cubecl_fusion::optim::reduce::{self, ReduceFuser, ReduceOptimization, ReduceSettings};
use burn_cubecl_fusion::optim::reduce_broadcasted::{
    self, ReduceBroadcastedFuser, ReduceBroadcastedOptimization,
};
use burn_cubecl_fusion::optim::{CubeOptimization, CubeOptimizationState, FusedOperation};
use burn_fusion::OperationFuser;
use core::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// A fuser competing for the operation segments of an execution stream,
/// [finishing](OperationFuser::finish) into a [`CubeOptimization`]. Wraps any
/// [`OperationFuser`].
pub struct CubeFuser<R: CubeRuntime> {
    fuser: Box<dyn OperationFuser<CubeOptimization<R>>>,
}

impl<R: CubeRuntime> CubeFuser<R> {
    /// Wrap the fuser.
    pub fn new(fuser: impl OperationFuser<CubeOptimization<R>> + 'static) -> Self {
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
struct NHWCRelayoutProvider;

impl<R: CubeRuntime> OptimizationProvider<R> for ElemwiseProvider {
    type Operation = ElemwiseOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        CubeFuser::new(ElementWiseFuser::new(device.clone()))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for MatmulProvider {
    type Operation = MatmulOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        CubeFuser::new(MatmulFuser::new(device.clone()))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for ReduceProvider {
    type Operation = ReduceOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        CubeFuser::new(ReduceFuser::new(device.clone(), ReduceSettings::Always))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for ReduceBroadcastedProvider {
    type Operation = ReduceBroadcastedOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
        CubeFuser::new(ReduceBroadcastedFuser::new(device.clone()))
    }
}

impl<R: CubeRuntime> OptimizationProvider<R> for NHWCRelayoutProvider {
    type Operation = NHWCRelayoutOptimization<R>;

    fn fuser(&self, device: &R::Device) -> CubeFuser<R> {
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
fn builtins<R: CubeRuntime>() -> Vec<(String, Slot)> {
    vec![
        slot::<R>(ElemwiseProvider),
        slot::<R>(MatmulProvider),
        slot::<R>(ReduceProvider),
        slot::<R>(ReduceBroadcastedProvider),
        slot::<R>(NHWCRelayoutProvider),
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
///
/// # Warning
///
/// `R` must be the exact runtime the fusion backend executes on, generic
/// parameters included — e.g. `WgpuRuntime<SpirvCompiler>` for the Vulkan
/// backend, not the default-compiler `WgpuRuntime`. Registering for a runtime
/// that never runs is not an error; the provider just never competes.
pub fn register<R: CubeRuntime>(
    provider: impl OptimizationProvider<R>,
) -> Result<(), RegistryError> {
    let (name, slot) = slot::<R>(provider);
    let mut registry = registry().lock().unwrap();
    entry_of::<R>(&mut registry).register(runtime_name::<R>(), name, slot)
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
    let mut registry = registry().lock().unwrap();
    entry_of::<R>(&mut registry).remove(runtime_name::<R>(), name)
}

/// Restore the optimization described by `state` through its provider's
/// [`restore`](OptimizationProvider::restore).
pub(crate) fn restore<R: CubeRuntime>(
    device: &R::Device,
    state: CubeOptimizationState,
) -> CubeOptimization<R> {
    let mut registry = registry().lock().unwrap();
    entry_of::<R>(&mut registry)
        .provider(&state.name)
        .map(|slot| downcast::<R>(slot).restore(device, &state))
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
pub(crate) fn fusers<R: CubeRuntime>(
    device: &R::Device,
) -> Vec<Box<dyn OperationFuser<CubeOptimization<R>>>> {
    let mut registry = registry().lock().unwrap();
    entry_of::<R>(&mut registry)
        .start()
        .providers
        .iter()
        .map(|(_, slot)| downcast::<R>(slot).fuser(device).fuser)
        .collect()
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

fn runtime_name<R: CubeRuntime>() -> &'static str {
    core::any::type_name::<R>()
}

/// The registry entry for `R`, seeded with the built-in providers when first
/// touched.
fn entry_of<R: CubeRuntime>(registry: &mut Registry) -> &mut Entry {
    registry.entry(TypeId::of::<R>(), builtins::<R>)
}

/// Recover the typed provider from a stored slot.
fn downcast<R: CubeRuntime>(slot: &Slot) -> &dyn DynProvider<R> {
    slot.downcast_ref::<ProviderSlot<R>>()
        .expect("registry entries are keyed by runtime type")
        .0
        .as_ref()
}

/// Wraps a provider so it can live in the type-erased registry; recovered by
/// downcasting on the runtime's own `TypeId` key.
struct ProviderSlot<R: CubeRuntime>(Box<dyn DynProvider<R>>);

/// A type-erased provider stored in the non-generic registry core.
type Slot = Box<dyn Any + Send + Sync>;

/// The non-generic registry core: one [`Entry`] per runtime, keyed by the
/// runtime's `TypeId`.
#[derive(Default)]
struct Registry {
    entries: HashMap<TypeId, Entry>,
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
}

/// A runtime's providers and its fusion-service flag; sealed against changes
/// once the service starts.
struct Entry {
    providers: Vec<(String, Slot)>,
    started: bool,
}

impl Entry {
    fn register(
        &mut self,
        runtime: &'static str,
        name: String,
        slot: Slot,
    ) -> Result<(), RegistryError> {
        self.ensure_open(runtime)?;
        if self.providers.iter().any(|(other, _)| *other == name) {
            return Err(RegistryError::DuplicateOptimization { name });
        }
        self.providers.push((name, slot));
        Ok(())
    }

    fn remove(&mut self, runtime: &'static str, name: &str) -> Result<(), RegistryError> {
        self.ensure_open(runtime)?;
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
    fn ensure_open(&self, runtime: &'static str) -> Result<(), RegistryError> {
        if self.started {
            return Err(RegistryError::ServiceRunning { runtime });
        }
        Ok(())
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

    fn empty() -> Entry {
        Entry {
            providers: Vec::new(),
            started: false,
        }
    }

    fn seeded() -> Vec<(String, Slot)> {
        vec![("builtin".into(), slot())]
    }

    #[test]
    fn register_then_remove_round_trips() {
        let mut entry = empty();

        entry
            .register("A", "custom".into(), slot())
            .expect("first registration succeeds");
        entry.remove("A", "custom").expect("removal succeeds");

        // The provider is gone, so the same name registers again.
        entry
            .register("A", "custom".into(), slot())
            .expect("re-registration after removal succeeds");
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut entry = empty();

        entry.register("A", "custom".into(), slot()).unwrap();
        assert_eq!(
            entry.register("A", "custom".into(), slot()),
            Err(RegistryError::DuplicateOptimization {
                name: "custom".into()
            })
        );
    }

    #[test]
    fn started_entry_is_sealed() {
        let mut entry = empty();
        entry.start();

        assert_eq!(
            entry.register("A", "custom".into(), slot()),
            Err(RegistryError::ServiceRunning { runtime: "A" })
        );
        assert_eq!(
            entry.remove("A", "builtin"),
            Err(RegistryError::ServiceRunning { runtime: "A" })
        );
    }

    #[test]
    fn provider_lookup_by_name() {
        let mut entry = empty();
        entry.register("A", "custom".into(), slot()).unwrap();

        assert!(entry.provider("custom").is_some());
        assert!(entry.provider("unknown").is_none());
    }

    #[test]
    fn runtimes_are_independent() {
        let mut registry = Registry::default();
        registry.entry(TypeId::of::<RuntimeA>(), Vec::new).start();

        // Runtime B is unaffected by A's running service.
        registry
            .entry(TypeId::of::<RuntimeB>(), Vec::new)
            .register("B", "custom".into(), slot())
            .expect("other runtime still accepts registrations");
    }

    #[test]
    fn defaults_seed_the_entry_once() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        assert!(registry.entry(id, seeded).provider("builtin").is_some());
        // The entry already exists; later defaults are not re-applied.
        assert_eq!(registry.entry(id, seeded).providers.len(), 1);
    }

    #[test]
    fn defaults_are_removable_and_reserve_their_name() {
        let mut registry = Registry::default();
        let entry = registry.entry(TypeId::of::<RuntimeA>(), seeded);

        assert_eq!(
            entry.register("A", "builtin".into(), slot()),
            Err(RegistryError::DuplicateOptimization {
                name: "builtin".into()
            })
        );

        entry.remove("A", "builtin").unwrap();
        assert!(entry.provider("builtin").is_none());
        entry
            .register("A", "builtin".into(), slot())
            .expect("the name is free after removal");
    }
}
