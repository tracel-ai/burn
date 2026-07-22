use crate::CubeRuntime;
use burn_cubecl_fusion::optim::reduce::{ReduceFuser, ReduceSettings};
use burn_cubecl_fusion::optim::reduce_broadcasted::ReduceBroadcastedFuser;
use burn_cubecl_fusion::optim::{CubeOptimization, elemwise::ElementWiseFuser, matmul::MatmulFuser};
use burn_fusion::OperationFuser;
use core::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// The fuser type of the cubecl fusion runtime.
type CubeFuser<R> = Box<dyn OperationFuser<Box<dyn CubeOptimization<R>>>>;

/// A user-provided fusion optimization: builds one
/// [`OperationFuser`] per execution stream, competing with the built-in
/// fusers. The fuser's [`finish`](OperationFuser::finish) boxes an
/// implementation of [`CubeOptimization`].
///
/// Register a provider with [`register`] **at the start of the program**,
/// before the first tensor operation on the fusion backend.
pub trait OptimizationProvider<R: CubeRuntime>: Send + Sync + 'static {
    /// Name identifying the optimization — the handle [`remove`] takes.
    fn name(&self) -> &str;

    /// Build a fuser for a new execution stream on `device`.
    fn fuser(&self, device: &R::Device) -> CubeFuser<R>;
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
/// is already registered.
pub fn register<R: CubeRuntime>(
    provider: impl OptimizationProvider<R>,
) -> Result<(), RegistryError> {
    let name = provider.name().to_string();
    let slot: Box<dyn Any + Send + Sync> = Box::new(ProviderSlot::<R>(Box::new(provider)));
    registry()
        .lock()
        .unwrap()
        .register(TypeId::of::<R>(), runtime_name::<R>(), name, slot)
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
        .remove(TypeId::of::<R>(), runtime_name::<R>(), name)
}

/// Names of the built-in fusion optimizations, in the order streams try them —
/// the values [`remove`] accepts besides registered provider names.
pub const BUILTIN_NAMES: [&str; 4] = ["element-wise", "matmul", "reduce", "reduce-broadcasted"];

/// The fusers for a new execution stream: the built-ins, minus the
/// [`remove`]d names, plus one fuser per registered provider. Seals the
/// registry for `R` — streams only exist once the fusion service runs, and
/// later registrations could not apply to the streams already built.
pub(crate) fn fusers<R: CubeRuntime>(device: &R::Device) -> Vec<CubeFuser<R>> {
    let (removed, mut user) = {
        let mut registry = registry().lock().unwrap();
        let entry = registry.start(TypeId::of::<R>());
        let user = entry
            .providers
            .iter()
            .map(|(_, slot)| {
                slot.downcast_ref::<ProviderSlot<R>>()
                    .expect("registry entries are keyed by runtime type")
                    .0
                    .fuser(device)
            })
            .collect::<Vec<_>>();
        (entry.removed.clone(), user)
    };

    let [elemwise, matmul, reduce, broadcasted] = BUILTIN_NAMES;
    let builtins: [(&str, CubeFuser<R>); 4] = [
        (elemwise, Box::new(ElementWiseFuser::new(device.clone()))),
        (matmul, Box::new(MatmulFuser::new(device.clone()))),
        (
            reduce,
            Box::new(ReduceFuser::new(device.clone(), ReduceSettings::Always)),
        ),
        (
            broadcasted,
            Box::new(ReduceBroadcastedFuser::new(device.clone())),
        ),
    ];

    let mut fusers: Vec<_> = builtins
        .into_iter()
        .filter(|(name, _)| !removed.iter().any(|removed| removed == name))
        .map(|(_, fuser)| fuser)
        .collect();
    fusers.append(&mut user);
    fusers
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
struct ProviderSlot<R: CubeRuntime>(Box<dyn OptimizationProvider<R>>);

/// The non-generic registry core: per-runtime provider lists, removed names,
/// and the started flag, keyed by the runtime's `TypeId`.
#[derive(Default)]
struct Registry {
    entries: HashMap<TypeId, Entry>,
}

#[derive(Default)]
struct Entry {
    providers: Vec<(String, Box<dyn Any + Send + Sync>)>,
    removed: Vec<String>,
    started: bool,
}

impl Registry {
    fn register(
        &mut self,
        runtime: TypeId,
        runtime_name: &'static str,
        name: String,
        slot: Box<dyn Any + Send + Sync>,
    ) -> Result<(), RegistryError> {
        let entry = self.entries.entry(runtime).or_default();
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
    ) -> Result<(), RegistryError> {
        let entry = self.entries.entry(runtime).or_default();
        if entry.started {
            return Err(RegistryError::ServiceRunning {
                runtime: runtime_name,
            });
        }
        entry.providers.retain(|(other, _)| other != name);
        if !entry.removed.iter().any(|other| other == name) {
            entry.removed.push(name.to_string());
        }
        Ok(())
    }

    fn start(&mut self, runtime: TypeId) -> &Entry {
        let entry = self.entries.entry(runtime).or_default();
        entry.started = true;
        entry
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct RuntimeA;
    struct RuntimeB;

    fn slot() -> Box<dyn Any + Send + Sync> {
        Box::new(())
    }

    #[test]
    fn register_then_remove_round_trips() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        registry
            .register(id, "A", "custom".into(), slot())
            .expect("first registration succeeds");
        registry.remove(id, "A", "custom").expect("removal succeeds");

        // The provider is gone, so the same name registers again.
        registry
            .register(id, "A", "custom".into(), slot())
            .expect("re-registration after removal succeeds");
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        registry.register(id, "A", "custom".into(), slot()).unwrap();
        assert_eq!(
            registry.register(id, "A", "custom".into(), slot()),
            Err(RegistryError::DuplicateOptimization {
                name: "custom".into()
            })
        );
    }

    #[test]
    fn started_service_seals_the_registry() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();
        registry.start(id);

        assert_eq!(
            registry.register(id, "A", "custom".into(), slot()),
            Err(RegistryError::ServiceRunning { runtime: "A" })
        );
        assert_eq!(
            registry.remove(id, "A", "matmul"),
            Err(RegistryError::ServiceRunning { runtime: "A" })
        );
    }

    #[test]
    fn runtimes_are_independent() {
        let mut registry = Registry::default();
        registry.start(TypeId::of::<RuntimeA>());

        // Runtime B is unaffected by A's running service.
        registry
            .register(TypeId::of::<RuntimeB>(), "B", "custom".into(), slot())
            .expect("other runtime still accepts registrations");
    }

    #[test]
    fn removed_names_accumulate_without_duplicates() {
        let mut registry = Registry::default();
        let id = TypeId::of::<RuntimeA>();

        registry.remove(id, "A", "matmul").unwrap();
        registry.remove(id, "A", "matmul").unwrap();

        assert_eq!(registry.entries[&id].removed, vec!["matmul".to_string()]);
    }
}
