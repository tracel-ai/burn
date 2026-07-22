use crate::{FusionRuntime, OperationFuser};
use core::any::{Any, TypeId};
use hashbrown::HashMap;
use std::sync::{Mutex, OnceLock};

/// A user-provided fusion optimization: builds one [`OperationFuser`] per
/// execution stream, competing with the runtime's built-in fusers.
///
/// Register a provider with [`register`] **at the start of the program**,
/// before the first operation runs on the fusion backend; every stream created
/// afterwards asks the provider for a fresh fuser. The fuser's
/// [`finish`](OperationFuser::finish) must produce the runtime's optimization
/// type — cubecl backends expose a `Custom` variant for exactly this purpose.
pub trait OptimizationProvider<R: FusionRuntime>: Send + Sync + 'static {
    /// Name identifying the optimization — the handle [`remove`] takes.
    fn name(&self) -> &str;

    /// Build a fuser for a new execution stream on `device`.
    fn fuser(&self, device: &R::FusionDevice) -> Box<dyn OperationFuser<R::Optimization>>;
}

/// Error returned by [`register`] and [`remove`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// The fusion backend service is already running for this runtime, so the
    /// change could not be applied consistently across streams.
    ServiceRunning {
        /// Type name of the [`FusionRuntime`] whose service is running.
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
/// fuser alongside the runtime's built-in ones; the fusion search treats them
/// all equally, picking the best-scoring optimization per segment.
///
/// # Errors
///
/// Fails with [`RegistryError::ServiceRunning`] once the fusion backend
/// service for `R` has started — call this at the start of the program, before
/// the first tensor operation on the fusion backend — and with
/// [`RegistryError::DuplicateOptimization`] when a provider with the same name
/// is already registered.
pub fn register<R: FusionRuntime>(
    provider: impl OptimizationProvider<R>,
) -> Result<(), RegistryError> {
    let name = provider.name().to_string();
    let slot: Box<dyn Any + Send + Sync> = Box::new(ProviderSlot::<R>(Box::new(provider)));
    registry()
        .lock()
        .unwrap()
        .register(TypeId::of::<R>(), runtime_name::<R>(), name, slot)
}

/// Remove the fusion optimization named `name` for the runtime `R` — a
/// built-in one (masked from every stream created afterwards) or a previously
/// [`register`]ed provider. Built-in names are defined by the runtime; the
/// cubecl backends use `"element-wise"`, `"matmul"`, `"reduce"` and
/// `"reduce-broadcasted"`.
///
/// Removing a name that matches nothing is not an error: the built-in list is
/// only known once a device exists, so it cannot be validated here.
///
/// # Errors
///
/// Fails with [`RegistryError::ServiceRunning`] once the fusion backend
/// service for `R` has started; call this at the start of the program.
pub fn remove<R: FusionRuntime>(name: &str) -> Result<(), RegistryError> {
    registry()
        .lock()
        .unwrap()
        .remove(TypeId::of::<R>(), runtime_name::<R>(), name)
}

/// The fusers for a new execution stream: the runtime's built-in list, minus
/// the [`remove`]d names, plus one fuser per registered provider.
pub(crate) fn fusers<R: FusionRuntime>(
    device: R::FusionDevice,
) -> Vec<Box<dyn OperationFuser<R::Optimization>>> {
    let (removed, mut user) = {
        let registry = registry().lock().unwrap();
        let Some(entry) = registry.entries.get(&TypeId::of::<R>()) else {
            return R::fusers(device);
        };
        let user = entry
            .providers
            .iter()
            .map(|(_, slot)| {
                slot.downcast_ref::<ProviderSlot<R>>()
                    .expect("registry entries are keyed by runtime type")
                    .0
                    .fuser(&device)
            })
            .collect::<Vec<_>>();
        (entry.removed.clone(), user)
    };

    let mut fusers: Vec<_> = R::fusers(device)
        .into_iter()
        .filter(|fuser| !removed.iter().any(|name| name == fuser.name()))
        .collect();
    fusers.append(&mut user);
    fusers
}

/// Seal the registry for the runtime `R`: called when its fusion backend
/// service starts, after which [`register`] and [`remove`] fail.
pub(crate) fn mark_started<R: FusionRuntime>() {
    registry().lock().unwrap().mark_started(TypeId::of::<R>());
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(Default::default)
}

fn runtime_name<R: FusionRuntime>() -> &'static str {
    core::any::type_name::<R>()
}

/// Wraps a provider so it can live in the type-erased registry; recovered by
/// downcasting on the runtime's own `TypeId` key.
struct ProviderSlot<R: FusionRuntime>(Box<dyn OptimizationProvider<R>>);

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

    fn mark_started(&mut self, runtime: TypeId) {
        self.entries.entry(runtime).or_default().started = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{Context, OrderedExecution};
    use crate::{FuserProperties, FuserStatus, NumOperations, Optimization};
    use burn_backend::{DeviceId, DeviceOps, DeviceSettings, FloatDType};

    struct RuntimeA;
    struct RuntimeB;

    /// A fake runtime, distinct per `N` so each test gets its own registry
    /// entry (the registry is process-global, keyed by runtime type).
    #[derive(Debug)]
    struct FakeRuntime<const N: usize>;

    #[derive(Clone, Default, PartialEq, Debug)]
    struct FakeDevice;

    impl burn_backend::Device for FakeDevice {
        fn from_id(_device_id: DeviceId) -> Self {
            Self
        }
        fn to_id(&self) -> DeviceId {
            DeviceId {
                type_id: 0,
                index_id: 0,
            }
        }
    }

    impl DeviceOps for FakeDevice {
        fn defaults(&self) -> DeviceSettings {
            DeviceSettings {
                float_dtype: FloatDType::F32,
                int_dtype: burn_backend::IntDType::I32,
                bool_dtype: burn_backend::BoolDType::U8,
                quantization: Default::default(),
            }
        }
    }

    #[derive(Debug)]
    struct FakeOptimization;

    impl NumOperations for FakeOptimization {
        fn len(&self) -> usize {
            0
        }
        fn name(&self) -> &'static str {
            "fake"
        }
    }

    impl<const N: usize> Optimization<FakeRuntime<N>> for FakeOptimization {
        fn execute(
            &mut self,
            _context: &mut Context<()>,
            _execution: &OrderedExecution<FakeRuntime<N>>,
        ) {
            unimplemented!("never executed by the registry tests")
        }
        fn to_state(&self) {}
        fn from_state(_device: &FakeDevice, _state: ()) -> Self {
            Self
        }
    }

    /// A fuser that never fuses; only its name matters to these tests.
    struct FakeFuser(&'static str);

    impl<O> OperationFuser<O> for FakeFuser {
        fn name(&self) -> &'static str {
            self.0
        }
        fn fuse(&mut self, _operation: &burn_ir::OperationIr) {}
        fn finish(&mut self) -> O {
            unimplemented!("never fuses")
        }
        fn reset(&mut self) {}
        fn status(&self) -> FuserStatus {
            FuserStatus::Closed
        }
        fn properties(&self) -> FuserProperties {
            FuserProperties::default()
        }
        fn len(&self) -> usize {
            0
        }
        fn clone_dyn(&self) -> Box<dyn OperationFuser<O>> {
            Box::new(Self(self.0))
        }
    }

    impl<const N: usize> FusionRuntime for FakeRuntime<N> {
        type OptimizationState = ();
        type Optimization = FakeOptimization;
        type FusionHandle = ();
        type FusionDevice = FakeDevice;

        fn fusers(_device: FakeDevice) -> Vec<Box<dyn OperationFuser<FakeOptimization>>> {
            vec![
                Box::new(FakeFuser("builtin-a")),
                Box::new(FakeFuser("builtin-b")),
            ]
        }
    }

    struct FakeProvider(&'static str);

    impl<const N: usize> OptimizationProvider<FakeRuntime<N>> for FakeProvider {
        fn name(&self) -> &str {
            self.0
        }
        fn fuser(&self, _device: &FakeDevice) -> Box<dyn OperationFuser<FakeOptimization>> {
            Box::new(FakeFuser(self.0))
        }
    }

    fn names<const N: usize>() -> Vec<&'static str> {
        fusers::<FakeRuntime<N>>(FakeDevice)
            .iter()
            .map(|fuser| fuser.name())
            .collect()
    }

    #[test]
    fn streams_see_builtins_when_nothing_is_registered() {
        assert_eq!(names::<0>(), vec!["builtin-a", "builtin-b"]);
    }

    #[test]
    fn registered_providers_extend_the_builtins() {
        register::<FakeRuntime<1>>(FakeProvider("custom")).unwrap();
        assert_eq!(names::<1>(), vec!["builtin-a", "builtin-b", "custom"]);
    }

    #[test]
    fn removed_builtins_are_masked() {
        remove::<FakeRuntime<2>>("matmul-that-does-not-exist").unwrap();
        remove::<FakeRuntime<2>>("builtin-a").unwrap();
        register::<FakeRuntime<2>>(FakeProvider("custom")).unwrap();
        assert_eq!(names::<2>(), vec!["builtin-b", "custom"]);
    }

    #[test]
    fn register_fails_once_the_service_started() {
        mark_started::<FakeRuntime<3>>();
        let error = register::<FakeRuntime<3>>(FakeProvider("custom")).unwrap_err();
        assert!(matches!(error, RegistryError::ServiceRunning { .. }));
        // The error tells the user what to do about it.
        assert!(error.to_string().contains("start of your program"));
        assert_eq!(names::<3>(), vec!["builtin-a", "builtin-b"]);
    }

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
        registry.mark_started(id);

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
        registry.mark_started(TypeId::of::<RuntimeA>());

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
