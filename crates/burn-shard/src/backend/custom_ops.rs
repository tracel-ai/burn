use std::{
    any::{Any, type_name},
    collections::HashMap,
    fmt,
    sync::Arc,
};

use burn_ir::{BackendIr, CustomOpIr};
use burn_router::CustomOpRegistry;

use super::{CollectiveOpRegistry, ShardError};
use crate::{OpKind, Sharding};

/// The custom operations a sharded compute device runs: the handlers its members run, the
/// handlers that run over every member at once, and how each operation splits.
#[derive(Clone, Default)]
pub struct ShardCustomOps {
    handlers: Option<ErasedRegistry>,
    collectives: Option<ErasedRegistry>,
    rules: CustomOpRuleMap,
}

/// How each custom operation runs on a sharded compute device, by its id.
pub type CustomOpRuleMap = HashMap<String, Arc<dyn CustomOpRule>>;

impl ShardCustomOps {
    /// Every member runs `registry`'s handlers, which are the members' backend's.
    pub fn with_handlers<B: BackendIr>(mut self, registry: CustomOpRegistry<B>) -> Self {
        self.handlers = Some(ErasedRegistry::new::<B, _>(registry));
        self
    }

    /// The handlers a [`CustomOpPlan::Collective`] runs, which are the members' backend's.
    pub fn with_collectives<B: BackendIr>(mut self, registry: CollectiveOpRegistry<B>) -> Self {
        self.collectives = Some(ErasedRegistry::new::<B, _>(registry));
        self
    }

    /// The custom operation `id` splits as `rule` says. Without a rule, it gathers its inputs and
    /// runs whole on every member.
    pub fn with_rule(mut self, id: &str, rule: impl CustomOpRule + 'static) -> Self {
        self.rules.insert(id.into(), Arc::new(rule));
        self
    }
}

impl<B: BackendIr> From<CustomOpRegistry<B>> for ShardCustomOps {
    fn from(registry: CustomOpRegistry<B>) -> Self {
        Self::default().with_handlers(registry)
    }
}

impl fmt::Debug for ShardCustomOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let backend =
            |erased: &Option<ErasedRegistry>| erased.as_ref().map(|erased| erased.backend);
        f.debug_struct("ShardCustomOps")
            .field("handlers", &backend(&self.handlers))
            .field("collectives", &backend(&self.collectives))
            .field("rules", &self.rules.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// The custom operations of members that run `B`.
pub struct MemberCustomOps<B: BackendIr> {
    /// What each member runs for an operation of its own.
    pub handlers: CustomOpRegistry<B>,
    /// What runs once over every member for a collective operation.
    pub collectives: CollectiveOpRegistry<B>,
    /// How each operation splits.
    pub rules: CustomOpRuleMap,
}

/// Refused with [`ShardError::CustomOpsBackend`] when the handlers are another backend's.
impl<B: BackendIr> TryFrom<ShardCustomOps> for MemberCustomOps<B> {
    type Error = ShardError;

    fn try_from(custom_ops: ShardCustomOps) -> Result<Self, ShardError> {
        Ok(Self {
            handlers: ErasedRegistry::get::<B, _>(custom_ops.handlers.as_ref())?,
            collectives: ErasedRegistry::get::<B, _>(custom_ops.collectives.as_ref())?,
            rules: custom_ops.rules,
        })
    }
}

/// How one custom operation splits over a sharded compute device.
///
/// A rule reads the operation's IR alone, never where its inputs sit: the device then plans the
/// operation as it plans a built-in one.
///
/// An operation that draws randomness takes its `RandomKey` as an argument, reserved by the
/// caller: a handler that reserves its own draws different values on each member.
pub trait CustomOpRule: Send + Sync {
    /// How the members run `op`.
    fn plan<'a>(&self, op: &'a CustomOpIr) -> CustomOpPlan<'a>;
}

/// How the members run one custom operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CustomOpPlan<'a> {
    /// Every input whole, and the operation run whole on every member.
    Gathered,
    /// Split as a built-in operation of this kind is, each member running the operation on its
    /// own shards. An operation with several outputs, or of a random or slice kind, gathers
    /// instead.
    Kind(OpKind<'a>),
    /// Run once over every member by its [collective handler](CollectiveOpRegistry): each input
    /// reaches it sharded as `inputs` says, in input order, and each output comes out as
    /// `outputs` says. When a tensor cannot take its sharding, the outputs fail on every member;
    /// another number of shardings than the operation has tensors panics.
    Collective {
        inputs: Vec<Sharding>,
        outputs: Vec<Sharding>,
    },
}

/// A registry of one backend, its type erased and its backend kept by name.
#[derive(Clone)]
struct ErasedRegistry {
    backend: &'static str,
    registry: Arc<dyn Any + Send + Sync>,
}

impl ErasedRegistry {
    fn new<B: BackendIr, R: Any + Send + Sync>(registry: R) -> Self {
        Self {
            backend: type_name::<B>(),
            registry: Arc::new(registry),
        }
    }

    /// The registry of `B` in `erased`, or an empty one when there is none.
    fn get<B: BackendIr, R: Any + Clone + Default>(erased: Option<&Self>) -> Result<R, ShardError> {
        let Some(erased) = erased else {
            return Ok(R::default());
        };
        erased
            .registry
            .downcast_ref::<R>()
            .cloned()
            .ok_or_else(|| ShardError::CustomOpsBackend {
                registry: erased.backend.into(),
                members: type_name::<B>().into(),
            })
    }
}
