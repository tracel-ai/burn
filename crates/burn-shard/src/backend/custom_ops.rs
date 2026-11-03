use std::{
    any::{Any, type_name},
    fmt,
    sync::Arc,
};

use burn_ir::BackendIr;
use burn_router::CustomOpRegistry;

use super::ShardError;

/// The custom operation handlers every member of a sharded compute device runs, for the backend
/// the members run.
#[derive(Clone, Default)]
pub struct ShardCustomOps {
    registry: Option<AnyRegistry>,
}

impl ShardCustomOps {
    /// The handlers of `B`'s custom operations, for members that run `B`.
    ///
    /// # Errors
    ///
    /// [`ShardError::CustomOpsBackend`] when the handlers were registered for another backend.
    pub fn registry<B: BackendIr>(&self) -> Result<CustomOpRegistry<B>, ShardError> {
        let Some(any) = &self.registry else {
            return Ok(CustomOpRegistry::new());
        };
        any.registry
            .downcast_ref::<CustomOpRegistry<B>>()
            .cloned()
            .ok_or_else(|| ShardError::CustomOpsBackend {
                registry: any.backend.into(),
                members: type_name::<B>().into(),
            })
    }
}

impl<B: BackendIr> From<CustomOpRegistry<B>> for ShardCustomOps {
    fn from(registry: CustomOpRegistry<B>) -> Self {
        Self {
            registry: Some(AnyRegistry {
                backend: type_name::<B>(),
                registry: Arc::new(registry),
            }),
        }
    }
}

impl fmt::Debug for ShardCustomOps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ShardCustomOps")
            .field("backend", &self.registry.as_ref().map(|any| any.backend))
            .finish()
    }
}

/// A [`CustomOpRegistry`] of some backend, named by it.
#[derive(Clone)]
struct AnyRegistry {
    backend: &'static str,
    registry: Arc<dyn Any + Send + Sync>,
}
