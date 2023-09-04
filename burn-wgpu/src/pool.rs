use crate::{context::Context, GraphicsApi, WgpuDevice};
use std::{
    any::TypeId,
    collections::HashMap,
    sync::{Arc, Mutex},
};

static POOL_CONTEXT: Mutex<Option<ContextPool>> = Mutex::new(None);

#[derive(Default)]
struct ContextPool {
    contexts: HashMap<Key, Arc<Context>>,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct Key {
    api_id: TypeId,
    device: WgpuDevice,
}

impl Key {
    fn new<G: GraphicsApi>(device: &WgpuDevice) -> Self {
        Self {
            api_id: TypeId::of::<G>(),
            device: device.clone(),
        }
    }
}

/// Get a [context](Context) for the given [device](WGPUDevice).
///
/// # Notes
///
/// If a context already exist for the current [device](WGPUDevice), the same instance will be
/// returned.
pub fn get_context<G: GraphicsApi>(device: &WgpuDevice) -> Arc<Context> {
    let mut pool = POOL_CONTEXT.lock().unwrap();

    let context = if let Some(pool) = pool.as_mut() {
        // Fetch device in pool
        match pool.contexts.get(&Key::new::<G>(device)) {
            Some(context) => context.clone(),
            None => {
                // Init new device
                let context = Arc::new(Context::new::<G>(device));
                pool.contexts.insert(Key::new::<G>(device), context.clone());
                context
            }
        }
    } else {
        // Initialize pool
        let context = Arc::new(Context::new::<G>(device));
        let mut new_pool = ContextPool::default();

        new_pool
            .contexts
            .insert(Key::new::<G>(device), context.clone());
        *pool = Some(new_pool);
        context
    };

    context
}
