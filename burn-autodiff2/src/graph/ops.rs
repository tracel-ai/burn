use std::sync::Arc;

use burn_tensor::backend::Backend;
use dashmap::DashMap;

use crate::grads::Gradients;

pub struct OpsID {
    value: String,
}

pub struct OpsMetadata {
    parents: Vec<OpsID>,
}

pub type OpsBoxed<B> = Box<dyn Ops<B>>;
pub trait Ops<B: Backend>: Send + Sync {
    fn backward_step(self: Box<Self>, grads: &mut Gradients<B>);
    fn metadata(&self) -> &OpsMetadata;
}

pub type OpsMapRef<B> = Arc<OpsMap<B>>;
pub struct OpsMap<B: Backend> {
    map: DashMap<OpsID, OpsBoxed<B>>,
}

impl<B: Backend> std::fmt::Debug for OpsMap<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("OpsMap<{:?}>", B::name()).as_str())
    }
}
