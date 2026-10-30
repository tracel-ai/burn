use burn_backend::distributed::DistributedOps;

use crate::{BackendRouter, RouterChannel};

impl<R: RouterChannel> DistributedOps<Self> for BackendRouter<R> {}
