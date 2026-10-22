use burn_backend::distributed::DistributedOps;

use crate::{Fusion, FusionBackend};

impl<B: FusionBackend> DistributedOps<Self> for Fusion<B> {}
