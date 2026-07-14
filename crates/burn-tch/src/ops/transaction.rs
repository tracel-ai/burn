use burn_backend::distributed::DistributedOps;
use burn_backend::ops::TransactionOps;

use crate::LibTorch;

impl TransactionOps<Self> for LibTorch {}

// DistributedOps has default implementations; LibTorch does not support collective operations.
impl DistributedOps<Self> for LibTorch {}
