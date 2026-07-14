use crate::NdArray;
use burn_backend::distributed::DistributedOps;
use burn_backend::ops::TransactionOps;

impl TransactionOps<Self> for NdArray {}

// DistributedOps has default implementations; NdArray does not support collective operations.
impl DistributedOps<Self> for NdArray {}
