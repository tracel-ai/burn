//! Transaction operations for the Flex backend.

use crate::Flex;
use burn_backend::distributed::DistributedOps;
use burn_backend::ops::{CollectiveOps, TransactionOps};

// TransactionOps has default implementations.
impl TransactionOps<Flex> for Flex {}

// DistributedOps has default implementations; Flex does not support collective operations.
impl DistributedOps<Flex> for Flex {}

// Flex has one device, so the composed collectives are all it needs.
impl CollectiveOps<Flex> for Flex {}
