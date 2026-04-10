//! Transaction operations for the Flex backend.

use crate::Flex;
use burn_backend::ops::TransactionOps;

// TransactionOps has default implementations.
impl TransactionOps<Flex> for Flex {}
