use burn_backend::ops::CollectiveOps;

use crate::Flex;

// Flex has one device, so the composed collectives are all it needs.
impl CollectiveOps<Flex> for Flex {}
