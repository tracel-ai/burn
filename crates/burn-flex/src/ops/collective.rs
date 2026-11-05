use burn_backend::ops::CollectiveOps;

use crate::Flex;

// Flex has one device, so the emulated collectives are all it needs.
impl CollectiveOps<Flex> for Flex {}
