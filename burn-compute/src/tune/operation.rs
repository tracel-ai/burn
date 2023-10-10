use core::hash::Hash;

use crate::server::ComputeServer;

/// TODO
pub trait InputHashable: PartialEq + Eq + Hash {
    /// TODO
    fn custom_hash(&self) -> String;
}

pub trait Operation: PartialEq + Eq + Hash {
    type Input: InputHashable;
}
