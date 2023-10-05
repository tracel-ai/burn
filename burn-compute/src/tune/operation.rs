use core::hash::Hash;

use crate::server::ComputeServer;

/// TODO
pub trait InputHashable: PartialEq + Eq + Hash {
    /// TODO
    fn custom_hash(&self) -> String;
}

pub trait Operation<S: ComputeServer>: PartialEq + Eq + Hash {
    // Important: For input, Hash and stuff not derived, must be custom
    type Input: InputHashable;
}
