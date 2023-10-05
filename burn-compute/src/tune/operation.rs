use core::hash::Hash;

use crate::server::{ComputeServer, Handle};

/// TODO
pub trait InputHashable<S: ComputeServer>: PartialEq + Eq + Hash {
    /// TODO
    fn custom_hash(&self) -> String;
    /// TODO
    fn make_handles(&self) -> &[&Handle<S>];
}

pub trait Operation<S: ComputeServer>: PartialEq + Eq + Hash {
    // Important: For input, Hash and stuff not derived, must be custom
    type Input: InputHashable<S>;
}
