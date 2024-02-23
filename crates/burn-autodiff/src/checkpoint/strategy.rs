use core::fmt::Debug;

/// Enum variant of the strategy for the amount of checkpointing to do during autodiff/
pub enum Strategy {
    /// All operations are considered compute bound, notwithstanding how they are marked
    NoCheckpointing,

    /// Operation properties are as they are marked (compute or memory bound)
    BalancedCheckpointing,
}

/// Strategy for the amount of checkpointing to do during autodiff
pub trait CheckpointStrategy: Clone + Copy + Debug + Default + Send + Sync + 'static {
    /// Returns the enum variant of the struct implementing the trait
    fn as_enum() -> Strategy;
}

#[derive(Clone, Copy, Debug, Default)]
/// All operations are considered compute bound, notwithstanding how they are marked
pub struct NoCheckpointing {}
impl CheckpointStrategy for NoCheckpointing {
    fn as_enum() -> Strategy {
        Strategy::NoCheckpointing
    }
}

#[derive(Clone, Copy, Debug, Default)]
/// Operation properties are as they are marked (compute or memory bound)
pub struct BalancedCheckpointing {}
impl CheckpointStrategy for BalancedCheckpointing {
    fn as_enum() -> Strategy {
        Strategy::BalancedCheckpointing
    }
}
