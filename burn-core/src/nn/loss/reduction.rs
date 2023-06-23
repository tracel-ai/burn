/// The reduction type for the loss.
pub enum Reduction {
    /// The mean of the losses will be returned.
    Mean,

    /// The sum of the losses will be returned.
    Sum,

    /// The mean of the losses will be returned.
    Auto,
}
