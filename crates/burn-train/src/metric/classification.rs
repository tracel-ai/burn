/// The reduction strategy for classification metrics.
#[derive(Copy, Clone, Default)]
pub enum ClassReduction {
    /// Computes the statistics over all classes before averaging
    Micro,
    /// Computes the statistics independently for each class before averaging
    #[default]
    Macro,
}
