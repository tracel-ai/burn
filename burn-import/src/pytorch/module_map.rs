use strum_macros::EnumString;

/// Type of the module used to tag incoming tensors for automatic conversion to burn module structures.
///
/// These modules have weights and their names/shapes need to be converted to burn's structure.
#[derive(Debug, Clone, EnumString)]
pub enum ModuleType {
    BatchNorm,
    Conv1d,
    Conv2d,
    ConvTranspose1d,
    ConvTranspose2d,
    Embedding,
    GroupNorm,
    LayerNorm,
    Linear,
}
