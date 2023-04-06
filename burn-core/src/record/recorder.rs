use alloc::format;
use alloc::string::String;
use serde::{de::DeserializeOwned, Serialize};

/// Record any item implementing [Serialize](Serialize) and [DeserializeOwned](DeserializeOwned).
pub trait Recorder: Send + Sync + core::default::Default + core::fmt::Debug {
    /// Arguments used to record objects.
    type RecordArgs: Clone;
    /// Record output type.
    type RecordOutput;
    /// Arguments used to load recorded objects.
    type LoadArgs: Clone;

    fn record<Item: Serialize + DeserializeOwned>(
        item: Item,
        args: Self::RecordArgs,
    ) -> Result<Self::RecordOutput, RecorderError>;
    /// Load an item from the given arguments.
    fn load<Item: Serialize + DeserializeOwned>(
        args: Self::LoadArgs,
    ) -> Result<Item, RecorderError>;
}

#[derive(Debug)]
pub enum RecorderError {
    FileNotFound(String),
    Unknown(String),
}

impl core::fmt::Display for RecorderError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format!("{self:?}").as_str())
    }
}

// TODO: Move from std to core after Error is core (see https://github.com/rust-lang/rust/issues/103765)
#[cfg(feature = "std")]
impl std::error::Error for RecorderError {}

pub(crate) fn bin_config() -> bincode::config::Configuration {
    bincode::config::standard()
}
