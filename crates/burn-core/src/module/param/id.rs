use alloc::string::ToString;
use burn_common::id::IdGenerator;

/// Parameter ID.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
pub struct ParamId {
    value: u64,
}

impl From<u64> for ParamId {
    fn from(value: u64) -> Self {
        Self { value }
    }
}

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamId {
    /// Create a new parameter ID.
    pub fn new() -> Self {
        Self {
            value: IdGenerator::generate(),
        }
    }

    /// Gets the internal value of the id.
    pub fn val(&self) -> u64 {
        self.value
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.value.to_string())
    }
}
