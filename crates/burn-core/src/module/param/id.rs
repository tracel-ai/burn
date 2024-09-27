use alloc::string::String;
use burn_common::id::IdGenerator;
use data_encoding::BASE32_DNSSEC;

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

    /// Convert the parameter ID into a string.
    pub fn encode(self) -> String {
        BASE32_DNSSEC.encode(&self.value.to_le_bytes())
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.encode())
    }
}
