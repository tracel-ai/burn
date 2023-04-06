use alloc::string::{String, ToString};

use burn_common::id::IdGenerator;

use serde::{Deserialize, Serialize};

#[derive(Debug, Hash, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct ParamId {
    value: String,
}

impl From<&str> for ParamId {
    fn from(val: &str) -> Self {
        Self {
            value: val.to_string(),
        }
    }
}

impl From<String> for ParamId {
    fn from(value: String) -> Self {
        Self { value }
    }
}

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamId {
    pub fn new() -> Self {
        Self {
            value: IdGenerator::generate(),
        }
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.value.as_str())
    }
}
