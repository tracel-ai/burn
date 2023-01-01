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

impl Default for ParamId {
    fn default() -> Self {
        Self::new()
    }
}

impl ParamId {
    pub fn new() -> Self {
        Self {
            value: nanoid::nanoid!(),
        }
    }
}

impl std::fmt::Display for ParamId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.value.as_str())
    }
}
