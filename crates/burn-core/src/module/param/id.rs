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
    pub fn serialize(self) -> String {
        BASE32_DNSSEC.encode(&self.value.to_le_bytes())
    }

    /// Deserialize a param id.
    ///
    /// Nb: The format used to be 6 bytes, so this
    /// code supports both.
    pub fn deserialize(encoded: &str) -> ParamId {
        let bytes = BASE32_DNSSEC
            .decode(encoded.as_bytes())
            .expect("Invalid id");
        let mut buffer = [0u8; 8];
        buffer[..bytes.len()].copy_from_slice(&bytes);
        ParamId::from(u64::from_le_bytes(buffer))
    }
}

impl core::fmt::Display for ParamId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.serialize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_serde_deserialize() {
        let val = ParamId::from(123456u64);
        let deserialized = ParamId::deserialize(&val.serialize());
        assert_eq!(val, deserialized);
    }

    #[test]
    fn param_serde_deserialize_legacy() {
        let legacy_val = [45u8; 6];
        let param_id = ParamId::deserialize(&BASE32_DNSSEC.encode(&legacy_val));
        assert_eq!(param_id.val().to_le_bytes()[0..6], legacy_val);
        assert_eq!(param_id.val().to_le_bytes()[6..], [0, 0]);
    }
}
