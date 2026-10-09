use super::QuantValue;

/// What the codes of a quantization value type are.
pub trait QuantValueCodes {
    /// Whether codes are the integers they stand for, so they compare as their values do; a float
    /// code is sign and magnitude, where a larger code can stand for a more negative value.
    fn codes_are_integers(&self) -> bool;
}

impl QuantValueCodes for QuantValue {
    fn codes_are_integers(&self) -> bool {
        match self {
            QuantValue::E4M3 | QuantValue::E5M2 | QuantValue::E2M1 => false,
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S => true,
        }
    }
}
