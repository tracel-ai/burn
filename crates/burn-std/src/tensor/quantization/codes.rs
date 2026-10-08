use alloc::vec::Vec;
use cubecl_common::{e4m3, e5m2};

use super::QuantValue;

/// The codes a quantization value type stores, and the values they stand for.
///
/// Slices rather than single values, so the type is matched once and the loop, generic over the
/// scales, compiles in the caller's crate where it inlines and vectorizes. That takes a `scale_of`
/// owning what it reads: a scale behind a reference is reloaded for every value.
pub trait QuantCodes {
    /// The code of each value divided by its scale, `scale_of` giving the scale at an index:
    /// clamped to the type's range, then rounded to the nearest representable value, ties to even.
    fn encode_all(&self, values: &[f32], scale_of: impl Fn(usize) -> f32) -> Vec<i8>;

    /// The value each code stands for, times the scale `scale_of` gives at its index.
    fn decode_all(&self, codes: &[i8], scale_of: impl Fn(usize) -> f32) -> Vec<f32>;

    /// Whether codes are the integers they stand for, so they compare as their values do; a float
    /// code is sign and magnitude, where a larger code can stand for a more negative value.
    fn codes_are_integers(&self) -> bool;
}

impl QuantCodes for QuantValue {
    fn encode_all(&self, values: &[f32], scale_of: impl Fn(usize) -> f32) -> Vec<i8> {
        let (min, max) = self.range();
        let scaled = values
            .iter()
            .enumerate()
            .map(move |(index, &value)| (value / scale_of(index)).clamp(min, max));
        match self {
            QuantValue::E4M3 => scaled
                .map(|scaled| e4m3::from_f32(scaled).to_bits() as i8)
                .collect(),
            QuantValue::E5M2 => scaled
                .map(|scaled| e5m2::from_f32(scaled).to_bits() as i8)
                .collect(),
            QuantValue::E2M1 => scaled.map(|scaled| E2M1::encode(scaled) as i8).collect(),
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S => scaled.map(|scaled| HalfEven::round(scaled) as i8).collect(),
        }
    }

    fn decode_all(&self, codes: &[i8], scale_of: impl Fn(usize) -> f32) -> Vec<f32> {
        let codes = codes.iter().enumerate();
        match self {
            QuantValue::E4M3 => codes
                .map(move |(index, &code)| scale_of(index) * e4m3::from_bits(code as u8).to_f32())
                .collect(),
            QuantValue::E5M2 => codes
                .map(move |(index, &code)| scale_of(index) * e5m2::from_bits(code as u8).to_f32())
                .collect(),
            QuantValue::E2M1 => codes
                .map(move |(index, &code)| scale_of(index) * E2M1::decode(code as u8))
                .collect(),
            QuantValue::Q8F
            | QuantValue::Q8S
            | QuantValue::Q4F
            | QuantValue::Q4S
            | QuantValue::Q2F
            | QuantValue::Q2S => codes
                .map(move |(index, &code)| scale_of(index) * code as f32)
                .collect(),
        }
    }

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

/// Rounding half to even, which a GPU's `round` does and a CPU's does not.
struct HalfEven;

impl HalfEven {
    /// 1.5 * 2^23: a sum this large keeps no fraction bits, so adding it rounds to nearest even.
    const SHIFT: f32 = 12_582_912.0;

    /// Exact below 2^22 in magnitude, which every integer code range is, and plain arithmetic, so
    /// it vectorizes where `round` is a library call.
    fn round(value: f32) -> f32 {
        value + Self::SHIFT - Self::SHIFT
    }
}

/// E2M1 codes, encoded here because cubecl-common's E2M1 type needs std.
struct E2M1;

impl E2M1 {
    /// The magnitudes of the codes without their sign bit, in code order.
    const MAGNITUDES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    const SIGN: u8 = 0x8;

    /// The code counts the midpoints between magnitudes that `value` clears; strict and non-strict
    /// comparisons alternate so each tie lands on the even code, and a NaN clears none.
    fn encode(value: f32) -> u8 {
        let sign = if value.is_sign_negative() {
            Self::SIGN
        } else {
            0
        };
        let magnitude = num_traits::Float::abs(value);
        let cleared = [
            magnitude > 0.25,
            magnitude >= 0.75,
            magnitude > 1.25,
            magnitude >= 1.75,
            magnitude > 2.5,
            magnitude >= 3.5,
            magnitude > 5.0,
        ];
        cleared.into_iter().filter(|&above| above).count() as u8 | sign
    }

    fn decode(code: u8) -> f32 {
        let magnitude = Self::MAGNITUDES[usize::from(code & !Self::SIGN)];
        if code & Self::SIGN != 0 {
            -magnitude
        } else {
            magnitude
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(value: QuantValue, scaled: f32) -> i8 {
        value.encode_all(&[scaled], |_| 1.0)[0]
    }

    fn decode(value: QuantValue, code: i8) -> f32 {
        value.decode_all(&[code], |_| 1.0)[0]
    }

    #[test]
    fn float_codes_decode_to_their_format_values() {
        for (value, code, expected) in [
            (QuantValue::E4M3, 0x7E, 448.0),
            (QuantValue::E4M3, 0x38, 1.0),
            (QuantValue::E4M3, 0x01, 1.0 / 512.0),
            (QuantValue::E4M3, 0xC4, -3.0),
            (QuantValue::E5M2, 0x7B, 57344.0),
            (QuantValue::E5M2, 0x3C, 1.0),
            (QuantValue::E5M2, 0x01, 1.0 / 65536.0),
            (QuantValue::E5M2, 0x7C, f32::INFINITY),
            (QuantValue::E2M1, 0x1, 0.5),
            (QuantValue::E2M1, 0x7, 6.0),
            (QuantValue::E2M1, 0xD, -3.0),
        ] {
            assert_eq!(
                decode(value, code as u8 as i8),
                expected,
                "{value:?} {code:#x}"
            );
        }
        assert!(decode(QuantValue::E4M3, 0x7F).is_nan());
        assert!(decode(QuantValue::E5M2, 0x7D).is_nan());
    }

    #[test]
    fn every_finite_float_code_decodes_and_encodes_back() {
        for (value, codes) in [
            (QuantValue::E4M3, 0..=u8::MAX),
            (QuantValue::E5M2, 0..=u8::MAX),
            (QuantValue::E2M1, 0..=0x0F),
        ] {
            for code in codes {
                let decoded = decode(value, code as i8);
                if decoded.is_finite() {
                    assert_eq!(encode(value, decoded) as u8, code, "{value:?} {decoded}");
                }
            }
        }
    }

    #[test]
    fn codes_round_to_the_nearest_value_and_saturate() {
        let e2m1 = |scaled: f32| decode(QuantValue::E2M1, encode(QuantValue::E2M1, scaled));

        assert_eq!(e2m1(0.3), 0.5);
        assert_eq!(e2m1(0.25), 0.0, "a tie rounds to the even code");
        assert_eq!(
            encode(QuantValue::Q4S, -6.5),
            -6,
            "a tie rounds to the even integer"
        );
        assert_eq!(encode(QuantValue::Q8S, 2.5), 2);
        assert_eq!(encode(QuantValue::Q8S, 3.5), 4);
        assert_eq!(encode(QuantValue::Q8S, -126.5), -126);
        assert_eq!(encode(QuantValue::Q8S, 1.4), 1);
        assert_eq!(encode(QuantValue::Q8S, 1e9), 127);
        assert_eq!(encode(QuantValue::Q8S, -1e9), -127);
        assert_eq!(e2m1(2.4), 2.0);
        assert_eq!(e2m1(100.0), 6.0);
        assert_eq!(e2m1(-100.0), -6.0);
        assert_eq!(
            decode(QuantValue::E4M3, encode(QuantValue::E4M3, 1e6)),
            448.0
        );
    }
}
