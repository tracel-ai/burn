//! Regression tests for the previously-panicking and previously-lenient paths
//! in `pytorch_reader::nested::de`.
//!
//! These used to abort the process via `unimplemented!()`:
//! * `u32` / `i8` struct fields
//! * untagged enums (which route through `deserialize_any`)
//! * tuple structs
//!
//! They now deserialize normally; malformed input yields a `serde::de::Error`
//! instead of panicking. The tests below additionally pin the strict behaviour
//! requested in review: out-of-range integer conversions, tuple length
//! mismatches, and unresolvable enum variants must all error cleanly.
//!
//! Run with:
//! ```sh
//! cargo test -p pytorch-reader --test nested_de_robustness
//! ```

use pytorch_reader::nested::adapter::DefaultAdapter;
use pytorch_reader::nested::data::NestedValue;
use pytorch_reader::nested::de::Deserializer;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, PartialEq)]
struct HasU32 {
    n: u32,
}

#[derive(Debug, Deserialize, PartialEq)]
struct HasI8 {
    n: i8,
}

#[derive(Debug, Deserialize, PartialEq)]
#[serde(untagged)]
enum Untagged {
    Num(i32),
    Text(String),
}

#[derive(Debug, Deserialize, PartialEq)]
struct TupleHolder(i32, i32);

#[derive(Debug, Deserialize, PartialEq)]
enum Tagged {
    A,
    B,
}

fn de_map(pairs: Vec<(&str, NestedValue)>) -> Deserializer<DefaultAdapter> {
    let mut map = HashMap::new();
    for (key, value) in pairs {
        map.insert(key.to_string(), value);
    }
    Deserializer::<DefaultAdapter>::new(NestedValue::Map(map), false)
}

#[test]
fn u32_field_deserializes_without_panicking() {
    let de = de_map(vec![("n", NestedValue::U64(7))]);
    assert_eq!(HasU32::deserialize(de).unwrap(), HasU32 { n: 7 });
}

#[test]
fn i8_field_deserializes_without_panicking() {
    let de = de_map(vec![("n", NestedValue::I16(-1))]);
    assert_eq!(HasI8::deserialize(de).unwrap(), HasI8 { n: -1 });
}

#[test]
fn untagged_enum_deserializes_numeric_variant() {
    let de = Deserializer::<DefaultAdapter>::new(NestedValue::I32(5), false);
    assert_eq!(Untagged::deserialize(de).unwrap(), Untagged::Num(5));
}

#[test]
fn untagged_enum_deserializes_string_variant() {
    let de = Deserializer::<DefaultAdapter>::new(NestedValue::String("hi".to_string()), false);
    assert_eq!(
        Untagged::deserialize(de).unwrap(),
        Untagged::Text("hi".to_string())
    );
}

#[test]
fn untagged_enum_rejects_unsupported_value_with_source_type() {
    let de = Deserializer::<DefaultAdapter>::new(
        NestedValue::Unsupported("numpy.int64".to_string()),
        false,
    );
    let err = Untagged::deserialize(de).unwrap_err();
    assert!(err.to_string().contains("numpy.int64"), "{err}");
}

#[test]
fn tuple_struct_field_deserializes() {
    let de = Deserializer::<DefaultAdapter>::new(
        NestedValue::Vec(vec![NestedValue::I32(1), NestedValue::I32(2)]),
        false,
    );
    assert_eq!(TupleHolder::deserialize(de).unwrap(), TupleHolder(1, 2));
}

#[test]
fn type_mismatch_returns_error_instead_of_panicking() {
    // A string where a `u32` is expected must be a clean error, not a panic.
    let de = de_map(vec![("n", NestedValue::String("not a number".to_string()))]);
    assert!(HasU32::deserialize(de).is_err());
}

#[test]
fn out_of_range_i8_returns_error() {
    // 300 does not fit in `i8`; it must error instead of wrapping.
    let de = de_map(vec![("n", NestedValue::I32(300))]);
    assert!(HasI8::deserialize(de).is_err());
}

#[test]
fn out_of_range_u32_returns_error() {
    // `u32::MAX + 1` does not fit in `u32`; it must error instead of wrapping.
    let de = de_map(vec![("n", NestedValue::U64(u64::from(u32::MAX) + 1))]);
    assert!(HasU32::deserialize(de).is_err());
}

#[test]
fn tuple_length_mismatch_returns_error() {
    // Three elements for a two-element tuple struct must error, not drop data.
    let de = Deserializer::<DefaultAdapter>::new(
        NestedValue::Vec(vec![
            NestedValue::I32(1),
            NestedValue::I32(2),
            NestedValue::I32(3),
        ]),
        false,
    );
    assert!(TupleHolder::deserialize(de).is_err());
}

#[test]
fn enum_with_tag_resolves_variant() {
    let de = de_map(vec![("DType", NestedValue::String("B".to_string()))]);
    assert_eq!(Tagged::deserialize(de).unwrap(), Tagged::B);
}

#[test]
fn enum_without_tag_is_rejected() {
    // No `"DType"` tag at all: the variant is ambiguous and must be rejected.
    let de = de_map(vec![]);
    assert!(Tagged::deserialize(de).is_err());
}

#[test]
fn enum_with_non_map_value_is_rejected() {
    let de = Deserializer::<DefaultAdapter>::new(NestedValue::I32(1), false);
    assert!(Tagged::deserialize(de).is_err());
}
