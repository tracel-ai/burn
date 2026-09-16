//! Regression tests for the previously-panicking paths in
//! `burn_store::nested::de`.
//!
//! These used to abort the process via `unimplemented!()`:
//! * `u32` / `i8` struct fields
//! * untagged enums (which route through `deserialize_any`)
//! * tuple structs
//!
//! They now deserialize normally; malformed input yields a `serde::de::Error`
//! instead of panicking.
//!
//! Run with:
//! ```sh
//! cargo test -p burn-store --test nested_de_robustness
//! ```

#![cfg(feature = "pytorch")]

use burn_store::nested::adapter::DefaultAdapter;
use burn_store::nested::data::NestedValue;
use burn_store::nested::de::Deserializer;
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
