//! Regression test for the former unsound visitor cloning in
//! `pytorch_reader::nested::de`.
//!
//! The former `Deserializer::deserialize_enum` implementation cloned the
//! caller-supplied visitor with a raw `ptr::copy_nonoverlapping` bitwise copy and
//! no `Copy`/`Clone` bound. For a visitor that owns heap memory, this aliased the
//! allocation across two owners and caused a double free. This regression test
//! should complete successfully under Miri:
//!
//! ```sh
//! cargo +nightly miri test -p pytorch-reader --test nested_de_miri
//! ```

use pytorch_reader::nested::adapter::DefaultAdapter;
use pytorch_reader::nested::data::NestedValue;
use pytorch_reader::nested::de::Deserializer;
use serde::Deserialize;
use serde::de::{EnumAccess, IgnoredAny, Visitor};
use std::collections::HashMap;

/// A type that deserializes through `deserialize_enum` using a *stateful*
/// (non-zero-sized) visitor. Ordinary `#[derive(Deserialize)]` visitors are
/// zero-sized, which is what has hidden this bug in practice: a ZST bitwise
/// copy is harmless, whereas this visitor carries owned heap data.
#[derive(Debug, PartialEq)]
struct StatefulEnum;

impl<'de> Deserialize<'de> for StatefulEnum {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct StatefulVisitor {
            // A heap allocation owned by the visitor. If the visitor is
            // bitwise-duplicated this buffer is freed twice.
            _owned: Vec<String>,
        }

        impl<'de> Visitor<'de> for StatefulVisitor {
            type Value = StatefulEnum;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                formatter.write_str("an enum")
            }

            fn visit_enum<A>(self, data: A) -> Result<Self::Value, A::Error>
            where
                A: EnumAccess<'de>,
            {
                let _variant = data.variant::<IgnoredAny>()?;
                Ok(StatefulEnum)
            }
        }

        deserializer.deserialize_enum(
            "StatefulEnum",
            &["A", "B"],
            StatefulVisitor {
                _owned: vec!["owned by the visitor".to_string()],
            },
        )
    }
}

#[test]
fn deserialize_enum_with_non_zst_visitor_is_not_unsound() {
    let mut map = HashMap::new();
    map.insert("DType".to_string(), NestedValue::String("A".to_string()));

    let de = Deserializer::<DefaultAdapter>::new(NestedValue::Map(map), false);
    let value = StatefulEnum::deserialize(de).expect("deserialization should succeed");
    assert_eq!(value, StatefulEnum);
}
