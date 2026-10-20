//! A [`Duration`] as a whole number of nanoseconds, which any reader of
//! the file can use.

use std::time::Duration;

use serde::{Deserialize, Deserializer, Serializer};

pub fn serialize<S: Serializer>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
    // Saturating: a pass would have to run for 584 years to overflow.
    let nanoseconds = u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX);
    serializer.serialize_u64(nanoseconds)
}

pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Duration, D::Error> {
    u64::deserialize(deserializer).map(Duration::from_nanos)
}

/// An optional [`Duration`], as nanoseconds or `null`.
pub mod optional {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(
        duration: &Option<Duration>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        match duration {
            Some(duration) => super::serialize(duration, serializer),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Option<Duration>, D::Error> {
        Option::<u64>::deserialize(deserializer)
            .map(|nanoseconds| nanoseconds.map(Duration::from_nanos))
    }
}
