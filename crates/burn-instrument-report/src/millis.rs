//! A duration as a report prints it.

use std::fmt::{self, Display};
use std::time::Duration;

/// A duration in milliseconds, to the precision the format asks for: `{:.3}` for the
/// microsecond a comparison needs, and a tenth — the precision a reader can act on — where
/// it asks for none.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Millis(pub(crate) Duration);

impl Display for Millis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let millis = self.0.as_secs_f64() * 1e3;
        match f.precision() {
            Some(_) => Display::fmt(&millis, f),
            None => f.pad(&format!("{millis:.1}")),
        }
    }
}
