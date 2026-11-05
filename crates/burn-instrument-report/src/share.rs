//! A part of a whole as a report prints it.

use std::fmt::{self, Display};
use std::time::Duration;

/// `part` as a percentage of `whole`, to a tenth, or blank where there is no whole to take one
/// of.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Share {
    pub(crate) part: Duration,
    pub(crate) whole: Duration,
}

impl Share {
    /// `part` of `whole`, from 0 to 1; `None` where there is no whole to take one of.
    pub(crate) fn fraction(self) -> Option<f64> {
        (!self.whole.is_zero()).then(|| self.part.as_secs_f64() / self.whole.as_secs_f64())
    }
}

impl Display for Share {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.fraction() {
            Some(fraction) => Percent(fraction).fmt(f),
            None => f.pad(""),
        }
    }
}

/// A fraction from 0 to 1 as a report prints it: a percentage, to a tenth.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Percent(pub(crate) f64);

impl Display for Percent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(&format!("{:.1}%", self.0 * 100.0))
    }
}
