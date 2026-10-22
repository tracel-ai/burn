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

impl Display for Share {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.whole.is_zero() {
            return f.pad("");
        }
        let percent = self.part.as_secs_f64() / self.whole.as_secs_f64() * 100.0;
        f.pad(&format!("{percent:.1}%"))
    }
}
