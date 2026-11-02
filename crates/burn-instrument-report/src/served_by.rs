//! Which executions served a component, as a report prints them.

use std::fmt::{self, Display};

use burn_instrument::Component;

/// Which executions served a component's entries: empty where it declares no variant, which
/// is most components.
///
/// A single execution that served every entry prints as its bare name: its
/// count would only repeat the component's own. Several print with theirs, because
/// *how the entries split* is the whole reason to look — `tile ×8, tensor ops
/// ×1` says a routing decision went one way eight times and the other once, and
/// that is a finding.
pub(crate) struct ServedBy<'a>(pub(crate) &'a Component);

impl ServedBy<'_> {
    /// Whether the component declares no variant, so there is nothing to print.
    pub(crate) fn is_empty(&self) -> bool {
        self.0.variants.is_empty()
    }
}

impl Display for ServedBy<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let component = self.0;
        match component.variants.as_slice() {
            [] => Ok(()),
            [only] if only.entries == component.entries => f.write_str(&only.name),
            several => {
                for (index, variant) in several.iter().enumerate() {
                    if index > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{} ×{}", variant.name, variant.entries)?;
                }
                Ok(())
            }
        }
    }
}
