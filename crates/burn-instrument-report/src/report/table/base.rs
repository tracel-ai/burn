//! The table a report prints, section by section.

use std::fmt::{self, Display};

use super::components::ComponentTable;
use super::device::DeviceHeader;
use super::merges::MergeNotes;
use super::method::Method;
use super::recoverable::Recoverable;
use super::roofline_note::RooflineNote;
use super::totals::Totals;
use super::worklist::Worklist;
use crate::Report;

impl Display for Report<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let components = ComponentTable::new(self);
        write!(f, "{}", DeviceHeader::new(self))?;
        write!(f, "{}", Method::new(self))?;
        write!(f, "{components}")?;
        write!(f, "{}", Totals::new(self, components.width()))?;
        write!(f, "{}", RooflineNote::new(self))?;
        write!(f, "{}", Recoverable::new(self.ladder()))?;
        write!(f, "{}", MergeNotes::new(self))?;
        write!(f, "{}", Worklist::new(self))
    }
}
