//! Which member of the profiled device an execution ran on.

/// A member of the profiled device, by its index in member order: `0` on a device of one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Member(pub usize);

/// The device a window ran on, as far as its launching thread tells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RanOn {
    /// The one device a device of one runs everything on, whichever thread launched it.
    Device,
    /// A member, launched from its fusion worker.
    Member(Member),
    /// Some member, which one unknown: launched from a thread that is no member's worker, so it
    /// fills no member's gaps and opens none.
    UnknownMember,
}
