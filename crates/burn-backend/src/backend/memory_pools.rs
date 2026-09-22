//! What a device's dynamic memory pools held.
//!
//! An allocator that only grows keeps whatever page it ever needed, so a
//! long-running workload reserves its worst moment for life. These types read
//! back what a workload actually held, so what its pages cost is a measurement
//! rather than a guess.
//!
//! The vocabulary is the backend's own rather than any runtime's, since
//! [`Backend`](super::Backend) is also implemented by backends with no pools at
//! all.


/// One dynamic pool's measured state, in the order allocations are routed
/// through the pools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlicedPoolReport {
    /// Size of each page in bytes.
    pub page_size: u64,
    /// Pages currently held.
    pub pages: u64,
    /// The most pages ever held at once.
    pub pages_peak: u64,
    /// The largest single allocation served, in requested bytes.
    pub largest_alloc: u64,
}

/// One reading of a device allocator's state, across the runtime's streams and
/// pools.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryPoolUsage {
    /// Live allocations, not pages.
    pub number_allocs: u64,
    /// Bytes those allocations use, excluding padding.
    pub bytes_in_use: u64,
    /// Bytes of padding inside them.
    pub bytes_padding: u64,
    /// Total bytes reserved on the device: at least `bytes_in_use`, plus pages
    /// held for reuse.
    pub bytes_reserved: u64,
}

