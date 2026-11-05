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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

/// One dynamic pool that gives each allocation a device buffer of its own, kept
/// for reuse or not: its measured state, which a sliced pool's page size cannot
/// describe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExclusivePoolReport {
    /// Pages, one an allocation, currently held.
    pub pages: u64,
    /// The most pages ever held at once.
    pub pages_peak: u64,
    /// The largest single allocation served, in requested bytes, which bounds
    /// every page's.
    pub largest_alloc: u64,
}

/// What the allocator of the calling stream holds: its state across its
/// pools, and each dynamic pool's. A runtime keeps a stream's pools apart from
/// every other's, so a workload is measured on the stream that runs it, and
/// another workload's marks never count in its own.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MemoryReport {
    /// The allocator's state, across the stream's pools.
    pub usage: MemoryPoolUsage,
    /// Each pool that carves pages, in the order allocations are routed
    /// through them, the pools a growth left behind last, then those of the
    /// memories the runtime keeps beside them for its own use (staging and
    /// uniform buffers).
    pub pools: alloc::vec::Vec<SlicedPoolReport>,
    /// Each dynamic pool giving each allocation a page of its own, in the same
    /// order. The persistent pool, which holds what is kept for good, and the
    /// allocations made outside every pool
    /// ([`AllocationMode::Dedicated`]) are in neither list.
    pub exclusive_pools: alloc::vec::Vec<ExclusivePoolReport>,
}

/// Where the allocations made inside a window are placed, for buffers whose
/// lifetime the dynamic pools would serve badly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocationMode {
    /// Kept for good — weights, tables — packed tightly and reused only at the
    /// same size, so they neither fragment nor shape the dynamic pools.
    Persistent,
    /// A device allocation each, outside every pool and given back to the
    /// driver once freed: a buffer too large or too long-lived for a pool's
    /// pages, such as a cache sized from the room a device has left, whose
    /// size would otherwise set the pages every later transient is carved
    /// from.
    Dedicated,
}
