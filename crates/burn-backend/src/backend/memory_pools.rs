//! The layout of a device's dynamic memory pools, and what one reports.
//!
//! An allocator that only grows keeps whatever page it ever needed, so a
//! long-running workload reserves its worst moment for life. These types let a
//! caller install a layout of its own — a pool per size class, each capped at a
//! number of pages — and read back what the workload actually held, so the caps
//! are a measurement rather than a guess.
//!
//! The vocabulary is the backend's own rather than any runtime's, since
//! [`Backend`](super::Backend) is also implemented by backends with no pools at
//! all.

use alloc::string::String;
use alloc::vec::Vec;

/// A layout for a device's dynamic memory pools, applied with
/// [`Backend::memory_install_pools`](super::Backend::memory_install_pools).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MemoryPoolLayout {
    /// An ordered list of pools that sub-slice fixed-size pages. An allocation
    /// lands in the first pool whose [`max_slice`](SlicedPool::max_slice)
    /// accepts its size, so a small pool listed before a large one captures the
    /// small-allocation traffic. Pages are allocated on first use.
    Sliced(Vec<SlicedPool>),
    /// One direct pool for everything: every allocation is its own, sized to
    /// the request and reused by exact size — no pages, no sub-slicing, no
    /// padding beyond alignment. With no page size chosen in advance, this is
    /// what a workload runs on for its largest allocation to be read back as it
    /// was asked for.
    Direct,
    /// The runtime's default: a ladder of size-bucketed pools that sub-slice
    /// large pages.
    SubSlices,
    /// One page per allocation, in exponentially spaced size buckets.
    ExclusivePages,
}

/// One pool of a [`MemoryPoolLayout::Sliced`] layout: allocations are slices of
/// fixed-size pages, capped at `pages` pages. An allocation that no longer fits
/// goes to the next pool that accepts it, and fails when none does.
///
/// Sizes are rounded up to the device's alignment, so a pool holds at least
/// what it was asked for. A layout that cannot be honoured at all — a zero
/// size, `max_slice` past `page_size`, `pages` outside `1..=65535` — is refused
/// with [`InvalidLayout`](InstallMemoryPoolsError::InvalidLayout).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlicedPool {
    /// Size of each page in bytes; also the largest single allocation this pool
    /// can serve. Rounded up to the device's alignment.
    pub page_size: u64,
    /// How many pages the pool may hold. `None` grows without a cap, which is
    /// what a workload is measured on before its caps are known.
    pub pages: Option<u64>,
    /// Largest allocation routed to this pool; `None` accepts anything up to
    /// `page_size`. Later pools see only what this one declines.
    pub max_slice: Option<u64>,
}

/// One dynamic pool's measured state, in the order the pools were installed.
///
/// The read side of a measured layout: install a growable one, run the
/// workload, read these, re-install capped at `pages_peak`. Pool placement is
/// deterministic, so the same allocations fit the capped layout by
/// construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SlicedPoolReport {
    /// Size of each page in bytes; `0` for a [direct](MemoryPoolLayout::Direct)
    /// pool, which has no pages to size.
    pub page_size: u64,
    /// Pages currently held — for a direct pool, live allocations.
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

/// Why installing a pool layout did not take effect.
///
/// The distinction a caller needs is **transient or permanent**: a refused
/// rebuild is worth retrying once whatever holds the pools drains, while a
/// backend with no configurable pools refuses forever. Treating the two alike
/// either gives up on a layout that would have installed, or repeats an
/// expensive measurement that can never succeed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InstallMemoryPoolsError {
    /// The pools being rebuilt still hold live allocations, so the previous
    /// layout was kept. Transient.
    PoolsInUse {
        /// Bytes still live in those pools.
        bytes_in_use: u64,
    },
    /// The calling stream is already in an error state, so its pools were not
    /// rebuilt. The layout still applies to streams created afterwards, and the
    /// underlying failure surfaces at the next flush or sync.
    StreamUnavailable,
    /// This backend has no configurable dynamic pools. Permanent.
    Unsupported,
    /// The layout itself cannot be honoured, so the previous one was kept: an
    /// empty pool list, a zero size, a slice larger than its page, an
    /// unusable cap, or a pool shape this build has none of. Permanent — what
    /// has to change is the layout, not the moment it is installed at.
    InvalidLayout {
        /// What the backend objected to, in its own words.
        reason: String,
    },
}

impl core::fmt::Display for InstallMemoryPoolsError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PoolsInUse { bytes_in_use } => {
                write!(formatter, "{bytes_in_use} B still live in the pools")
            }
            Self::StreamUnavailable => {
                write!(formatter, "the calling stream is in an error state")
            }
            Self::Unsupported => {
                write!(formatter, "this backend has no configurable memory pools")
            }
            Self::InvalidLayout { reason } => {
                write!(formatter, "the pool layout cannot be honoured: {reason}")
            }
        }
    }
}

impl core::error::Error for InstallMemoryPoolsError {}
