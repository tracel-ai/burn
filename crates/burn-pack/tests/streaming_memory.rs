//! The peak-memory guarantee, measured rather than asserted in prose.
//!
//! `Writer` promises that a deferred tensor's bytes are dropped before the next tensor's are
//! requested, which is what bounds a save by the largest single tensor instead of by the whole
//! model. Every other test in the suite would still pass if `write_tensors` collected all the
//! bytes into a `Vec` first and wrote them afterwards: the providers would still run once each,
//! still in order, and the container would still be byte-identical.
//!
//! So this file counts live heap bytes with its own global allocator. It lives in a separate
//! test binary because that allocator is process-wide, and the counter would be meaningless
//! with unrelated tests allocating alongside it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Only payload-sized allocations are counted, so ordinary test and I/O bookkeeping stays out
/// of the measurement.
const THRESHOLD: usize = 256 * 1024;
const PAYLOAD: usize = 1024 * 1024;

static LIVE: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() && layout.size() >= THRESHOLD {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() >= THRESHOLD {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.dealloc(ptr, layout) };
    }

    // Forwarded rather than left to the default impls, which would drop `System`'s in-place
    // growth and its use of pre-zeroed pages for the whole binary.
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            if layout.size() >= THRESHOLD {
                LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            }
            if new_size >= THRESHOLD {
                LIVE.fetch_add(new_size, Ordering::Relaxed);
            }
        }
        new
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() && layout.size() >= THRESHOLD {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        ptr
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

use burn_pack::{Bytes, DType, Tensor, Writer};

/// Writing to a file streams, so the only large allocations are the tensor payloads
/// themselves. Each provider checks, before allocating its own, that no earlier tensor's
/// payload is still live.
#[test]
fn each_tensor_is_freed_before_the_next_is_produced() {
    let dir = tempfile::tempdir().unwrap();
    let dest = dir.path().join("model.bpk");

    let tensors: Vec<Tensor> = (0..4u8)
        .map(|i| {
            Tensor::deferred(
                format!("t{i}"),
                DType::U8,
                vec![PAYLOAD],
                None,
                PAYLOAD,
                move || {
                    let live = LIVE.load(Ordering::Relaxed);
                    assert_eq!(
                        live, 0,
                        "tensor t{i}: {live} bytes of an earlier tensor were still live"
                    );
                    Ok(Bytes::from_bytes_vec(vec![i; PAYLOAD]))
                },
            )
        })
        .collect();

    Writer::new(tensors).write_to_file(&dest).unwrap();

    assert_eq!(
        LIVE.load(Ordering::Relaxed),
        0,
        "the last tensor's payload outlived the write"
    );
    // Guards against a vacuous pass: the providers must actually have run and been written.
    assert!(
        std::fs::metadata(&dest).unwrap().len() as usize >= 4 * PAYLOAD,
        "all four tensors should have reached the file"
    );
}
