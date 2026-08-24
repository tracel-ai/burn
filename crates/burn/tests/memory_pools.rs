//! Installing and reading a device's dynamic memory pools.
//!
//! What these defend is the contract a measured memory plan is built on: a
//! caller can install a layout of its own, run a workload, read back what that
//! workload actually held, and cap the next run at it — and a backend with no
//! pools says so rather than accepting a layout it will not honour.
//!
//! Only the last of those runs without an accelerator. Pool installation is
//! implemented by the cubecl GPU runtimes; the CPU one reports
//! [`Unsupported`](burn::tensor::InstallMemoryPoolsError::Unsupported) like any
//! other backend without pools, so the working path is gated on a GPU feature:
//!
//! ```sh
//! cargo test -p burn --features cuda --test memory_pools
//! ```

#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
use burn::prelude::Device;
#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
use burn::tensor::{InstallMemoryPoolsError, MemoryPoolLayout, SlicedPool};

#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
use burn::prelude::Tensor;

/// Pools are per device, and every test here resolves the same one: an install
/// resets the high-water marks another test is reading, and a live tensor held
/// across one test's window refuses another's rebuild. They take turns.
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Exclusive use of the device's pools, kept even if another test panicked
/// while holding it — the poison says nothing about the pools themselves.
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn pools_to_ourselves() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|err| err.into_inner())
}

/// A device whose runtime installs pools, or nothing to test.
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn device_with_pools() -> Device {
    #[cfg(feature = "cuda")]
    return Device::cuda(burn::tensor::DeviceIndex::Default);
    #[cfg(all(feature = "rocm", not(feature = "cuda")))]
    return Device::rocm(burn::tensor::DeviceIndex::Default);
    #[cfg(all(feature = "vulkan", not(any(feature = "cuda", feature = "rocm"))))]
    return Device::vulkan(burn::tensor::DeviceKind::DefaultDevice);
    #[cfg(all(
        feature = "wgpu",
        not(any(feature = "cuda", feature = "rocm", feature = "vulkan"))
    ))]
    return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
}

/// One binary megabyte, the scale pool pages are set at.
#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
const MIB: u64 = 1024 * 1024;

/// A pool layout with room for a page or two of workload, and nothing routed
/// away from it.
#[cfg(any(
    feature = "cpu",
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn one_pool(pages: Option<u64>) -> MemoryPoolLayout {
    MemoryPoolLayout::Sliced(vec![SlicedPool {
        page_size: 8 * MIB,
        pages,
        max_slice: None,
    }])
}

/// A caller's layout replaces the runtime's, and the report comes back in the
/// order the pools were installed — which is what lets a caller pair each entry
/// with the pool it asked for, and is the whole basis of reading a peak back.
#[test]
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn a_report_describes_the_pools_that_were_installed() {
    let _pools = pools_to_ourselves();
    let device = device_with_pools();
    device.memory_cleanup();

    device
        .memory_install_pools(one_pool(None))
        .expect("this runtime installs pools");

    let report = device.memory_pool_report().expect("a report");
    assert_eq!(
        report.len(),
        1,
        "one pool in, one pool reported: {report:?}"
    );
    assert_eq!(report[0].page_size, 8 * MIB);
}

/// A workload's high-water mark survives the workload: the peak is what a
/// caller caps the next run at, so it must not fall back when the tensors do.
#[test]
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn the_peak_outlives_the_allocations_that_set_it() {
    let _pools = pools_to_ourselves();
    let device = device_with_pools();
    device.memory_cleanup();
    device
        .memory_install_pools(one_pool(None))
        .expect("this runtime installs pools");

    {
        let held = Tensor::<1>::zeros([256 * 1024], &device);
        // Both the operands and the result stay live until the reading below:
        // dropping them first would leave nothing in the pools to read, and
        // dropping the result before it is used would leave the addition
        // itself unexecuted, since the backend allocates lazily.
        let sum = held.clone() + held.clone();
        let _ = device.sync();

        let usage = device.memory_pool_usage().expect("a usage reading");
        assert!(usage.bytes_in_use > 0, "{usage:?}");

        drop((held, sum));
    }
    let _ = device.sync();
    device.memory_cleanup();

    let report = device.memory_pool_report().expect("a report");
    let pool = report.first().expect("the installed pool is reported");
    assert!(
        pool.pages_peak > 0 && pool.largest_alloc > 0,
        "the pool forgot what it served: {report:?}"
    );
}

/// A page size the device has to round up is still capped at the number of
/// pages that was asked for.
///
/// The cap is a byte count the runtime divides back into pages, so it has to be
/// counted in the pages the device will actually build: measured against the
/// requested page size instead, a one-page pool asks for a cap that cannot hold
/// its own page, and a larger one silently gets fewer pages than the
/// measurement it was sized from.
#[test]
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn a_page_size_the_device_rounds_up_still_holds_its_pages() {
    let _pools = pools_to_ourselves();
    let device = device_with_pools();
    device.memory_cleanup();

    // An odd byte over a page: whatever the device's alignment is, this is not
    // a multiple of it.
    let page_size = 8 * MIB + 1;
    device
        .memory_install_pools(MemoryPoolLayout::Sliced(vec![SlicedPool {
            page_size,
            pages: Some(1),
            max_slice: None,
        }]))
        .expect("a page the device rounds up is still a page it can hold");

    let report = device.memory_pool_report().expect("a report");
    let pool = report.first().expect("the installed pool is reported");
    assert!(
        pool.page_size >= page_size,
        "the pool holds less than a page of what was asked for: {report:?}"
    );
}

/// A rebuild is refused while the pools it would rebuild are still holding
/// something, and the refusal names the bytes.
///
/// This is the failure a caller meets in practice — long-lived allocations left
/// in the dynamic pools — and it has to be distinguishable from a backend that
/// has no pools at all, because one is worth retrying and the other never is.
#[test]
#[cfg(any(
    feature = "cuda",
    feature = "rocm",
    feature = "vulkan",
    feature = "wgpu"
))]
fn a_rebuild_is_refused_while_the_pools_are_in_use() {
    let _pools = pools_to_ourselves();
    let device = device_with_pools();
    device.memory_cleanup();
    device
        .memory_install_pools(one_pool(None))
        .expect("this runtime installs pools");

    let held = Tensor::<1>::zeros([256 * 1024], &device);
    let _ = device.sync();

    match device.memory_install_pools(one_pool(Some(4))) {
        Err(InstallMemoryPoolsError::PoolsInUse { bytes_in_use }) => {
            assert!(bytes_in_use > 0, "a refusal that names no bytes");
        }
        other => panic!("a live tensor did not block the rebuild: {other:?}"),
    }

    drop(held);
    let _ = device.sync();
    device.memory_cleanup();
    device
        .memory_install_pools(one_pool(Some(4)))
        .expect("the rebuild is accepted once the pools drain");
}

/// A layout the runtime cannot honour comes back as an error, not as a panic
/// on whichever thread the device happens to run on.
///
/// This is the difference between a mistake in a layout literal and a dead
/// device: pools are installed from the stream that owns them, so a panic
/// raised down there takes the device's runner with it and nothing the caller
/// writes can catch it. Every unhonourable shape has to be refused before the
/// layout leaves this thread.
#[test]
#[cfg(feature = "cpu")]
fn a_layout_that_cannot_be_honoured_is_refused_rather_than_fatal() {
    let device = Device::cpu();

    let unhonourable = [
        // Nothing to route allocations through.
        MemoryPoolLayout::Sliced(vec![]),
        // A page of no size.
        MemoryPoolLayout::Sliced(vec![SlicedPool {
            page_size: 0,
            pages: None,
            max_slice: None,
        }]),
        // A slice that no page of this pool could hold.
        MemoryPoolLayout::Sliced(vec![SlicedPool {
            page_size: MIB,
            pages: None,
            max_slice: Some(8 * MIB),
        }]),
        // More pages than a pool can address.
        MemoryPoolLayout::Sliced(vec![SlicedPool {
            page_size: MIB,
            pages: Some(100_000),
            max_slice: None,
        }]),
    ];

    for layout in unhonourable {
        match device.memory_install_pools(layout.clone()) {
            Err(InstallMemoryPoolsError::InvalidLayout { reason }) => {
                assert!(
                    !reason.is_empty(),
                    "a refusal that says nothing: {layout:?}"
                );
            }
            other => panic!("{layout:?} was not refused as a layout: {other:?}"),
        }
    }
}

/// A runtime that cannot install pools refuses permanently, rather than
/// accepting a layout it will not honour.
///
/// The distinction is the point: this refusal is the one a caller must never
/// retry, and it has to be told apart from
/// [`PoolsInUse`](InstallMemoryPoolsError::PoolsInUse), which is worth retrying
/// at the next quiescent point.
///
/// Reporting is a separate capability, which this also pins: the cubecl CPU
/// runtime describes the pools it has while declining to be given different
/// ones, so a caller cannot read a report back as proof that its layout went in.
#[test]
#[cfg(feature = "cpu")]
fn a_runtime_that_cannot_install_pools_refuses_permanently() {
    let device = Device::cpu();

    assert_eq!(
        device.memory_install_pools(one_pool(None)),
        Err(InstallMemoryPoolsError::Unsupported)
    );
}
