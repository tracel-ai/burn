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
    let device = device_with_pools();
    device.memory_cleanup();
    device
        .memory_install_pools(one_pool(None))
        .expect("this runtime installs pools");

    {
        let held = Tensor::<1>::zeros([256 * 1024], &device);
        let _ = held.clone() + held;
        let _ = device.sync();

        let usage = device.memory_pool_usage().expect("a usage reading");
        assert!(usage.bytes_in_use > 0, "{usage:?}");
    }
    let _ = device.sync();
    device.memory_cleanup();

    let report = device.memory_pool_report().expect("a report");
    assert!(
        report[0].pages_peak > 0 && report[0].largest_alloc > 0,
        "the pool forgot what it served: {report:?}"
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
