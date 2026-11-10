//! Tests of the compute environments a device keeps its compiled kernels and
//! autotune answers in: [`ComputeEnvironment`].
//!
//! Kept in its own test binary because the cubecl runtimes keep one active
//! environment for the whole process, so an activation reaches the work of
//! unrelated tests; the tests are `#[serial]` for the same reason. The
//! environments live in a temporary directory, never the machine's cache. Run
//! with the `remote` feature, they live on the server, in the directory it
//! keeps them in.

#![cfg(any(all(feature = "cube", feature = "persistence"), feature = "remote"))]

extern crate alloc;

pub type FloatElem = f32;
#[allow(unused)]
pub type IntElem = i32;

#[path = "common/backend.rs"]
mod backend;

use backend::*;
use burn_tensor::{
    Bytes, ComputeEnvironment, Device, EnvironmentBundle, EnvironmentError, EnvironmentSummary,
    PruneOptions, RecordPruning, TensorData,
};
use serial_test::serial;

/// What the device tunes while an environment is active lands in it, is in
/// its save, and is in an environment created from that save.
#[test]
#[serial]
fn a_saved_environment_holds_its_work_and_loads_back() {
    let device = device();
    let empty = ComputeEnvironment::new(&device, "empty").unwrap();
    let empty_size = empty.save().unwrap().bytes.len();

    let built = ComputeEnvironment::new(&device, "built").unwrap();
    let previous = built.activate(&device).unwrap();
    let lhs = TestTensor::<2>::ones([37, 53], &device);
    let rhs = TestTensor::<2>::ones([53, 29], &device);
    lhs.matmul(rhs).into_data();
    let bundle = built.save().unwrap();
    previous.activate(&device).unwrap();

    assert!(
        bundle.bytes.len() > empty_size,
        "the save holds what the work tuned: {} bytes, an empty one {empty_size}",
        bundle.bytes.len()
    );
    let loaded = ComputeEnvironment::load(&device, "loaded", bundle.clone()).unwrap();
    assert!(loaded.save().unwrap().bytes.len() > empty_size);
    let forked = loaded.fork("forked").unwrap();
    assert!(forked.save().unwrap().bytes.len() > empty_size);
}

/// Activating an environment answers the one it replaces, which activates
/// back.
#[test]
#[serial]
fn activating_answers_the_environment_it_replaces() {
    let device = device();
    let before = ComputeEnvironment::active(&device).unwrap();
    let other = ComputeEnvironment::new(&device, "other").unwrap();

    let replaced = other.activate(&device).unwrap();
    assert_eq!(replaced.name(), before.name());
    assert_eq!(
        ComputeEnvironment::active(&device).unwrap().name(),
        other.name()
    );

    replaced.activate(&device).unwrap();
    assert_eq!(
        ComputeEnvironment::active(&device).unwrap().name(),
        before.name()
    );
}

/// The environment a device's work uses is not replaced under it.
#[test]
#[serial]
fn the_active_environment_is_not_replaced() {
    let device = device();
    let used = ComputeEnvironment::new(&device, "used").unwrap();
    let previous = used.activate(&device).unwrap();

    let replaced = ComputeEnvironment::new(&device, "used");
    previous.activate(&device).unwrap();

    assert!(
        matches!(replaced, Err(EnvironmentError::Active { .. })),
        "{replaced:?}"
    );
}

/// An environment is not created from itself, which would empty it before
/// copying it in.
#[test]
#[serial]
fn an_environment_is_not_forked_into_itself() {
    let device = device();
    let source = ComputeEnvironment::new(&device, "source").unwrap();

    let forked = source.fork("source");

    assert!(
        matches!(forked, Err(EnvironmentError::SelfFork { .. })),
        "{forked:?}"
    );
}

/// A bundle that does not read replaces nothing: the environment of that name
/// keeps what it held.
#[test]
#[serial]
fn a_bundle_that_does_not_read_replaces_nothing() {
    let device = device();
    let kept = ComputeEnvironment::new(&device, "kept").unwrap();
    let previous = kept.activate(&device).unwrap();
    let lhs = TestTensor::<2>::ones([41, 59], &device);
    let rhs = TestTensor::<2>::ones([59, 31], &device);
    lhs.matmul(rhs).into_data();
    previous.activate(&device).unwrap();
    let held = kept.save().unwrap().bytes.len();

    let garbage = EnvironmentBundle {
        bytes: Bytes::from_bytes_vec(b"not an environment".to_vec()),
    };
    let refused = ComputeEnvironment::load(&device, "kept", garbage);

    assert!(
        matches!(refused, Err(EnvironmentError::Storage { .. })),
        "{refused:?}"
    );
    assert_eq!(kept.save().unwrap().bytes.len(), held);
}

/// A merge brings what another environment holds in beside what this one
/// already held, and leaves the other as it was.
#[test]
#[serial]
fn a_merge_keeps_what_the_environment_held_beside_what_it_brings() {
    let device = device();
    let (target, target_size) = built(&device, "merge-target", [43, 61, 23]);
    let (source, source_size) = built(&device, "merge-source", [47, 67, 19]);

    target.merge(&source).unwrap();

    let merged = target.save().unwrap().bytes.len();
    assert!(
        merged > target_size && merged > source_size,
        "the merge holds both: {merged} bytes, {target_size} and {source_size} apart"
    );
    assert_eq!(source.save().unwrap().bytes.len(), source_size);
}

/// A bundle merges as the environment it was saved from does, into the
/// active environment too.
#[test]
#[serial]
fn a_bundle_merges_into_the_active_environment() {
    let device = device();
    let (source, _) = built(&device, "bundle-source", [53, 71, 17]);
    let active = ComputeEnvironment::new(&device, "bundle-target").unwrap();
    let empty = active.save().unwrap().bytes.len();
    let previous = active.activate(&device).unwrap();

    let merged = active.merge_bundle(source.save().unwrap());
    let size = active.save().unwrap().bytes.len();
    previous.activate(&device).unwrap();

    merged.unwrap();
    assert!(size > empty, "{size} bytes, {empty} empty");
}

/// Merging an environment into itself changes nothing.
#[test]
#[serial]
fn an_environment_merged_into_itself_is_unchanged() {
    let device = device();
    let (itself, size) = built(&device, "merge-itself", [59, 73, 13]);

    itself.merge(&itself).unwrap();

    assert_eq!(itself.save().unwrap().bytes.len(), size);
}

/// A prune removes the build records and, where the runtime keeps compiled
/// kernels, those only the tune's race loaded, keeping those its work loaded;
/// the save shrinks by what went. The autotune answers stay: the work run
/// again from what is left computes the same without tuning again, so the
/// environment does not grow.
#[test]
#[serial]
fn a_pruned_environment_still_computes_its_work() {
    let device = device();
    let shape = [43, 31, 19];
    let (built, before) = built(&device, "pruned", shape);

    let pruned = built.prune(PruneOptions::default()).unwrap();
    let after = built.save().unwrap().bytes.len();

    assert!(pruned.records.removed > 0, "{pruned:?}");
    assert_eq!(pruned.records.kept, 0, "{pruned:?}");
    assert!(after < before, "{pruned:?}: {after} bytes, {before} before");
    let stored = pruned.kernels.total();
    match (KernelStorage::new(), stored) {
        (KernelStorage::NotKept, _) => assert_eq!(stored, 0, "{pruned:?}"),
        // A runtime whose target keeps none of what it compiles, as wgpu's
        // without SPIR-V, stores none however it is configured.
        (KernelStorage::Kept, 0) | (KernelStorage::OnServer, _) => {}
        (KernelStorage::Kept, _) => assert!(
            pruned.kernels.kept > 0 && pruned.kernels.removed > 0,
            "the kernels the work loaded stay and those only the race loaded go: {pruned:?}"
        ),
    }

    // Activating the pruned environment resets the tuner, so an answer the
    // prune removed would be tuned again and written back.
    let previous = built.activate(&device).unwrap();
    let product = product_of_ones(&device, shape).into_data();
    previous.activate(&device).unwrap();

    assert_product_of_ones(product, shape, &device);
    assert_eq!(
        built.save().unwrap().bytes.len(),
        after,
        "the work ran again from what the prune kept, and wrote nothing"
    );
}

/// A prune waits for the work queued before it, read back or not: what that
/// work tunes lands in the environment before the prune, rather than in the
/// pruned environment after it, and the result reads the same once pruned.
#[test]
#[serial]
fn a_prune_takes_the_work_queued_before_it() {
    let device = device();
    let shape = [37, 23, 29];
    let queued = ComputeEnvironment::new(&device, "pruned-queued").unwrap();
    let previous = queued.activate(&device).unwrap();

    let product = product_of_ones(&device, shape);
    let pruned = queued.prune(PruneOptions::default());
    let product = product.into_data();
    previous.activate(&device).unwrap();

    let pruned = pruned.unwrap();
    assert!(
        pruned.records.removed > 0,
        "the queued work's tune is recorded before the prune: {pruned:?}"
    );
    assert_product_of_ones(product, shape, &device);
}

/// A prune that keeps the build records removes none of them, and the build
/// left some to keep.
#[test]
#[serial]
fn a_prune_keeping_the_records_removes_none() {
    let device = device();
    let (built, _) = built(&device, "pruned-with-records", [41, 17, 23]);

    let pruned = built
        .prune(PruneOptions {
            records: RecordPruning::KeepAll,
        })
        .unwrap();

    assert_eq!(pruned.records.removed, 0, "{pruned:?}");
    assert!(pruned.records.kept > 0, "{pruned:?}");
}

/// An environment is counted as it stands, without saving it: empty when
/// created, then holding what the work tuned, which its save holds too.
#[test]
#[serial]
fn an_environment_is_counted_as_it_stands() {
    let device = device();
    let counted = ComputeEnvironment::new(&device, "counted").unwrap();
    assert_eq!(counted.summary().unwrap(), EnvironmentSummary::default());

    let previous = counted.activate(&device).unwrap();
    product_of_ones(&device, [47, 13, 31]).into_data();
    let summary = counted.summary().unwrap();
    previous.activate(&device).unwrap();

    assert!(summary.autotune_answers > 0, "{summary:?}");
    assert!(summary.records > 0, "{summary:?}");
    assert_eq!(counted.summary().unwrap(), summary);
}

/// A bundle saved under a name reads back from its file without being
/// loaded: the name, when and by which executable it was saved, and what the
/// environment held then; a file that is not there fails as every other
/// environment call does.
#[test]
#[serial]
#[cfg(feature = "persistence")]
fn a_named_save_reads_back_its_name_stamp_and_counts() {
    use burn_tensor::ReadBundle;

    let device = device();
    let (built, _) = built(&device, "named", [29, 41, 17]);
    let path = std::env::temp_dir().join(format!(
        "burn-compute-environment-{}-named.cubecl",
        std::process::id()
    ));

    std::fs::write(
        &path,
        &*built.save_named("built for this device").unwrap().bytes,
    )
    .unwrap();
    let read = EnvironmentBundle::read(&path).unwrap();

    assert_eq!(read.name().unwrap(), "built for this device");
    assert!(read.saved_at().unwrap().is_some());
    assert_eq!(read.summary().unwrap(), built.summary().unwrap());
    let saved_by = read.saved_by().unwrap();
    match KernelStorage::new() {
        // A remote environment is saved by the server's executable.
        KernelStorage::OnServer => assert!(saved_by.is_some()),
        KernelStorage::Kept | KernelStorage::NotKept => {
            assert_eq!(saved_by, Some(burn_tensor::BuildStamp::current()))
        }
    }
    std::fs::remove_file(&path).unwrap();
    assert!(
        matches!(
            EnvironmentBundle::read(&path),
            Err(EnvironmentError::Storage { .. })
        ),
        "a missing file answers with the error every environment call answers with"
    );
}

/// The default device, its environments kept in a directory of this run's.
///
/// The runtime's configuration places the environments when it first loads,
/// so the device is touched before the directory is set, or the load would
/// move it back.
fn device() -> Device {
    let device = Device::default();
    #[cfg(feature = "cube")]
    {
        static ROOT: std::sync::Once = std::sync::Once::new();
        ROOT.call_once(|| {
            TestTensor::<1>::from_data([0.0], &device).into_data();
            let root = std::env::temp_dir()
                .join(format!("burn-compute-environment-{}", std::process::id()));
            cubek::cubecl::environment::set_root(root);
        });
    }
    device
}

/// An environment the device's work wrote into, with its save's size.
fn built(device: &Device, name: &str, shape: [usize; 3]) -> (ComputeEnvironment, usize) {
    let built = ComputeEnvironment::new(device, name).unwrap();
    let previous = built.activate(device).unwrap();
    product_of_ones(device, shape).into_data();
    previous.activate(device).unwrap();
    let size = built.save().unwrap().bytes.len();
    (built, size)
}

/// The work the tests tune: the product of an `m`×`k` and a `k`×`n` matrix
/// of ones, registered and not yet read.
fn product_of_ones(device: &Device, [m, k, n]: [usize; 3]) -> TestTensor<2> {
    TestTensor::<2>::ones([m, k], device).matmul(TestTensor::<2>::ones([k, n], device))
}

/// `product` is what [`product_of_ones`] computes for `shape`: `k` everywhere.
fn assert_product_of_ones(product: TensorData, [m, k, n]: [usize; 3], device: &Device) {
    let expected = TestTensor::<2>::full([m, n], k as f32, device);
    product.assert_eq(&expected.into_data(), false);
}

/// Whether the runtime keeps the kernels it compiles, so an environment holds
/// some for a prune to tell apart.
///
/// This binary's harness loads the runtime's configuration in a constructor,
/// before any test could ask it to keep them, so it keeps them only when the
/// configuration file says so: the kernels a prune keeps and removes are
/// cubecl's tests' to prove, on a runtime that keeps them.
#[allow(
    dead_code,
    reason = "a build constructs only the variants its backend can answer"
)]
enum KernelStorage {
    /// The runtime keeps the kernels it compiles.
    Kept,
    /// The runtime's configuration keeps none.
    NotKept,
    /// The environments are a remote server's, configured where it runs.
    OnServer,
}

impl KernelStorage {
    fn new() -> Self {
        #[cfg(feature = "cube")]
        {
            use cubek::cubecl::config::{CubeClRuntimeConfig, RuntimeConfig};

            match CubeClRuntimeConfig::get().compilation.cache {
                true => Self::Kept,
                false => Self::NotKept,
            }
        }
        #[cfg(not(feature = "cube"))]
        {
            Self::OnServer
        }
    }
}
