//! A safetensors save draws its tensors during the write, so a failure partway through must
//! leave the previous file at the destination untouched rather than a truncated one.
//! Regression test for #5479.

#![cfg(all(feature = "std", feature = "safetensors"))]

// The `Module` derive expands to `::burn::...` paths.
use burn_core as burn;

use burn_core::module::{Module, Param, ParamId};
use burn_core::tensor::{DType, Device, Tensor};
use burn_pack::Tensor as PackTensor;
use burn_store::{ModuleAdapter, ModuleContext, ModuleSnapshot, SafetensorsStore};

#[derive(Module, Debug)]
struct Model {
    w: Param<Tensor<1>>,
}

fn model(device: &Device) -> Model {
    Model {
        w: Param::initialized(
            ParamId::new(),
            Tensor::<1>::from_data([1.0, 2.0, 3.0, 4.0], device),
        ),
    }
}

/// Replaces each tensor with one whose data cannot be produced, standing in for a device
/// readback that fails partway through a save.
#[derive(Debug, Clone, Default)]
struct FailingAdapter;

impl ModuleAdapter for FailingAdapter {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        burn_store::bridge::deferred(
            tensor.name.clone(),
            DType::F32,
            tensor.shape.clone(),
            None,
            || Err(burn_pack::Error::IoError("device readback failed".into())),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[test]
fn a_failed_save_leaves_the_previous_file_intact() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let device: Device = Default::default();
    let model = model(&device);

    let mut store = SafetensorsStore::from_file(&path);
    model
        .save_into(&mut store)
        .expect("the first save succeeds");
    let good = std::fs::read(&path).unwrap();
    assert!(!good.is_empty());

    let mut store = SafetensorsStore::from_file(&path)
        .overwrite(true)
        .with_to_adapter(FailingAdapter);
    let outcome = model.save_into(&mut store);

    assert!(outcome.is_err(), "the save should report the failure");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        good,
        "the previous file was modified by a save that failed"
    );
}

/// The scratch file the atomic write builds in must not outlive the failure that abandoned it.
#[test]
fn a_failed_save_leaves_no_scratch_file_behind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let device: Device = Default::default();
    let model = model(&device);

    let mut store = SafetensorsStore::from_file(&path).with_to_adapter(FailingAdapter);
    assert!(model.save_into(&mut store).is_err());

    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert!(
        leftovers.is_empty(),
        "a failed save left files behind: {leftovers:?}"
    );
}

/// The ordinary path still works: the container ends up at the destination, readable.
#[test]
fn a_successful_save_lands_at_the_destination() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let device: Device = Default::default();
    let model = model(&device);

    let mut store = SafetensorsStore::from_file(&path);
    model.save_into(&mut store).unwrap();

    let mut loaded = Model {
        w: Param::initialized(ParamId::new(), Tensor::<1>::zeros([4], &device)),
    };
    let mut store = SafetensorsStore::from_file(&path);
    loaded.load_from(&mut store).unwrap();

    assert_eq!(
        loaded.w.val().to_data().try_to_vec::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );

    // Nothing but the container itself.
    assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
}

/// Saving replaces the destination rather than truncating it, so the new file is a new inode.
/// Its permissions must not fall back to the process umask: a checkpoint the user narrowed to
/// `0600` would otherwise come back world-readable on the next save.
#[cfg(unix)]
#[test]
fn a_save_keeps_the_permissions_of_the_checkpoint_it_replaces() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("model.safetensors");
    let device: Device = Default::default();
    let model = model(&device);

    let mut store = SafetensorsStore::from_file(&path);
    model.save_into(&mut store).unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();

    let mut store = SafetensorsStore::from_file(&path).overwrite(true);
    model.save_into(&mut store).unwrap();

    assert_eq!(
        std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600,
        "the save republished the checkpoint at the process umask"
    );
}
