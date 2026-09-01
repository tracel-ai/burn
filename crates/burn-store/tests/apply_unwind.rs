//! `ModuleSnapshot::apply` moves the module out of `&mut self` and writes a mapped one back.
//! A panic in between leaves `self` moved-from, so it must end the process rather than return
//! and let the owner drop that value a second time.
//!
//! The abort is only observable from outside, so the test re-runs itself in a child process
//! and inspects how that child died. Regression test for #5477.

#![cfg(feature = "std")]

// The `Module` derive expands to `::burn::...` paths.
use burn_core as burn;

use burn_core::module::{Module, Param, ParamId};
use burn_core::tensor::{Device, Tensor, TensorData};
use burn_pack::Tensor as PackTensor;
use burn_store::{ModuleAdapter, ModuleContext, ModuleSnapshot};

/// Set only in the child, which is the process that actually runs the panicking apply.
const CHILD: &str = "BURN_STORE_APPLY_UNWIND_CHILD";

/// What a test binary exits with when the harness catches an ordinary panic. Seeing this
/// instead of an abort is exactly the pre-fix behaviour, so the test distinguishes the two.
const HARNESS_TEST_FAILED: i32 = 101;

#[derive(Module, Debug)]
struct Model {
    w: Param<Tensor<1>>,
}

/// Stands in for any caller code that can panic inside `map`: an adapter, or a backend's
/// `from_data`. `ModuleAdapter` is a public trait, so this is ordinary safe code.
#[derive(Debug, Clone, Default)]
struct PanickingAdapter;

impl ModuleAdapter for PanickingAdapter {
    fn adapt(&self, _tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        panic!("adapter panicked mid-map");
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}

#[test]
fn a_panic_inside_map_aborts_instead_of_double_dropping() {
    if std::env::var_os(CHILD).is_some() {
        let device: Device = Default::default();
        let mut model = Model {
            w: Param::initialized(ParamId::new(), Tensor::<1>::from_data([1.0, 2.0], &device)),
        };
        let tensor =
            burn_store::bridge::from_data(TensorData::from([9.0f32, 8.0]), "w".into(), None);

        // Catching here is what a caller trying to recover would do, and is what made the
        // double drop reachable: without the guard this returns and `model` is dropped again.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            model.apply(vec![tensor], None, Some(Box::new(PanickingAdapter)), false)
        }));

        unreachable!("the abort guard should have ended the process before reaching here");
    }

    let output = std::process::Command::new(
        std::env::current_exe().expect("the test binary should know its own path"),
    )
    .args([
        "a_panic_inside_map_aborts_instead_of_double_dropping",
        "--exact",
        "--nocapture",
    ])
    .env(CHILD, "1")
    .output()
    .expect("re-running the test binary should succeed");

    assert!(
        !output.status.success(),
        "the child returned from a panicking apply instead of aborting"
    );
    assert_ne!(
        output.status.code(),
        Some(HARNESS_TEST_FAILED),
        "the child unwound and let the harness catch the panic, which means it returned \
         through the window where the module is moved-from"
    );

    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;

        // SIGABRT. Spelled out rather than pulled from `libc`, which is not a dependency here.
        const SIGABRT: i32 = 6;
        assert_eq!(
            output.status.signal(),
            Some(SIGABRT),
            "expected the child to abort, got {:?}",
            output.status
        );
    }
}
