//! Startup configuration cases run in separate processes because BurnConfig is immutable.

#![cfg(feature = "std")]

use burn_std::config::{BurnConfig, NanPolicy, RuntimeConfig, nan_policy};

#[test]
fn startup_cases() {
    for case in ["default", "propagate", "file", "set_over_file"] {
        let dir = tempfile::tempdir().unwrap();
        if matches!(case, "file" | "set_over_file") {
            std::fs::write(
                dir.path().join("burn.toml"),
                "[numerics]\nnan_policy = \"propagate\"\n",
            )
            .unwrap();
        }
        // The empty local file prevents finding a parent application's configuration.
        if !matches!(case, "file" | "set_over_file") {
            std::fs::write(dir.path().join("burn.toml"), "").unwrap();
        }
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "startup_child", "--ignored", "--nocapture"])
            .env("BURN_NUMERICS_TEST_CASE", case)
            .current_dir(dir.path())
            .status()
            .unwrap();
        assert!(status.success(), "startup case {case}");
    }
}

#[test]
#[ignore = "invoked in isolated processes by startup_cases"]
fn startup_child() {
    let case = std::env::var("BURN_NUMERICS_TEST_CASE").unwrap();
    let expected = match case.as_str() {
        "default" => NanPolicy::Native,
        "set_over_file" => {
            BurnConfig::set(BurnConfig::default().with_nan_policy(NanPolicy::Native));
            NanPolicy::Native
        }
        "propagate" => {
            BurnConfig::set(BurnConfig::default().with_nan_policy(NanPolicy::Propagate));
            NanPolicy::Propagate
        }
        "file" => NanPolicy::Propagate,
        _ => panic!("unknown startup case"),
    };
    assert_eq!(nan_policy(), expected);
    assert_eq!(BurnConfig::get().numerics().nan_policy, expected);
    assert_eq!(nan_policy(), expected);
}

#[test]
fn parses_policy_and_rejects_unknown_values() {
    for (value, expected) in [
        ("native", NanPolicy::Native),
        ("propagate", NanPolicy::Propagate),
    ] {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            format!("[numerics]\nnan_policy = \"{value}\"\n"),
        )
        .unwrap();
        let config = BurnConfig::from_file_path(file.path()).unwrap();
        assert_eq!(config.numerics().nan_policy, expected);
    }
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), "[numerics]\nnan_policy = \"ignore\"\n").unwrap();
    assert!(BurnConfig::from_file_path(file.path()).is_err());
}
