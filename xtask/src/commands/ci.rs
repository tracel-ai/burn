use xtask_common::{
    anyhow,
    commands::ci::{self, CICmdArgs, CICommand},
    utils::{helpers, rustup::rustup_add_target},
    ExecutionEnvironment,
};

// no-std additional build targets
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";

pub(crate) fn handle_command(
    mut args: CICmdArgs,
    exec_env: ExecutionEnvironment,
) -> anyhow::Result<()> {
    match exec_env {
        ExecutionEnvironment::NoStd => {
            // Install additional targets for no-std execution environments
            rustup_add_target(WASM32_TARGET)?;
            rustup_add_target(ARM_TARGET)?;
            let no_std_crates = vec![
                "burn",
                "burn-core",
                "burn-common",
                "burn-tensor",
                "burn-ndarray",
                "burn-no-std-tests",
            ];
            let no_std_features_args = vec!["--no-default-features"];
            let no_std_build_targets = ["Default", WASM32_TARGET, ARM_TARGET];
            no_std_build_targets.iter().try_for_each(|build_target| {
                let mut build_args = no_std_features_args.clone();
                if *build_target != "Default" {
                    build_args.extend(vec!["--target", *build_target]);
                }
                match args.command {
                    CICommand::Build => {
                        helpers::additional_crates_build(no_std_crates.clone(), build_args)
                    }
                    _ => Ok(()),
                }
            })?;
            let no_std_test_targets = ["Default"];
            no_std_test_targets.iter().try_for_each(|test_target| {
                let mut test_args = no_std_features_args.clone();
                if *test_target != "Default" {
                    test_args.extend(vec!["--target", *test_target]);
                }
                match args.command {
                    CICommand::UnitTests => {
                        helpers::additional_crates_unit_tests(no_std_crates.clone(), test_args)
                    }
                    _ => Ok(()),
                }
            })
        }
        ExecutionEnvironment::Std => {
            // Exclude crates that are not supported by CI
            match args.command {
                CICommand::Build | CICommand::UnitTests => {
                    args.exclude
                        .extend(vec!["burn-cuda".to_string(), "burn-tch".to_string()]);
                    if std::env::var("DISABLE_WGPU").is_ok() {
                        args.exclude.extend(vec!["burn-wgpu".to_string()]);
                    }
                }
                CICommand::DocTests => {
                    // Exclude problematic crates from documentation test
                    args.exclude.extend(vec!["burn-cuda".to_string()])
                }
                _ => {}
            };
            ci::handle_command(args.clone())?;
            // Specific additional commands to test specific features
            match args.command {
                CICommand::Build => {
                    // burn-dataset
                    helpers::additional_crates_build(vec!["burn-dataset"], vec!["--all-features"])?;
                }
                CICommand::UnitTests => {
                    // burn-dataset
                    helpers::additional_crates_unit_tests(
                        vec!["burn-dataset"],
                        vec!["--all-features"],
                    )?;
                    // burn-core
                    helpers::additional_crates_unit_tests(
                        vec!["burn-core"],
                        vec!["--features", "test-tch,record-item-custom-serde"],
                    )?;
                    if std::env::var("DISABLE_WGPU").is_err() {
                        helpers::additional_crates_unit_tests(
                            vec!["burn-core"],
                            vec!["--features", "test-wgpu"],
                        )?;
                    }
                    // MacOS specific tests
                    #[cfg(target_os = "macos")]
                    {
                        // burn-candle
                        helpers::additional_crates_unit_tests(
                            vec!["burn-candle"],
                            vec!["--features", "accelerate"],
                        )?;
                        // burn-ndarray
                        helpers::additional_crates_unit_tests(
                            vec!["burn-ndarray"],
                            vec!["--features", "blas-accelerate"],
                        )?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
    }
}
