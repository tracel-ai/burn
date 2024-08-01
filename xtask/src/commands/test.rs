use xtask_common::{
    anyhow,
    commands::test::{self, TestCmdArgs},
    utils::helpers,
    ExecutionEnvironment,
};

use crate::NO_STD_CRATES;

pub(crate) fn handle_command(
    mut args: TestCmdArgs,
    exec_env: ExecutionEnvironment,
) -> anyhow::Result<()> {
    match exec_env {
        ExecutionEnvironment::NoStd => {
            ["Default"].iter().try_for_each(|test_target| {
                let mut test_args = vec!["--no-default-features"];
                if *test_target != "Default" {
                    test_args.extend(vec!["--target", *test_target]);
                }
                helpers::custom_crates_tests(NO_STD_CRATES.to_vec(), test_args)
            })?;
            Ok(())
        }
        ExecutionEnvironment::Std => {
            // Exclude crates that are not supported on CI
            args.exclude
                .extend(vec!["burn-cuda".to_string(), "burn-tch".to_string()]);
            if std::env::var("DISABLE_WGPU").is_ok() {
                args.exclude.extend(vec!["burn-wgpu".to_string()]);
            };

            // test workspace
            test::handle_command(args.clone())?;

            // Specific additional commands to test specific features

            // burn-dataset
            helpers::custom_crates_tests(vec!["burn-dataset"], vec!["--all-features"])?;

            // burn-core
            helpers::custom_crates_tests(
                vec!["burn-core"],
                vec!["--features", "test-tch,record-item-custom-serde"],
            )?;

            if std::env::var("DISABLE_WGPU").is_err() {
                helpers::custom_crates_tests(vec!["burn-core"], vec!["--features", "test-wgpu"])?;
            }

            // MacOS specific tests
            #[cfg(target_os = "macos")]
            {
                // burn-candle
                helpers::custom_crates_tests(
                    vec!["burn-candle"],
                    vec!["--features", "accelerate"],
                )?;
                // burn-ndarray
                helpers::custom_crates_tests(
                    vec!["burn-ndarray"],
                    vec!["--features", "blas-accelerate"],
                )?;
            }
            Ok(())
        }
    }
}
