use tracel_xtask::prelude::{clap::ValueEnum, *};

use crate::NO_STD_CRATES;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct BurnTestCmdArgs {
    /// Test in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: BurnTestCmdArgs,
    exec_env: ExecutionEnvironment,
) -> anyhow::Result<()> {
    match exec_env {
        ExecutionEnvironment::NoStd => {
            ["Default"].iter().try_for_each(|test_target| {
                let mut test_args = vec!["--no-default-features"];
                if *test_target != "Default" {
                    test_args.extend(vec!["--target", *test_target]);
                }
                helpers::custom_crates_tests(
                    NO_STD_CRATES.to_vec(),
                    test_args,
                    None,
                    None,
                    "no-std",
                )
            })?;
            Ok(())
        }
        ExecutionEnvironment::Std => {
            if args.ci {
                // Exclude crates that are not supported on CI
                args.exclude.extend(vec![
                    "burn-cuda".to_string(),
                    "burn-hip".to_string(),
                    "burn-tch".to_string(),
                ]);
            }
            let disable_wgpu = std::env::var("DISABLE_WGPU")
                .map(|val| val == "1" || val == "true")
                .unwrap_or(false);

            if disable_wgpu {
                args.exclude.extend(vec![
                    "burn-wgpu".to_string(),
                    // "burn-router" uses "burn-wgpu" for the tests.
                    "burn-router".to_string(),
                ]);
            };

            // test workspace
            base_commands::test::handle_command(args.try_into().unwrap())?;

            // Specific additional commands to test specific features

            // burn-dataset
            helpers::custom_crates_tests(
                vec!["burn-dataset"],
                vec!["--all-features"],
                None,
                None,
                "std all features",
            )?;

            // burn-core
            helpers::custom_crates_tests(
                vec!["burn-core"],
                vec!["--features", "test-tch,record-item-custom-serde"],
                None,
                None,
                "std with features: test-tch,record-item-custom-serde",
            )?;

            // burn-vision
            helpers::custom_crates_tests(
                vec!["burn-vision"],
                vec!["--features", "test-cpu"],
                None,
                None,
                "std cpu",
            )?;

            if !disable_wgpu {
                helpers::custom_crates_tests(
                    vec!["burn-core"],
                    vec!["--features", "test-wgpu"],
                    None,
                    None,
                    "std wgpu",
                )?;
                helpers::custom_crates_tests(
                    vec!["burn-vision"],
                    vec!["--features", "test-wgpu"],
                    None,
                    None,
                    "std wgpu",
                )?;

                // Vulkan isn't available on MacOS
                #[cfg(not(target_os = "macos"))]
                {
                    let disable_wgpu_spirv = std::env::var("DISABLE_WGPU_SPIRV")
                        .map(|val| val == "1" || val == "true")
                        .unwrap_or(false);

                    if !disable_wgpu_spirv {
                        helpers::custom_crates_tests(
                            vec!["burn-core"],
                            vec!["--features", "test-wgpu-spirv"],
                            None,
                            None,
                            "std vulkan",
                        )?;
                        helpers::custom_crates_tests(
                            vec!["burn-vision"],
                            vec!["--features", "test-vulkan"],
                            None,
                            None,
                            "std vulkan",
                        )?;
                    }
                }
            }

            // MacOS specific tests
            #[cfg(target_os = "macos")]
            {
                // burn-candle
                helpers::custom_crates_tests(
                    vec!["burn-candle"],
                    vec!["--features", "accelerate"],
                    None,
                    None,
                    "std accelerate",
                )?;
                // burn-ndarray
                helpers::custom_crates_tests(
                    vec!["burn-ndarray"],
                    vec!["--features", "blas-accelerate"],
                    None,
                    None,
                    "std blas-accelerate",
                )?;
            }
            Ok(())
        }
        ExecutionEnvironment::All => ExecutionEnvironment::value_variants()
            .iter()
            .filter(|env| **env != ExecutionEnvironment::All)
            .try_for_each(|env| {
                handle_command(
                    BurnTestCmdArgs {
                        command: args.command.clone(),
                        target: args.target.clone(),
                        exclude: args.exclude.clone(),
                        only: args.only.clone(),
                        threads: args.threads,
                        jobs: args.jobs,
                        ci: args.ci,
                        features: args.features.clone(),
                        no_default_features: args.no_default_features,
                    },
                    env.clone(),
                )
            }),
    }
}
