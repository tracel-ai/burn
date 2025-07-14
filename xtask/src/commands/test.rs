use tracel_xtask::prelude::{clap::ValueEnum, *};

use crate::NO_STD_CRATES;

#[macros::extend_command_args(TestCmdArgs, Target, TestSubCommand)]
pub struct BurnTestCmdArgs {
    /// Test in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: CiTestType,
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum CiTestType {
    GithubRunner,
    GithubMacRunner,
    GcpCudaRunner,
    GcpVulkanRunner,
    GcpWgpuRunner,
}

pub(crate) fn handle_command(
    mut args: BurnTestCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    match context {
        Context::NoStd => {
            ["Default"].iter().try_for_each(|test_target| {
                let mut test_args = vec!["--no-default-features"];
                if *test_target != "Default" {
                    test_args.extend(vec!["--target", *test_target]);
                }
                helpers::custom_crates_tests(
                    NO_STD_CRATES.to_vec(),
                    handle_test_args(&test_args, args.release),
                    None,
                    None,
                    "no-std",
                )
            })?;
            Ok(())
        }
        Context::Std => {
            // 1) Tests with default features
            // ------------------------------
            match args.ci {
                CiTestType::GithubRunner => {
                    // Exclude crates that are not supported on CI
                    args.exclude.extend(vec![
                        "burn-cuda".to_string(),
                        "burn-rocm".to_string(),
                        // "burn-router" uses "burn-wgpu" for the tests.
                        "burn-router".to_string(),
                        "burn-tch".to_string(),
                        "burn-wgpu".to_string(),
                    ]);

                    // Burn remote tests don't work on windows for now
                    #[cfg(target_os = "windows")]
                    {
                        args.exclude.extend(vec!["burn-remote".to_string()]);
                    };
                }
                CiTestType::GithubMacRunner => {
                    args.target = Target::AllPackages;
                    args.only.push("burn-wgpu".to_string());
                    args.features
                        .get_or_insert_with(Vec::new)
                        .push("metal".to_string());
                }
                CiTestType::GcpCudaRunner => {
                    args.target = Target::AllPackages;
                    args.only.push("burn-cuda".to_string());
                }
                CiTestType::GcpVulkanRunner => {
                    args.target = Target::AllPackages;
                    args.only.push("burn-wgpu".to_string());
                    args.features
                        .get_or_insert_with(Vec::new)
                        .push("vulkan".to_string());
                }
                CiTestType::GcpWgpuRunner => {
                    args.target = Target::AllPackages;
                    // "burn-router" uses "burn-wgpu" for the tests.
                    args.only
                        .extend(vec!["burn-wgpu".to_string(), "burn-router".to_string()]);
                }
            }

            // test workspace
            base_commands::test::handle_command(args.clone().try_into().unwrap(), env, context)?;

            // 2) Specific additional commands to test specific features
            // ---------------------------------------------------------
            match args.ci {
                CiTestType::GithubRunner => {
                    // burn-dataset
                    helpers::custom_crates_tests(
                        vec!["burn-dataset"],
                        handle_test_args(&["--all-features"], args.release),
                        None,
                        None,
                        "std all features",
                    )?;

                    // burn-core
                    helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(
                            &["--features", "test-tch,record-item-custom-serde"],
                            args.release,
                        ),
                        None,
                        None,
                        "std with features: test-tch,record-item-custom-serde",
                    )?;

                    // burn-vision
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "test-cpu"], args.release),
                        None,
                        None,
                        "std cpu",
                    )?;
                }
                CiTestType::GcpCudaRunner => (),
                CiTestType::GcpVulkanRunner => {
                    helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(&["--features", "test-vulkan"], args.release),
                        None,
                        None,
                        "std vulkan",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "test-vulkan"], args.release),
                        None,
                        None,
                        "std vulkan",
                    )?;
                }
                CiTestType::GcpWgpuRunner => {
                    helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(&["--features", "test-wgpu"], args.release),
                        None,
                        None,
                        "std wgpu",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "test-wgpu"], args.release),
                        None,
                        None,
                        "std wgpu",
                    )?;
                }
                CiTestType::GithubMacRunner => {
                    // burn-candle
                    helpers::custom_crates_tests(
                        vec!["burn-candle"],
                        handle_test_args(&["--features", "accelerate"], args.release),
                        None,
                        None,
                        "std accelerate",
                    )?;
                    // burn-ndarray
                    helpers::custom_crates_tests(
                        vec!["burn-ndarray"],
                        handle_test_args(&["--features", "blas-accelerate"], args.release),
                        None,
                        None,
                        "std blas-accelerate",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(&["--features", "test-metal"], args.release),
                        None,
                        None,
                        "std metal",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "test-metal"], args.release),
                        None,
                        None,
                        "std metal",
                    )?;
                }
            }
            Ok(())
        }
        Context::All => Context::value_variants()
            .iter()
            .filter(|ctx| **ctx != Context::All)
            .try_for_each(|ctx| {
                handle_command(
                    BurnTestCmdArgs {
                        command: args.command.clone(),
                        target: args.target.clone(),
                        exclude: args.exclude.clone(),
                        only: args.only.clone(),
                        threads: args.threads,
                        jobs: args.jobs,
                        ci: args.ci.clone(),
                        features: args.features.clone(),
                        no_default_features: args.no_default_features,
                        release: args.release,
                        test: args.test.clone(),
                        force: args.force,
                        no_capture: args.no_capture,
                    },
                    env.clone(),
                    ctx.clone(),
                )
            }),
    }
}

fn handle_test_args<'a>(args: &'a [&'a str], release: bool) -> Vec<&'a str> {
    let mut args = args.to_vec();
    if release {
        args.push("--release");
    }
    args
}
