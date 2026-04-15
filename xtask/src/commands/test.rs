use tracel_xtask::{
    prelude::{clap::ValueEnum, *},
    utils::{
        process::{ExitSignal, ProcessExitError},
        workspace::WorkspaceMember,
    },
};

use crate::NO_STD_CRATES;

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;

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

#[derive(strum::Display)]
enum TestBackend {
    #[strum(to_string = "cuda")]
    Cuda,
    #[strum(to_string = "metal")]
    Metal,
    #[strum(to_string = "vulkan")]
    Vulkan,
    #[strum(to_string = "wgpu")]
    Wgpu,
    #[allow(unused)]
    #[strum(to_string = "rocm")]
    Rocm,
    #[strum(to_string = "flex")]
    Flex,
    #[strum(to_string = "ndarray")]
    Ndarray,
}
fn handle_backend_tests(
    mut args: TestCmdArgs,
    backend: TestBackend,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    args.target = Target::AllPackages;
    args.only.push("burn-backend-tests".to_string());
    args.no_default_features = true;

    let mut features = vec![backend.to_string()];
    if !matches!(context, Context::NoStd) {
        features.push("std".into())
    }
    args.features = Some(features);

    if !matches!(backend, TestBackend::Ndarray | TestBackend::Flex) {
        // Fusion enabled tests first
        let mut fusion_args = args.clone();
        if let Some(features) = fusion_args.features.as_mut() {
            features.push("fusion".into());
        }

        base_commands::test::handle_command(fusion_args, env.clone(), context.clone())?;
    }

    base_commands::test::handle_command(args, env, context)
}

fn handle_wgpu_test(member: &str, args: &TestCmdArgs) -> anyhow::Result<()> {
    #[cfg(unix)]
    let filter_err = |e: &&ProcessExitError| {
        e.status.signal() == Some(11) || matches!(e.signal, Some(ExitSignal { code: 11, .. }))
    };
    #[cfg(not(unix))]
    let filter_err = |e: &&ProcessExitError| matches!(e.signal, Some(ExitSignal { code: 11, .. }));

    let workspace_member = WorkspaceMember {
        name: member.into(),
        path: "".into(), // unused
    };

    if let Err(err) = base_commands::test::run_unit_test(&workspace_member, args) {
        let should_ignore = err
            .downcast_ref::<ProcessExitError>()
            .filter(filter_err)
            // Failed to execute unit test for '{member}'
            .map(|e| e.message.contains(member))
            .unwrap_or(false);

        if should_ignore {
            // Ignore intermittent successful failures
            // https://github.com/gfx-rs/wgpu/issues/2949
            // https://github.com/KhronosGroup/Vulkan-ValidationLayers/issues/4391
            eprintln!("⚠️ Ignored SIGSEGV in wgpu test");
        } else {
            return Err(err);
        }
    }
    Ok(())
}

pub(crate) fn handle_command(
    mut args: BurnTestCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    match context {
        Context::NoStd => {
            // burn-flex's unit tests use `std::f32::consts` and bare `vec!`
            // macros directly in test modules, so they only compile under std.
            // The build step (`xtask build --no-std`) still validates that
            // the crate itself compiles as no_std via `cargo build`, which
            // does not pull in test modules.
            let no_std_test_crates: Vec<&str> = NO_STD_CRATES
                .iter()
                .copied()
                .filter(|&c| c != "burn-flex")
                .collect();
            ["Default"].iter().try_for_each(|test_target| {
                let mut test_args = vec!["--no-default-features"];
                if *test_target != "Default" {
                    test_args.extend(vec!["--target", *test_target]);
                }
                helpers::custom_crates_tests(
                    no_std_test_crates.clone(),
                    handle_test_args(&test_args, args.release),
                    None,
                    None,
                    "no-std",
                )
            })?;
            handle_backend_tests(
                args.clone().try_into().unwrap(),
                TestBackend::Ndarray,
                env,
                context,
            )?;

            Ok(())
        }
        Context::Std => {
            // 1) Tests with default features
            // ------------------------------
            match args.ci {
                CiTestType::GithubRunner => {
                    // Exclude crates that are not supported on CI
                    args.exclude.extend(vec![
                        "burn-cpu".to_string(),
                        "burn-cuda".to_string(),
                        "burn-rocm".to_string(),
                        // "burn-router" uses "burn-wgpu" for the tests.
                        "burn-router".to_string(),
                        "burn-tch".to_string(),
                        "burn-wgpu".to_string(),
                        // dqn-agent example relies on gym-rs dependency which requires SDL2.
                        // It would be good to remove the gym-rs dependency in the future.
                        "dqn-agent".to_string(),
                        // Requires wgpu runtime
                        "burn-cubecl-fusion".to_string(),
                    ]);

                    // Burn remote tests don't work on windows for now
                    #[cfg(target_os = "windows")]
                    {
                        args.exclude.extend(vec!["burn-remote".to_string()]);
                    };

                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env.clone(),
                        context.clone(),
                    )?;

                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Ndarray,
                        env.clone(),
                        context.clone(),
                    )?;

                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Flex,
                        env,
                        context,
                    )?;
                }
                CiTestType::GithubMacRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Metal,
                        env.clone(),
                        context.clone(),
                    )?;

                    args.target = Target::AllPackages;
                    args.only.push("burn-wgpu".to_string());
                    args.features
                        .get_or_insert_with(Vec::new)
                        .push("metal".to_string());

                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env,
                        context,
                    )?;
                }
                CiTestType::GcpCudaRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Cuda,
                        env,
                        context,
                    )?;
                }
                CiTestType::GcpVulkanRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Vulkan,
                        env,
                        context,
                    )?;

                    args.target = Target::AllPackages;
                    let mut args_vulkan: TestCmdArgs = args.clone().try_into().unwrap();
                    args_vulkan.features = Some(vec!["test-vulkan".into()]);
                    handle_wgpu_test("burn-core", &args_vulkan)?;
                    handle_wgpu_test("burn-optim", &args_vulkan)?;
                    handle_wgpu_test("burn-nn", &args_vulkan)?;
                    handle_wgpu_test("burn-vision", &args_vulkan)?;
                }
                CiTestType::GcpWgpuRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Wgpu,
                        env,
                        context,
                    )?;
                    // "burn-router" uses "burn-wgpu" for the tests.
                    args.target = Target::AllPackages;
                    let mut args_wgpu = args.clone().try_into().unwrap();
                    handle_wgpu_test("burn-wgpu", &args_wgpu)?;
                    handle_wgpu_test("burn-router", &args_wgpu)?;
                    handle_wgpu_test("burn-cubecl-fusion", &args_wgpu)?;

                    args_wgpu.features = Some(vec!["test-wgpu".into()]);
                    handle_wgpu_test("burn-core", &args_wgpu)?;
                    handle_wgpu_test("burn-optim", &args_wgpu)?;
                    handle_wgpu_test("burn-nn", &args_wgpu)?;
                    handle_wgpu_test("burn-vision", &args_wgpu)?;
                }
            }

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

                    // burn-nn (pretrained and local tests)
                    // If the "CI" environment variable is missing, we are running locally.
                    // if std::env::var("CI").is_err() {
                    //     nn_features.push_str(",test-local");
                    // }
                    // burn-vision
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "test-cpu", "loss"], args.release),
                        None,
                        None,
                        "std cpu",
                    )?;

                    // burn-train vision (LPIPS, DISTS metrics)
                    helpers::custom_crates_tests(
                        vec!["burn-train"],
                        handle_test_args(&["--features", "vision"], args.release),
                        None,
                        None,
                        "std vision",
                    )?;
                }
                CiTestType::GcpCudaRunner => (),
                CiTestType::GcpVulkanRunner | CiTestType::GcpWgpuRunner => (), // handled in tests above
                CiTestType::GithubMacRunner => {
                    // burn-ndarray
                    helpers::custom_crates_tests(
                        vec!["burn-ndarray"],
                        handle_test_args(&["--features", "blas-accelerate"], args.release),
                        None,
                        None,
                        "std blas-accelerate",
                    )?;

                    // burn-train vision (LPIPS, DISTS metrics)
                    helpers::custom_crates_tests(
                        vec!["burn-train"],
                        handle_test_args(&["--features", "vision"], args.release),
                        None,
                        None,
                        "std vision",
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
                        miri: args.miri,
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
