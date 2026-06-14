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

// `cargo check` for examples
impl std::convert::TryInto<CompileCmdArgs> for BurnTestCmdArgs {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<CompileCmdArgs, Self::Error> {
        Ok(CompileCmdArgs {
            target: self.target,
            exclude: self.exclude,
            only: self.only,
        })
    }
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, ValueEnum, PartialEq)]
pub enum CiTestType {
    // Github runner shards
    Backends,
    Crates,
    Examples,
    // Other runners
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

fn set_burn_device(device: &str) {
    // SAFETY: This is called in a single-threaded context within the xtask before spawning child processes.
    unsafe {
        std::env::set_var("BURN_DEVICE", device);
    }
}

fn handle_backend_tests(
    args: TestCmdArgs,
    backend: TestBackend,
    context: Context,
) -> anyhow::Result<()> {
    let backend_name = backend.to_string();
    set_burn_device(&backend_name); // default device

    let mut test_args = vec!["--no-default-features", "--features", &backend_name];
    if !matches!(context, Context::NoStd) {
        test_args.extend(["--features", "std"])
    }

    if matches!(backend, TestBackend::Cuda) {
        // Collective (all-reduce) tests require a CUDA build with NCCL, which the CI runner
        // provides. Kept behind its own feature so plain `--features cuda` still works without it.
        test_args.extend(["--features", "distributed"]);
    }

    if !matches!(backend, TestBackend::Ndarray | TestBackend::Flex) {
        // Fusion enabled tests first
        let mut fusion_args = test_args.clone();
        fusion_args.extend(["--features", "fusion"]);

        helpers::custom_crates_tests(
            vec!["burn-backend-tests"],
            handle_test_args(&fusion_args, args.release),
            None,
            None,
            "fusion backend tests",
        )?;
        // base_commands::test::handle_command(fusion_args, env.clone(), context.clone())?;
    }

    // base_commands::test::handle_command(args, env, context)
    helpers::custom_crates_tests(
        vec!["burn-backend-tests"],
        handle_test_args(&test_args, args.release),
        None,
        None,
        "backend tests",
    )
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

const EXCLUDE_CRATES: &[&str] = &[
    "burn-cpu",
    "burn-cuda",
    "burn-rocm",
    // "burn-router" uses "burn-wgpu" for the tests.
    "burn-router",
    "burn-tch",
    "burn-wgpu",
    // Requires wgpu runtime
    "burn-cubecl-fusion",
    // Backends are tested individually
    "burn-backend-tests",
    "burn-ndarray",
    "burn-flex",
];

fn enumerate_examples() -> anyhow::Result<Vec<String>> {
    let metadata = cargo_metadata::MetadataCommand::new().exec()?;

    let workspace_root = metadata.workspace_root.as_std_path();
    let examples_dir = workspace_root.join("examples");

    Ok(metadata
        .workspace_packages()
        .into_iter()
        .filter(|package| {
            // Check if the package's Cargo.toml lives inside the examples/ folder
            package.manifest_path.starts_with(&examples_dir)
        })
        .map(|package| package.name.to_string())
        .collect())
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
                context,
            )?;

            Ok(())
        }
        Context::Std => {
            // 1) Tests with default features
            // ------------------------------
            match args.ci {
                CiTestType::Backends | CiTestType::GithubRunner => {
                    // Backend ops
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Ndarray,
                        context.clone(),
                    )?;

                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Flex,
                        context.clone(),
                    )?;

                    // Backend crates
                    args.target = Target::AllPackages;
                    args.only
                        .extend(["burn-ndarray".to_string(), "burn-flex".to_string()]);
                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env.clone(),
                        context,
                    )?;
                }
                CiTestType::Crates => {
                    // Default `Target::Workspace`
                    // Exclude crates that are not supported on CI
                    args.exclude
                        .extend(EXCLUDE_CRATES.iter().map(|&s| s.to_string()));
                    // Exclude examples
                    // workspace feature unification will cause binary bloat with examples default features
                    args.exclude.extend(enumerate_examples()?);

                    // Burn remote tests don't work on windows for now
                    #[cfg(target_os = "windows")]
                    {
                        args.exclude.extend(vec!["burn-remote".to_string()]);
                    };

                    set_burn_device("flex"); // default device for base tests
                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env.clone(),
                        context.clone(),
                    )?;
                }
                CiTestType::Examples => {
                    // NOTE: for the examples we simply run `cargo checks` (no tests, faster validation)
                    // TODO: switch to `cargo xtask build` or `check` eventually instead of including this in the tests
                    args.target = Target::AllPackages;
                    args.only.extend(enumerate_examples()?);
                    base_commands::compile::handle_command(
                        args.clone().try_into().unwrap(),
                        env.clone(),
                        context.clone(),
                    )?;
                }
                CiTestType::GithubMacRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Metal,
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
                        context,
                    )?;
                }
                CiTestType::GcpVulkanRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Vulkan,
                        context,
                    )?;

                    args.target = Target::AllPackages;
                    let mut args_vulkan = args.clone();
                    args_vulkan
                        .features
                        .get_or_insert_with(Vec::new)
                        .push("vulkan".to_string());

                    let args_vulkan = args_vulkan.try_into().unwrap();
                    handle_wgpu_test("burn-wgpu", &args_vulkan)?;
                    handle_wgpu_test("burn-core", &args_vulkan)?;
                    handle_wgpu_test("burn-vision", &args_vulkan)?;

                    // Enable burn-core/vulkan
                    args.features
                        .get_or_insert_with(Vec::new)
                        .push("burn-core/vulkan".to_string());
                    let args_vulkan = args.clone().try_into().unwrap();
                    handle_wgpu_test("burn-optim", &args_vulkan)?;
                    handle_wgpu_test("burn-nn", &args_vulkan)?;
                }
                CiTestType::GcpWgpuRunner => {
                    handle_backend_tests(
                        args.clone().try_into().unwrap(),
                        TestBackend::Wgpu,
                        context,
                    )?;
                    args.target = Target::AllPackages;
                    handle_wgpu_test("burn-cubecl-fusion", &args.clone().try_into().unwrap())?;

                    let mut args_wgpu = args.clone();
                    args_wgpu
                        .features
                        .get_or_insert_with(Vec::new)
                        .push("webgpu".to_string());

                    let args_wgpu = args.clone().try_into().unwrap();
                    handle_wgpu_test("burn-wgpu", &args_wgpu)?;
                    handle_wgpu_test("burn-core", &args_wgpu)?;
                    handle_wgpu_test("burn-vision", &args_wgpu)?;

                    // Enable burn-core/webgpu
                    args.features
                        .get_or_insert_with(Vec::new)
                        .push("burn-core/webgpu".to_string());
                    let args_wgpu = args.clone().try_into().unwrap();
                    handle_wgpu_test("burn-optim", &args_wgpu)?;
                    handle_wgpu_test("burn-nn", &args_wgpu)?;
                }
            }

            // 2) Specific additional commands to test specific features
            // ---------------------------------------------------------
            match args.ci {
                CiTestType::Backends | CiTestType::GithubRunner => (),
                CiTestType::Examples => (),
                CiTestType::Crates => {
                    // burn-dataset
                    helpers::custom_crates_tests(
                        vec!["burn-dataset"],
                        handle_test_args(&["--all-features"], args.release),
                        None,
                        None,
                        "std all features",
                    )?;

                    // burn-core
                    set_burn_device("tch"); // test-tch
                    helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(
                            &["--features", "tch,record-item-custom-serde"],
                            args.release,
                        ),
                        None,
                        None,
                        "std with features: tch,record-item-custom-serde",
                    )?;

                    // burn-nn (pretrained and local tests)
                    // If the "CI" environment variable is missing, we are running locally.
                    // if std::env::var("CI").is_err() {
                    //     nn_features.push_str(",test-local");
                    // }
                    // burn-vision
                    set_burn_device("flex");
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "flex", "loss"], args.release),
                        None,
                        None,
                        "std cpu (flex)",
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

                    set_burn_device("metal");
                    helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(&["--features", "metal"], args.release),
                        None,
                        None,
                        "std metal",
                    )?;
                    helpers::custom_crates_tests(
                        vec!["burn-vision"],
                        handle_test_args(&["--features", "metal"], args.release),
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
