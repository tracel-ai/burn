use tracel_xtask::{
    prelude::{clap::ValueEnum, *},
    utils::{
        process::{ExitSignal, ProcessExitError, run_process},
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

#[derive(Debug, Clone, ValueEnum, PartialEq, strum::Display)]
pub(crate) enum TestBackend {
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

pub(crate) fn handle_backend_tests(
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

    let linalg_backend = format!("burn-linalg/{backend_name}");
    let signal_backend = format!("burn-signal/{backend_name}");
    let mut extension_packages = vec!["burn-linalg"];
    let mut extension_features = vec![linalg_backend.as_str()];
    if !matches!(context, Context::NoStd) {
        extension_features.extend(["burn-linalg/std", "burn-linalg/autotune"]);
    }
    // Signal has no NdArray implementation; keep its suite on supported backends.
    if !matches!(backend, TestBackend::Ndarray) {
        extension_packages.push("burn-signal");
        extension_features.extend([signal_backend.as_str(), "burn-signal/autodiff"]);
        if !matches!(context, Context::NoStd) {
            extension_features.extend(["burn-signal/std", "burn-signal/autotune"]);
        }
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

        build_helpers::custom_crates_tests(
            vec!["burn-backend-tests"],
            handle_test_args(&fusion_args, args.release),
            None,
            None,
            "fusion backend tests",
        )?;

        let mut extension_fusion_features = extension_features.clone();
        extension_fusion_features.extend(["burn-linalg/fusion", "burn-signal/fusion"]);
        run_test_group(
            &extension_packages,
            &extension_fusion_features,
            args.release,
            "linalg and signal fusion backend tests",
        )?;
    }

    let group_cpu_tests = matches!(backend, TestBackend::Ndarray | TestBackend::Flex)
        && matches!(context, Context::Std);
    if group_cpu_tests {
        // Keep each backend separate, and leave SIMD/threading defaults to the
        // standalone backend crate tests. The extension suites request autotuning.
        let mut packages = vec!["burn-backend-tests"];
        packages.extend_from_slice(&extension_packages);
        let backend_feature = format!("burn-backend-tests/{backend_name}");
        let mut features = extension_features.clone();
        features.extend([backend_feature.as_str(), "burn-backend-tests/std"]);
        run_test_group(
            &packages,
            &features,
            args.release,
            &format!("{backend_name} backend and extension tests"),
        )?;
    } else {
        build_helpers::custom_crates_tests(
            vec!["burn-backend-tests"],
            handle_test_args(&test_args, args.release),
            None,
            None,
            "backend tests",
        )?;
    }

    if matches!(backend, TestBackend::Flex) {
        // These targets each need a second backend. Keep them out of the main suite, where
        // ndarray disables some Flex-specific tests.
        let mut transfer_args = test_args.clone();
        transfer_args.extend(["--features", "ndarray", "--test", "autodiff_transfer"]);
        build_helpers::custom_crates_tests(
            vec!["burn-backend-tests"],
            handle_test_args(&transfer_args, args.release),
            None,
            None,
            "autodiff backend transfer tests",
        )?;

        let mut placement_args = test_args.clone();
        placement_args.extend([
            "--features",
            "ndarray",
            "--features",
            "autodiff",
            "--test",
            "lazy_param_device",
            "--test",
            "pipeline_placement",
        ]);
        build_helpers::custom_crates_tests(
            vec!["burn-core"],
            handle_test_args(&placement_args, args.release),
            None,
            None,
            "device placement tests",
        )?;
    }

    if !group_cpu_tests {
        run_test_group(
            &extension_packages,
            &extension_features,
            args.release,
            "extension backend tests",
        )?;
    }
    Ok(())
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

/// Compile compatible Metal suites together instead of rebuilding their shared GPU stack
/// for every package. Keep defaults out of the non-fusion group: the WGPU, core, and
/// vision defaults enable fusion transitively.
fn handle_macos_tests(release: bool) -> anyhow::Result<()> {
    set_burn_device("metal");

    let packages = ["burn-backend-tests", "burn-linalg", "burn-signal"];
    let features = [
        "burn-backend-tests/metal",
        "burn-backend-tests/std",
        "burn-linalg/metal",
        "burn-linalg/std",
        "burn-linalg/autotune",
        "burn-signal/metal",
        "burn-signal/std",
        "burn-signal/autodiff",
        "burn-signal/autotune",
    ];

    let mut fusion_packages = packages.to_vec();
    fusion_packages.extend(["burn-wgpu", "burn-core", "burn-vision"]);
    let mut fusion_features = features.to_vec();
    fusion_features.extend([
        "burn-backend-tests/fusion",
        "burn-linalg/fusion",
        "burn-signal/fusion",
        // Extension tests share this Metal/Fusion build and use BURN_DEVICE=metal.
        "burn-core/extension-tests",
        // Preserve the default-feature coverage of the former standalone crate tests.
        // Qualify every feature so adding a package cannot enable its namesake feature.
        "burn-wgpu/default",
        "burn-wgpu/metal",
        "burn-core/default",
        "burn-core/metal",
        "burn-vision/default",
        "burn-vision/metal",
    ]);
    run_test_group(
        &fusion_packages,
        &fusion_features,
        release,
        "Metal with fusion",
    )?;
    run_test_group(&packages, &features, release, "Metal without fusion")?;

    // Keep Accelerate separate so it cannot change the ndarray reference backend used
    // by the Metal tests. It also doesn't need to compile the GPU dependencies.
    build_helpers::custom_crates_tests(
        vec!["burn-ndarray"],
        handle_test_args(&["--features", "blas-accelerate"], release),
        None,
        None,
        "std blas-accelerate",
    )
}

fn run_test_group(
    packages: &[&str],
    features: &[&str],
    release: bool,
    description: &str,
) -> anyhow::Result<()> {
    // An empty discovered group must not turn into an implicit workspace test.
    if packages.is_empty() {
        return Ok(());
    }
    let features = features.join(",");
    let mut args = vec!["test", "--color", "always", "--no-default-features"];
    for package in packages {
        args.extend(["-p", package]);
    }
    args.extend(["--features", &features]);
    if release {
        args.push("--release");
    }

    // custom_crates_tests loops over packages and invokes Cargo once per package.
    // One invocation is required here for Cargo to unify their dependency features.
    group!("Tests: {}", description);
    let result = run_process("cargo", &args, None, None, description);
    endgroup!();
    result
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
    let metadata = cargo_metadata::MetadataCommand::new().no_deps().exec()?;

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

/// Discover opt-in feature coverage without maintaining another crate allowlist.
/// Apply the same platform/package exclusions as the workspace-default suite.
fn feature_test_group(
    metadata: &cargo_metadata::Metadata,
    feature: &str,
    excluded: &[String],
) -> (Vec<String>, Vec<String>) {
    let examples_dir = metadata.workspace_root.join("examples");
    let mut packages: Vec<_> = metadata
        .workspace_packages()
        .into_iter()
        .filter(|package| {
            !package.manifest_path.starts_with(&examples_dir)
                && !excluded.iter().any(|name| name == package.name.as_str())
                && package.features.contains_key(feature)
        })
        .collect();
    packages.sort_by(|a, b| a.name.cmp(&b.name));

    let mut features = Vec::new();
    for package in &packages {
        features.push(format!("{}/{feature}", package.name));
        // run_test_group disables defaults globally; restore each package's
        // defaults explicitly when that feature exists.
        if package.features.contains_key("default") {
            features.push(format!("{}/default", package.name));
        }
        // Execution tests need an explicit backend, including capture/replay coverage.
        if feature != "flex" && package.features.contains_key("flex") {
            features.push(format!("{}/flex", package.name));
        }
    }
    (
        packages
            .iter()
            .map(|package| package.name.to_string())
            .collect(),
        features,
    )
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
                build_helpers::custom_crates_tests(
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
                    args.only.push("burn-ndarray".to_string());
                    base_commands::test::handle_command(
                        args.clone().try_into().unwrap(),
                        env.clone(),
                        context,
                    )?;

                    // Native FFT kernels are opt-in, but keep their backend unit tests covered.
                    build_helpers::custom_crates_tests(
                        vec!["burn-flex"],
                        handle_test_args(&["--features", "fft"], args.release),
                        None,
                        None,
                        "Flex backend with FFT kernels",
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

                    // Select the execution backend explicitly now that defaults are backend-free.
                    let metadata = cargo_metadata::MetadataCommand::new().no_deps().exec()?;
                    let (_, features) = feature_test_group(&metadata, "flex", &args.exclude);
                    args.features.get_or_insert_with(Vec::new).extend(features);
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
                    handle_macos_tests(args.release)?;
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

                    let args_wgpu = args_wgpu.try_into().unwrap();
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
                    // Capture is intentionally opt-in, so workspace-default tests don't compile
                    // the dispatch, tensor, core, or facade integration tests that exercise it.
                    if !args.exclude.iter().any(|name| name == "burn") {
                        super::validate::check_backend_features()?;
                    }
                    let metadata = cargo_metadata::MetadataCommand::new().no_deps().exec()?;
                    let (packages, features) =
                        feature_test_group(&metadata, "capture", &args.exclude);
                    run_test_group(
                        &packages.iter().map(String::as_str).collect::<Vec<_>>(),
                        &features.iter().map(String::as_str).collect::<Vec<_>>(),
                        args.release,
                        "std with graph capture",
                    )?;

                    // The relocated websocket FFT test requires explicit server features.
                    #[cfg(target_os = "linux")]
                    if !args.exclude.iter().any(|name| name == "burn-signal") {
                        for features in ["flex,std,remote-tests", "flex,std,remote-tests,fusion"] {
                            build_helpers::custom_crates_tests(
                                vec!["burn-signal"],
                                handle_test_args(
                                    &[
                                        "--no-default-features",
                                        "--features",
                                        features,
                                        "--test",
                                        "remote",
                                    ],
                                    args.release,
                                ),
                                None,
                                None,
                                "signal FFT over websocket",
                            )?;
                        }
                    }

                    // Keep all-features coverage in this shard to avoid consuming
                    // another runner from the organization's concurrency limit.
                    build_helpers::custom_crates_tests(
                        vec!["burn-dataset"],
                        handle_test_args(&["--all-features"], args.release),
                        None,
                        None,
                        "std dataset all features",
                    )?;

                    // burn-core
                    set_burn_device("tch"); // test-tch
                    build_helpers::custom_crates_tests(
                        vec!["burn-core"],
                        handle_test_args(&["--features", "tch"], args.release),
                        None,
                        None,
                        "std with features: tch",
                    )?;

                    // Both suites use Flex; share their training and model dependencies.
                    set_burn_device("flex");
                    run_test_group(
                        &["burn-vision", "burn-train"],
                        &[
                            "burn-vision/default",
                            "burn-vision/flex",
                            "burn-vision/loss",
                            "burn-train/default",
                            "burn-train/vision",
                        ],
                        args.release,
                        "std vision and training (flex)",
                    )?;
                }
                CiTestType::GcpCudaRunner | CiTestType::GithubMacRunner => (),
                CiTestType::GcpVulkanRunner | CiTestType::GcpWgpuRunner => (), // handled in tests above
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
