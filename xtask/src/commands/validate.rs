use tracel_xtask::prelude::*;
use tracel_xtask::utils::process::run_process;

use crate::NO_STD_CRATES;
use crate::commands::test::TestBackend;

#[derive(clap::Args)]
pub struct BurnValidateCmdArgs {
    /// Backend used by burn-backend-tests.
    #[arg(long, value_enum, default_value_t = TestBackend::Flex)]
    backend: TestBackend,
}

pub fn handle_command(
    args: BurnValidateCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    let target = Target::Workspace;

    // Keep the cheapest checks first so local validation fails quickly.
    [
        CheckSubCommand::Format,
        CheckSubCommand::Typos,
        CheckSubCommand::Audit,
        CheckSubCommand::Lint,
    ]
    .iter()
    .try_for_each(|command| {
        base_commands::check::handle_command(
            CheckCmdArgs {
                target: target.clone(),
                exclude: vec![],
                only: vec![],
                command: Some(command.clone()),
                ignore_audit: false,
                features: vec![],
                no_default_features: false,
                ignore_typos: false,
            },
            env.clone(),
            context.clone(),
        )
    })?;

    check_no_std()?;
    check_backend_features()?;

    super::test::handle_backend_tests(
        TestCmdArgs {
            target,
            exclude: vec![],
            only: vec![],
            threads: None,
            jobs: None,
            command: Some(TestSubCommand::All),
            features: None,
            no_default_features: false,
            force: false,
            no_capture: false,
            release: true,
            test: None,
            miri: None,
        },
        args.backend,
        Context::Std,
    )?;

    Ok(())
}

/// Check no-std compatibility on the host without compiling the full embedded target matrix.
fn check_no_std() -> anyhow::Result<()> {
    // Keep this separate: the integration-test crate selects Flex in the group below.
    run_process(
        "cargo",
        &["check", "-p", "burn", "--no-default-features"],
        None,
        None,
        "Backend-free no-std check failed",
    )?;

    let mut args = vec!["check", "--no-default-features", "--color", "always"];
    for package in NO_STD_CRATES {
        args.extend(["-p", package]);
    }

    run_process("cargo", &args, None, None, "Quick no-std check failed")
}

/// Resolve Burn independently so workspace feature unification cannot supply a backend.
pub(crate) fn check_backend_features() -> anyhow::Result<()> {
    run_process(
        "cargo",
        &["check", "-p", "burn"],
        None,
        None,
        "Backend-free default build failed",
    )?;

    for (features, forbidden) in [
        (
            "signal,linalg,vision",
            &[
                "burn-flex",
                "burn-cubecl",
                "burn-ndarray",
                "burn-tch",
                "burn-remote",
                "burn-capture",
            ][..],
        ),
        ("wgpu,signal,linalg,vision", &["burn-flex"][..]),
    ] {
        // Dev-dependencies intentionally select backends for execution tests.
        let output = std::process::Command::new("cargo")
            .args([
                "tree",
                "-p",
                "burn",
                "--features",
                features,
                "--edges",
                "normal,build",
                "--prefix",
                "none",
            ])
            .output()?;
        anyhow::ensure!(
            output.status.success(),
            "Dependency check failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        for package in String::from_utf8(output.stdout)?
            .lines()
            .filter_map(|line| line.split_whitespace().next())
        {
            anyhow::ensure!(
                !forbidden.contains(&package),
                "Features {features} unexpectedly enabled {package}"
            );
        }
    }
    Ok(())
}
