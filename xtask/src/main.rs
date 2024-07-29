use xtask_common::{
    clap,
    utils::{helpers, rustup::rustup_add_target},
    ExecutionEnvironment,
};

mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
use xtask_common::{
    anyhow::{self, Ok},
    commands::*,
    init_xtask,
    utils::time::format_duration,
};

// no-std additional build targets
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct XtaskArgs {
    #[command(subcommand)]
    command: Command,
}

#[xtask_macros::commands(Bump, Check, CI, Coverage, Doc, Dependencies, Publish, Vulnerabilities)]
pub enum Command {
    /// Run commands to manage Burn Books
    Books(commands::books::BooksArgs),
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();

    let args = init_xtask::<Command>()?;
    match args.command {
        Command::Books(cmd_args) => cmd_args.parse(),
        // Commands from common_xtask
        Command::Bump(cmd_args) => bump::handle_command(cmd_args),
        Command::Check(cmd_args) => check::handle_command(cmd_args, None),
        Command::CI(mut cmd_args) => {
            match args.execution_environment {
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
                        match cmd_args.command {
                            ci::CICommand::Build => {
                                helpers::additional_crates_build(no_std_crates.clone(), build_args)
                            },
                            _ => Ok(()),
                        }
                    })?;
                    let no_std_test_targets = ["Default"];
                    no_std_test_targets.iter().try_for_each(|test_target| {
                        let mut test_args = no_std_features_args.clone();
                        if *test_target != "Default" {
                            test_args.extend(vec!["--target", *test_target]);
                        }
                        match cmd_args.command {
                            ci::CICommand::UnitTests => helpers::additional_crates_unit_tests(
                                no_std_crates.clone(),
                                test_args,
                            ),
                            _ => Ok(()),
                        }
                    })
                }
                ExecutionEnvironment::Std => {
                    match cmd_args.command {
                        ci::CICommand::Build => {
                            // Exclude problematic crates from build
                            cmd_args
                                .exclude
                                .extend(vec!["burn-cuda".to_string(), "burn-tch".to_string()]);
                            // burn-dataset
                            helpers::additional_crates_build(
                                vec!["burn-dataset"],
                                vec!["--all-features"],
                            )?;
                        }
                        ci::CICommand::UnitTests => {
                            // Exclude problematic crates from tests
                            cmd_args
                                .exclude
                                .extend(vec!["burn-cuda".to_string(), "burn-tch".to_string()]);
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
                        ci::CICommand::DocTests => {
                            // TODO cargo_doc(["-p", "burn-dataset", "--all-features", "--no-deps"].into());
                            // Exclude problematic crates from documentation test
                            cmd_args.exclude.extend(vec!["burn-cuda".to_string()])
                        }
                        _ => {}
                    }
                    ci::handle_command(cmd_args.clone())
                }
            }
        }
        Command::Coverage(cmd_args) => coverage::handle_command(cmd_args),
        Command::Dependencies(cmd_args) => dependencies::handle_command(cmd_args),
        Command::Doc(mut cmd_args) => {
            match cmd_args.command {
                doc::DocCommand::Build => {
                    cmd_args.exclude.push("burn-cuda".to_string());
                    // burn-dataset
                    helpers::additional_crates_doc_build(
                        vec!["burn-dataset"],
                        vec!["--all-features"],
                    )?;
                }
            }
            doc::handle_command(cmd_args)
        }
        Command::Publish(cmd_args) => publish::handle_command(cmd_args),
        Command::Vulnerabilities(cmd_args) => vulnerabilities::handle_command(cmd_args),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
