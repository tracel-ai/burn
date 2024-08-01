use xtask_common::{clap, utils::rustup::rustup_add_target, ExecutionEnvironment};

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

// no-std
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";
const NO_STD_CRATES: &[&str] = &[
    "burn",
    "burn-core",
    "burn-common",
    "burn-tensor",
    "burn-ndarray",
    "burn-no-std-tests",
];

#[xtask_macros::commands(
    Bump,
    Build,
    Check,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Test,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Run commands to manage Burn Books.
    Books(commands::books::BooksArgs),
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>()?;

    if args.execution_environment == ExecutionEnvironment::NoStd {
        // Install additional targets for no-std execution environments
        rustup_add_target(WASM32_TARGET)?;
        rustup_add_target(ARM_TARGET)?;
    }

    match args.command {
        Command::Books(cmd_args) => cmd_args.parse(),
        Command::Build(cmd_args) => build::handle_command(cmd_args),
        Command::Bump(cmd_args) => bump::handle_command(cmd_args),
        Command::Check(cmd_args) => check::handle_command(cmd_args),
        Command::Coverage(cmd_args) => coverage::handle_command(cmd_args),
        Command::Compile(cmd_args) => compile::handle_command(cmd_args),
        Command::Dependencies(cmd_args) => dependencies::handle_command(cmd_args),
        Command::Doc(cmd_args) => commands::doc::handle_command(cmd_args),
        Command::Fix(cmd_args) => fix::handle_command(cmd_args, None),
        Command::Publish(cmd_args) => publish::handle_command(cmd_args),
        Command::Test(cmd_args) => test::handle_command(cmd_args),
        Command::Validate => commands::validate::handle_command(),
        Command::Vulnerabilities(cmd_args) => vulnerabilities::handle_command(cmd_args),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
