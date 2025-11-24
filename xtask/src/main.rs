mod commands;

#[macro_use]
extern crate log;

use std::time::Instant;
use tracel_xtask::prelude::*;

// no-std
const WASM32_TARGET: &str = "wasm32-unknown-unknown";
const ARM_TARGET: &str = "thumbv7m-none-eabi";
const ARM_NO_ATOMIC_PTR_TARGET: &str = "thumbv6m-none-eabi";
const NO_STD_CRATES: &[&str] = &[
    "burn",
    "burn-autodiff",
    "burn-core",
    "burn-common",
    "burn-shape",
    "burn-backend",
    "burn-tensor",
    "burn-ndarray",
    "burn-no-std-tests",
];

#[macros::base_commands(
    Bump,
    Check,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Run commands to manage Burn Books.
    Books(commands::books::BooksArgs),
    /// Build Burn in different modes.
    Build(commands::build::BurnBuildCmdArgs),
    /// Test Burn.
    Test(commands::test::BurnTestCmdArgs),
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let args = init_xtask::<Command>(parse_args::<Command>()?)?;

    if args.context == Context::NoStd {
        // Install additional targets for no-std execution environments
        rustup_add_target(WASM32_TARGET)?;
        rustup_add_target(ARM_TARGET)?;
        rustup_add_target(ARM_NO_ATOMIC_PTR_TARGET)?;
    }

    match args.command {
        Command::Books(cmd_args) => cmd_args.parse(),
        Command::Build(cmd_args) => {
            commands::build::handle_command(cmd_args, args.environment, args.context)
        }
        Command::Doc(cmd_args) => {
            commands::doc::handle_command(cmd_args, args.environment, args.context)
        }
        Command::Test(cmd_args) => {
            commands::test::handle_command(cmd_args, args.environment, args.context)
        }
        Command::Validate(cmd_args) => {
            commands::validate::handle_command(&cmd_args, args.environment, args.context)
        }
        _ => dispatch_base_commands(args),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
