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
    "burn-std",
    "burn-backend",
    "burn-capture",
    "burn-tensor",
    "burn-ndarray",
    "burn-no-std-tests",
];

#[derive(clap::Subcommand, strum::Display)]
pub enum Command {
    Bump(BumpCmdArgs),
    Check(CheckCmdArgs),
    Compile(CompileCmdArgs),
    Coverage(CoverageCmdArgs),
    Dependencies(DependenciesCmdArgs),
    Doc(DocCmdArgs),
    Fix(FixCmdArgs),
    Publish(PublishCmdArgs),
    Vulnerabilities(VulnerabilitiesCmdArgs),
    /// Run commands to manage Burn Books.
    Books(commands::books::BooksArgs),
    /// Build Burn in different modes.
    Build(commands::build::BurnBuildCmdArgs),
    /// Validate the remote backend end-to-end: spin up the `server` example, point
    /// `burn-backend-tests` at it via `BURN_DEVICE=remote`, tear it down on exit.
    Remote(commands::remote::RemoteCmdArgs),
    /// Test Burn.
    Test(commands::test::BurnTestCmdArgs),
    /// Run the fast checks expected before opening a pull request.
    Validate(commands::validate::BurnValidateCmdArgs),
}

fn dispatch_base_commands(args: XtaskArgs<Command>, env: Environment) -> anyhow::Result<()> {
    match args.command {
        Command::Bump(cmd) => base_commands::bump::handle_command(cmd, env, args.context),
        Command::Check(cmd) => base_commands::check::handle_command(cmd, env, args.context),
        Command::Compile(cmd) => base_commands::compile::handle_command(cmd, env, args.context),
        Command::Coverage(cmd) => base_commands::coverage::handle_command(cmd, env, args.context),
        Command::Dependencies(cmd) => {
            base_commands::dependencies::handle_command(cmd, env, args.context)
        }
        Command::Doc(cmd) => base_commands::doc::handle_command(cmd, env, args.context),
        Command::Fix(cmd) => base_commands::fix::handle_command(cmd, env, args.context, None),
        Command::Publish(cmd) => base_commands::publish::handle_command(cmd, env, args.context),
        Command::Vulnerabilities(cmd) => {
            base_commands::vulnerabilities::handle_command(cmd, env, args.context)
        }
        _ => Err(anyhow::anyhow!("Unknown command")),
    }
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let (args, environment) = init_xtask::<Command>(parse_args::<Command>()?)?;

    if args.context == Context::NoStd {
        // Install additional targets for no-std execution environments
        rustup_add_target(WASM32_TARGET)?;
        rustup_add_target(ARM_TARGET)?;
        rustup_add_target(ARM_NO_ATOMIC_PTR_TARGET)?;
    }

    match args.command {
        Command::Books(cmd_args) => cmd_args.parse(),
        Command::Build(cmd_args) => {
            commands::build::handle_command(cmd_args, environment, args.context)
        }
        Command::Remote(cmd_args) => {
            commands::remote::handle_command(cmd_args, environment, args.context)
        }
        Command::Doc(cmd_args) => {
            commands::doc::handle_command(cmd_args, environment, args.context)
        }
        Command::Test(cmd_args) => {
            commands::test::handle_command(cmd_args, environment, args.context)
        }
        Command::Validate(cmd_args) => {
            commands::validate::handle_command(cmd_args, environment, args.context)
        }
        _ => dispatch_base_commands(args, environment),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
