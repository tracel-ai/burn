use xtask_common::clap;

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

#[xtask_macros::commands(
    Bump,
    Build,
    Checks,
    CI,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Test,
    Vulnerabilities
)]
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
        Command::Build(cmd_args) => build::handle_command(cmd_args),
        Command::Bump(cmd_args) => bump::handle_command(cmd_args),
        Command::Checks => commands::checks::handle_command(),
        Command::CI(cmd_args) => commands::ci::handle_command(cmd_args, args.execution_environment),
        Command::Coverage(cmd_args) => coverage::handle_command(cmd_args),
        Command::Compile(cmd_args) => compile::handle_command(cmd_args),
        Command::Dependencies(cmd_args) => dependencies::handle_command(cmd_args),
        Command::Doc(cmd_args) => commands::doc::handle_command(cmd_args),
        Command::Fix(cmd_args) => fix::handle_command(cmd_args, None),
        Command::Publish(cmd_args) => publish::handle_command(cmd_args),
        Command::Test(cmd_args) => test::handle_command(cmd_args),
        Command::Vulnerabilities(cmd_args) => vulnerabilities::handle_command(cmd_args),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
