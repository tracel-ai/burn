use xtask_common::clap::{self, Parser, Subcommand};

mod books;
mod runchecks;

#[macro_use]
extern crate log;

use std::time::Instant;
use xtask_common::{anyhow, commands::*, init_xtask, utils::time::format_duration};

#[derive(clap::Parser)]
#[command(author, version, about, long_about = None)]
struct XtaskArgs {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    /// Run commands to manage Burn Books
    Books(books::BooksArgs),
    /// Run the specified `burn` tests and checks locally.
    RunChecks {
        /// The environment to run checks against
        #[clap(value_enum, default_value_t = runchecks::CheckType::default())]
        env: runchecks::CheckType,
    },

    // From common_xtask
    /// Bump the version of all crates to be published
    Bump(bump::BumpCmdArgs),
    /// Run the specified dependencies check locally
    Dependencies(dependencies::DependenciesCmdArgs),
    /// Publish a crate to crates.io
    Publish(publish::PublishCmdArgs),
    /// Run the specified vulnerability check locally. These commands must be called with 'cargo +nightly'.
    Vulnerabilities(vulnerabilities::VulnerabilitiesCmdArgs),
}

fn main() -> anyhow::Result<()> {
    init_xtask();
    let args = XtaskArgs::parse();

    let start = Instant::now();
    match args.command {
        Command::Books(args) => args.parse(),
        Command::RunChecks { env } => env.run(),

        // From common_xtask
        Command::Bump(args) => bump::handle_command(args),
        Command::Dependencies(args) => dependencies::handle_command(args),
        Command::Publish(args) => publish::handle_command(args),
        Command::Vulnerabilities(args) => vulnerabilities::handle_command(args),
    }?;

    let duration = start.elapsed();
    info!(
        "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
