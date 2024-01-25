use clap::{Parser, Subcommand};

mod logging;
mod publish;
mod runchecks;
mod utils;
mod vulnerabilities;

#[macro_use]
extern crate log;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Publish a crate to crates.io
    Publish {
        /// The name of the crate to publish on crates.io
        name: String,
    },
    /// Run the specified `burn` tests and checks locally.
    RunChecks {
        /// The environment to run checks against
        env: runchecks::CheckType,
    },
    /// Run the specified vulnerability check locally. This command should be called with 'cargo +nightly'.
    Vulnerabilities {
        /// The vulnerability to run
        vulnerability: vulnerabilities::VulnerabilityType,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Publish { name } => publish::run(name),
        Command::RunChecks { env } => runchecks::run(env),
        Command::Vulnerabilities { vulnerability } => vulnerabilities::run(vulnerability),
    }
}
