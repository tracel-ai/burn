use clap::{Parser, Subcommand};

mod dependencies;
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
    /// Run the specified dependencies check locally. This command should be called with 'cargo +nightly'.
    Dependencies {
        /// The dependency check to run
        dependency_check: dependencies::DependencyCheckType,
    },
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
        /// The vulnerability check to run.
        /// For the reference visit the page https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html
        vulnerability_check: vulnerabilities::VulnerabilityCheckType,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Dependencies { dependency_check } => dependencies::run(dependency_check),
        Command::Publish { name } => publish::run(name),
        Command::RunChecks { env } => runchecks::run(env),
        Command::Vulnerabilities {
            vulnerability_check,
        } => vulnerabilities::run(vulnerability_check),
    }
}
