use clap::{Parser, Subcommand};

mod books;
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
    /// Run commands to manage Burn Books
    Books(books::BookArgs),
    /// Run the specified dependencies check locally
    Dependencies {
        /// The dependency check to run
        dependency_check: dependencies::DependencyCheck,
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
    /// Run the specified vulnerability check locally. These commands must be called with 'cargo +nightly'.
    Vulnerabilities {
        /// The vulnerability check to run.
        /// For the reference visit the page `<https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html>`
        vulnerability_check: vulnerabilities::VulnerabilityCheck,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::Books(args) => args.parse(),
        Command::Dependencies { dependency_check } => dependency_check.run(),
        Command::Publish { name } => publish::run(name),
        Command::RunChecks { env } => env.run(),
        Command::Vulnerabilities {
            vulnerability_check,
        } => vulnerability_check.run(),
    }
}
