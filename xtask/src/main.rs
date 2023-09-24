use clap::{Parser, Subcommand};

mod publish;
mod runchecks;

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
        #[arg(short, long)]
        jobs: Option<u32>,
    },
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    match args.command {
        Command::RunChecks { env, jobs } => runchecks::run(env, jobs),
        Command::Publish { name } => publish::run(name),
    }
}
