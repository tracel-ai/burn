use clap::{Parser, Subcommand};

mod publish;
mod runchecks;
mod runwasm;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Set up the log level used by commands
    #[clap(long = "log", default_value = "Info")]
    log_level: log::LevelFilter,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Publish a crate to crates.io
    Publish(publish::Options),
    /// Run the specified `burn` tests and checks locally.
    RunChecks(runchecks::Options),
    /// Run `wasm` command.
    #[command(flatten)]
    RunWasm(runwasm::WasmCommand),
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Setting log level in order to show information and data about commands
    // in execution
    env_logger::builder()
        .filter(Some("xtask"), args.log_level)
        .init();

    match args.command {
        Command::RunChecks(options) => runchecks::run(options),
        Command::Publish(options) => publish::run(options),
        Command::RunWasm(command) => runwasm::run(command),
    }
}
