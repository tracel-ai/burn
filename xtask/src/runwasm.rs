use std::env;
use std::process::Command;

use crate::runchecks::{cargo_install, is_binary_installed, rustup};

// Targets constants
const WASM32_TARGET: &str = "wasm32-unknown-unknown";

fn run_wasm_pack(rustflags: &str) -> Command {
    // Set RUSTFLAGS parameters
    env::set_var("RUSTFLAGS", rustflags);

    // Build wasm-pack command
    let mut command = Command::new("wasm-pack");
    command.args(&["build", "--dev", "--target", "web", "--no-typescript"]);
    command
}

fn dist(arg: Build) -> anyhow::Result<()> {
    // Add target wasm32-unknown-unknown
    rustup("target", WASM32_TARGET);

    // Install wasm-pack
    if !is_binary_installed("wasm-pack") {
        cargo_install(["wasm-pack"].into());
    }

    // Retrieve build command checking whether the SIMD version build has been
    // enabled or not
    let build_command = if arg.is_simd {
        log::info!("Building SIMD version of wasm for web...");
        run_wasm_pack("-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 -Ctarget-feature=+simd128 --cfg web_sys_unstable_apis")
    } else {
        log::info!("Building Non-SIMD version of wasm for web...");
        run_wasm_pack(
            "-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 --cfg web_sys_unstable_apis",
        )
    };

    // Run wasm pack
    arg.base
        .build_command(build_command)
        .static_dir_path("examples/image-classification-web")
        .app_name("image-classification-web")
        .example("examples/image-classification-web")
        .run_in_workspace(true)
        .run("image-classification-web")?;

    /*if arg.optimize {
        xtask_wasm::WasmOpt::level(1)
            .shrink(2)
            .optimize(dist_result.wasm)?;
    }*/

    Ok(())
}

fn watch(arg: xtask_wasm::Watch) -> anyhow::Result<()> {
    let mut command = Command::new("cargo");
    command.arg("check");

    arg.run(command)?;

    Ok(())
}

fn start(arg: Server) -> anyhow::Result<()> {
    arg.server
        .arg("dist")
        .start(xtask_wasm::default_dist_dir(arg.release))?;
    Ok(())
}

#[derive(clap::Parser)]
pub enum WasmCommand {
    /// Run `dist` command
    Dist(Build),
    /// Run `watch` command
    Watch(xtask_wasm::Watch),
    /// Run `start` command
    Start(Server),
}

#[derive(clap::Parser)]
pub struct Build {
    /// Optimize the generated package using `wasm-opt`.
    #[clap(long)]
    optimize: bool,

    /// Whether is a SIMD version build
    #[clap(long)]
    is_simd: bool,

    #[clap(flatten)]
    base: xtask_wasm::Dist,
}

#[derive(clap::Parser)]
pub struct Server {
    /// Get `target/release` as default directory to start the server
    release: bool,

    #[clap(flatten)]
    server: xtask_wasm::DevServer,
}

pub fn run(command: WasmCommand) -> anyhow::Result<()> {
    match command {
        WasmCommand::Dist(arg) => {
            log::info!("Generating package...");
            dist(arg)?;
        }
        WasmCommand::Watch(arg) => {
            log::info!("Watching for changes and check...");
            watch(arg)?;
        }
        WasmCommand::Start(arg) => {
            log::info!("Starting the development server...");
            start(arg)?;
        }
    }
    Ok(())
}
