//! End-to-end validation for the remote backend.
//!
//! Builds the `server` example with a chosen backend, spawns it as a child process, waits
//! for it to bind, then runs `burn-backend-tests --features remote` against it with
//! `BURN_DEVICE=remote`. The child server is killed on exit regardless of how the test run
//! finishes.

use std::{
    net::{SocketAddr, TcpStream},
    path::PathBuf,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct RemoteCmdArgs {
    /// Backend the remote server should use.
    #[arg(long, value_enum, default_value_t = ServerBackend::Flex)]
    pub backend: ServerBackend,

    /// Port the server should listen on (also used by the client).
    #[arg(long, default_value_t = 7373)]
    pub port: u16,

    /// Build the server (and run tests) in release mode.
    #[arg(long)]
    pub release: bool,

    /// Maximum seconds to wait for the server to bind before giving up.
    #[arg(long, default_value_t = 30)]
    pub startup_timeout: u64,

    /// Test filter passed through to `cargo test` (positional, after `--`).
    #[arg(long)]
    pub test: Option<String>,
}

#[derive(Debug, Clone, clap::ValueEnum, strum::Display)]
pub enum ServerBackend {
    #[strum(to_string = "flex")]
    Flex,
    #[strum(to_string = "vulkan")]
    Vulkan,
    #[strum(to_string = "webgpu")]
    Webgpu,
    #[strum(to_string = "cuda")]
    Cuda,
}

pub fn handle_command(
    args: RemoteCmdArgs,
    _env: Environment,
    _context: Context,
) -> anyhow::Result<()> {
    let backend = args.backend.to_string();

    info!(
        "Building server example (backend = {backend}, release = {}) ...",
        args.release
    );
    build_server(&backend, args.release)?;

    let server_bin = server_binary_path(args.release);
    info!(
        "Spawning server at {} on port {} ...",
        server_bin.display(),
        args.port
    );
    let _server = ServerGuard::spawn(&server_bin, args.port)?;

    wait_for_bind(args.port, args.startup_timeout)?;
    info!("Server ready on 127.0.0.1:{}", args.port);

    run_remote_tests(args.port, args.release, args.test.as_deref())?;

    info!("Remote validation succeeded — tearing down the server.");
    Ok(())
}

fn build_server(backend: &str, release: bool) -> anyhow::Result<()> {
    // `--no-default-features` so we don't pull in the example's `webgpu` default on top of
    // whatever backend the caller picked (the example's `cfg_if!` would then build twice as
    // many backends for nothing).
    let mut cargo_args = vec![
        "build",
        "--example",
        "server",
        "--no-default-features",
        "--features",
        backend,
    ];
    if release {
        cargo_args.push("--release");
    }
    run_process(
        "cargo",
        &cargo_args,
        None,
        None,
        "Failed to build the `server` example",
    )
}

fn server_binary_path(release: bool) -> PathBuf {
    let profile = if release { "release" } else { "debug" };
    PathBuf::from("target")
        .join(profile)
        .join("examples")
        .join("server")
}

fn run_remote_tests(port: u16, release: bool, filter: Option<&str>) -> anyhow::Result<()> {
    let address = format!("ws://127.0.0.1:{port}");

    // Bypass `run_process`'s `&str` borrowing by owning the formatted strings here.
    let mut owned_args: Vec<String> = vec![
        "test".into(),
        "-p".into(),
        "burn-backend-tests".into(),
        "--features".into(),
        "remote".into(),
    ];
    if release {
        owned_args.push("--release".into());
    }
    // Tests share a single client connection — keep them single-threaded so a hang in
    // one test doesn't deadlock the whole suite.
    owned_args.push("--".into());
    owned_args.push("--test-threads=1".into());
    if let Some(f) = filter {
        owned_args.push(f.to_string());
    }
    let args: Vec<&str> = owned_args.iter().map(String::as_str).collect();

    let envs = std::collections::HashMap::from([
        ("BURN_DEVICE", "remote"),
        ("BURN_REMOTE_ADDRESS", address.as_str()),
    ]);

    run_process(
        "cargo",
        &args,
        Some(envs),
        None,
        "burn-backend-tests failed against the remote backend",
    )
}

fn wait_for_bind(port: u16, timeout_secs: u64) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse()?;
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    while Instant::now() < deadline {
        if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    anyhow::bail!(
        "Server failed to start on port {port} within {timeout_secs}s — check the server logs above."
    )
}

/// Owns the server `Child`. On drop, kills the process so a panicking / failing
/// test run still leaves no orphaned server.
struct ServerGuard(Child);

impl ServerGuard {
    fn spawn(bin: &std::path::Path, port: u16) -> anyhow::Result<Self> {
        let child = Command::new(bin)
            .env("REMOTE_BACKEND_PORT", port.to_string())
            // Inherit stdio so the server's logs show up in the xtask output. This is the
            // first thing you want when something goes wrong (e.g. "device not found").
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| anyhow::anyhow!("Failed to spawn server binary {}: {e}", bin.display()))?;
        Ok(Self(child))
    }
}

impl Drop for ServerGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
