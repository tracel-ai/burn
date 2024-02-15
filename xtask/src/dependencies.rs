use std::{collections::HashMap, time::Instant};

use crate::{
    endgroup, group,
    logging::init_logger,
    utils::{
        cargo::{ensure_cargo_crate_is_installed, run_cargo},
        rustup::is_current_toolchain_nightly,
        time::format_duration,
        Params,
    },
};

#[derive(clap::ValueEnum, Default, Copy, Clone, PartialEq, Eq)]
pub(crate) enum DependencyCheck {
    /// Run all dependency checks.
    #[default]
    All,
    /// Perform an audit of all dependencies using the cargo-audit crate `<https://crates.io/crates/cargo-audit>`
    Audit,
    /// Run cargo-deny check `<https://crates.io/crates/cargo-deny>`
    Deny,
    /// Run cargo-udeps to find unused dependencies `<https://crates.io/crates/cargo-udeps>`
    Unused,
}

impl DependencyCheck {
    pub(crate) fn run(&self) -> anyhow::Result<()> {
        // Setup logger
        init_logger().init();
        // Start time measurement
        let start = Instant::now();
        match self {
            Self::Audit => cargo_audit(),
            Self::Deny => cargo_deny(),
            Self::Unused => cargo_udeps(),
            Self::All => {
                cargo_audit();
                cargo_deny();
                cargo_udeps();
            }
        }

        // Stop time measurement
        //
        // Compute runtime duration
        let duration = start.elapsed();

        // Print duration
        info!(
            "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
            format_duration(&duration)
        );

        Ok(())
    }
}

/// Run cargo-audit
fn cargo_audit() {
    ensure_cargo_crate_is_installed("cargo-audit");
    // Run cargo audit
    group!("Cargo: run audit checks");
    run_cargo(
        "audit",
        Params::from([]),
        HashMap::new(),
        "Cargo audit should be installed and it should correctly run",
    );
    endgroup!();
}

/// Run cargo-deny
fn cargo_deny() {
    ensure_cargo_crate_is_installed("cargo-deny");
    // Run cargo deny
    group!("Cargo: run deny checks");
    run_cargo(
        "deny",
        Params::from(["check"]),
        HashMap::new(),
        "Cargo deny should be installed and it should correctly run",
    );
    endgroup!();
}

/// Run cargo-udeps
fn cargo_udeps() {
    if is_current_toolchain_nightly() {
        ensure_cargo_crate_is_installed("cargo-udeps");
        // Run cargo udeps
        group!("Cargo: run unused dependencies checks");
        run_cargo(
            "udeps",
            Params::from([]),
            HashMap::new(),
            "Cargo udeps should be installed and it should correctly run",
        );
        endgroup!();
    } else {
        error!(
            "You must use 'cargo +nightly' to check for unused dependencies.
Install a nightly toolchain with 'rustup toolchain install nightly'."
        )
    }
}
