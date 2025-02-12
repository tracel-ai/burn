use std::collections::HashMap;

use strum::IntoEnumIterator;
use tracel_xtask::prelude::*;

use crate::{ARM_NO_ATOMIC_PTR_TARGET, ARM_TARGET, NO_STD_CRATES, WASM32_TARGET};

#[macros::extend_command_args(BuildCmdArgs, Target, None)]
pub struct BurnBuildCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: BurnBuildCmdArgs,
    exec_env: ExecutionEnvironment,
) -> anyhow::Result<()> {
    match exec_env {
        ExecutionEnvironment::NoStd => {
            [
                "Default",
                WASM32_TARGET,
                ARM_TARGET,
                ARM_NO_ATOMIC_PTR_TARGET,
            ]
            .iter()
            .try_for_each(|build_target| {
                let mut build_args = vec!["--no-default-features"];
                let mut env_vars = HashMap::new();
                if *build_target != "Default" {
                    build_args.extend(vec!["--target", *build_target]);
                }
                if *build_target == ARM_NO_ATOMIC_PTR_TARGET {
                    env_vars.insert(
                        "RUSTFLAGS",
                        "--cfg portable_atomic_unsafe_assume_single_core",
                    );
                }
                // RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
                // https://docs.rs/getrandom/latest/getrandom/#webassembly-support
                if *build_target == WASM32_TARGET {
                    env_vars.insert("RUSTFLAGS", "--cfg getrandom_backend=\"wasm_js\"");
                }
                helpers::custom_crates_build(
                    NO_STD_CRATES.to_vec(),
                    build_args,
                    Some(env_vars),
                    None,
                    &format!("no-std with target {}", *build_target),
                )
            })?;
            Ok(())
        }
        ExecutionEnvironment::Std => {
            if args.ci {
                // Exclude crates that are not supported on CI
                args.exclude.extend(vec![
                    "burn-cuda".to_string(),
                    "burn-hip".to_string(),
                    "burn-tch".to_string(),
                ]);
                if std::env::var("DISABLE_WGPU").is_ok() {
                    args.exclude.extend(vec!["burn-wgpu".to_string()]);
                };
            }
            // Build workspace
            base_commands::build::handle_command(args.try_into().unwrap())?;
            // Specific additional commands to test specific features
            // burn-dataset
            helpers::custom_crates_build(
                vec!["burn-dataset"],
                vec!["--all-features"],
                None,
                None,
                "std with all features",
            )?;
            Ok(())
        }
        ExecutionEnvironment::All => ExecutionEnvironment::iter()
            .filter(|env| *env != ExecutionEnvironment::All)
            .try_for_each(|env| {
                handle_command(
                    BurnBuildCmdArgs {
                        target: args.target.clone(),
                        exclude: args.exclude.clone(),
                        only: args.only.clone(),
                        ci: args.ci,
                    },
                    env,
                )
            }),
    }
}
