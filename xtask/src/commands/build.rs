use std::collections::HashMap;

use tracel_xtask::prelude::{clap::ValueEnum, *};

use crate::{ARM_NO_ATOMIC_PTR_TARGET, ARM_TARGET, NO_STD_CRATES, WASM32_TARGET};

#[macros::extend_command_args(BuildCmdArgs, Target, None)]
pub struct BurnBuildCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: BurnBuildCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    match context {
        Context::NoStd => {
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

                let mut crates = NO_STD_CRATES.to_vec();

                if *build_target == ARM_NO_ATOMIC_PTR_TARGET {
                    // Temporarily remove `burn-autodiff` from building with the
                    // target `thumbv6m-none-eabi` as it requires enabling the
                    // `arbitrary_self_types` feature for the
                    // `clone_if_require_grad` method of
                    // `burn-autodiff::graph::Node`.
                    //
                    // Also temporarily remove `burn-flex` and `burn-no-std-tests`
                    // from this general pass; they are built separately below
                    // with the `critical-section` feature, because `burn-flex`
                    // pulls `once_cell` transitively via `gemm`, and `once_cell`
                    // uses `AtomicPtr::compare_exchange` which is unavailable on
                    // targets without atomic CAS unless its `critical-section`
                    // backend is enabled.
                    crates.retain(|&v| {
                        v != "burn-autodiff" && v != "burn-flex" && v != "burn-no-std-tests"
                    });

                    env_vars.insert(
                        "RUSTFLAGS",
                        "--cfg portable_atomic_unsafe_assume_single_core",
                    );
                }
                helpers::custom_crates_build(
                    crates,
                    build_args.clone(),
                    Some(env_vars.clone()),
                    None,
                    &format!("no-std with target {}", *build_target),
                )?;

                // Second pass for `thumbv6m-none-eabi`: build burn-flex and
                // burn-no-std-tests with the `critical-section` feature enabled
                // so `once_cell` (transitively pulled from `gemm`) uses
                // portable-atomic for CAS emulation.
                //
                // Note: `portable-atomic`'s `critical-section` feature is
                // mutually exclusive with the `portable_atomic_unsafe_assume_single_core`
                // cfg, so this pass does not carry over that cfg from the first
                // pass. `portable-atomic` uses its critical-section backend.
                if *build_target == ARM_NO_ATOMIC_PTR_TARGET {
                    let mut cs_args = build_args.clone();
                    cs_args.extend(vec!["--features", "critical-section"]);
                    helpers::custom_crates_build(
                        vec!["burn-flex", "burn-no-std-tests"],
                        cs_args,
                        None,
                        None,
                        &format!("no-std with target {} (critical-section)", *build_target),
                    )?;
                }

                anyhow::Ok(())
            })?;
            Ok(())
        }
        Context::Std => {
            if args.ci {
                // Exclude crates that are not supported on CI
                args.exclude.extend(vec![
                    "burn-cuda".to_string(),
                    "burn-rocm".to_string(),
                    "burn-tch".to_string(),
                ]);
                if std::env::var("DISABLE_WGPU").is_ok() {
                    args.exclude.extend(vec!["burn-wgpu".to_string()]);
                };
            }
            // Build workspace
            base_commands::build::handle_command(args.try_into().unwrap(), env, context)?;
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
        Context::All => Context::value_variants()
            .iter()
            .filter(|ctx| **ctx != Context::All)
            .try_for_each(|ctx| {
                handle_command(
                    BurnBuildCmdArgs {
                        target: args.target.clone(),
                        exclude: args.exclude.clone(),
                        only: args.only.clone(),
                        ci: args.ci,
                        release: args.release,
                        features: args.features.clone(),
                        no_default_features: args.no_default_features,
                    },
                    env.clone(),
                    ctx.clone(),
                )
            }),
    }
}
