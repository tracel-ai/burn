use tracel_xtask::prelude::*;

fn set_burn_device(device: &str) {
    // SAFETY: This is called in a single-threaded context within the xtask before spawning
    // child processes.
    unsafe {
        std::env::set_var("BURN_DEVICE", device);
    }
}

pub(crate) fn handle_command(
    mut args: DocCmdArgs,
    env: Environment,
    ctx: Context,
) -> anyhow::Result<()> {
    // The doc examples execute, so they need a device. flex is lightweight and portable, so
    // it is always available on the CI runners.
    set_burn_device("flex");

    if args.get_command() == DocSubCommand::Build {
        args.exclude
            .extend(vec!["burn-cuda".to_string(), "burn-rocm".to_string()]);
    }

    // Execute documentation command on workspace
    base_commands::doc::handle_command(args.clone(), env, ctx)?;

    // Specific additional commands to build other docs
    if args.get_command() == DocSubCommand::Build {
        // burn-dataset
        helpers::custom_crates_doc_build(
            vec!["burn-dataset"],
            vec!["--all-features"],
            None,
            None,
            "All features",
        )?;
    }
    Ok(())
}
