use tracel_xtask::prelude::*;

pub(crate) fn handle_command(mut args: DocCmdArgs) -> anyhow::Result<()> {
    if args.get_command() == DocSubCommand::Build {
        args.exclude.push("burn-cuda".to_string());
    }

    // Execute documentation command on workspace
    base_commands::doc::handle_command(args.clone())?;

    // Specific additional commands to build other docs
    if args.get_command() == DocSubCommand::Build {
        // burn-dataset
        helpers::custom_crates_doc_build(vec!["burn-dataset"], vec!["--all-features"])?;
    }
    Ok(())
}
