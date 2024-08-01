use xtask_common::{
    anyhow,
    commands::doc::{self, DocCmdArgs},
    utils::helpers,
};

pub(crate) fn handle_command(mut args: DocCmdArgs) -> anyhow::Result<()> {
    if args.command == doc::DocCommand::Build {
        args.exclude.push("burn-cuda".to_string());
    }
    doc::handle_command(args.clone())?;
    // Specific additional commands to build other docs
    if args.command == doc::DocCommand::Build {
        // burn-dataset
        helpers::additional_crates_doc_build(vec!["burn-dataset"], vec!["--all-features"])?;
    }
    Ok(())
}
