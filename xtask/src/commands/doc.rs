use xtask_common::{
    anyhow,
    commands::doc::{self, DocCmdArgs},
    utils::helpers,
};

pub(crate) fn handle_command(mut args: DocCmdArgs) -> anyhow::Result<()> {
    match args.command {
        // Exclude crates that are not supported by CI
        doc::DocCommand::Build => {
            args.exclude.push("burn-cuda".to_string());
        }
    }
    doc::handle_command(args)?;
    // Specific additional commands to build other docs
    match args.command {
        // Exclude crates that are not supported by CI
        doc::DocCommand::Build => {
            // burn-dataset
            helpers::additional_crates_doc_build(vec!["burn-dataset"], vec!["--all-features"])?;
        }
    }
    Ok(())
}
