use xtask_common::{
    anyhow,
    commands::{
        ci::{self as common_ci, CICommand},
        doc::{self as common_doc},
        Target,
    },
    ExecutionEnvironment,
};

pub fn handle_command() -> anyhow::Result<()> {
    let target = Target::Workspace;
    let exclude = vec![];
    let only = vec![];

    // std checks
    // ==========
    [
        CICommand::Audit,
        CICommand::Format,
        CICommand::Lint,
        CICommand::Typos,
        CICommand::Build,
        CICommand::AllTests,
    ]
    .iter()
    .try_for_each(|c| {
        super::ci::handle_command(
            common_ci::CICmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
                command: c.clone(),
            },
            ExecutionEnvironment::Std,
        )
    })?;
    super::doc::handle_command(common_doc::DocCmdArgs {
        target: target.clone(),
        exclude: exclude.clone(),
        only: only.clone(),
        command: common_doc::DocCommand::Build,
    })?;

    // no-std checks
    // =============
    #[cfg(target_os = "linux")]
    [CICommand::Build, CICommand::AllTests]
        .iter()
        .try_for_each(|c| {
            super::ci::handle_command(
                common_ci::CICmdArgs {
                    target: target.clone(),
                    exclude: exclude.clone(),
                    only: only.clone(),
                    command: c.clone(),
                },
                ExecutionEnvironment::NoStd,
            )
        })?;

    Ok(())
}
