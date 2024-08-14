use tracel_xtask::prelude::*;

pub fn handle_command() -> anyhow::Result<()> {
    let target = Target::Workspace;
    let exclude = vec![];
    let only = vec![];

    // ==============
    // std validation
    // ==============

    // checks
    [
        CheckSubCommand::Audit,
        CheckSubCommand::Format,
        CheckSubCommand::Lint,
        CheckSubCommand::Typos,
    ]
    .iter()
    .try_for_each(|c| {
        base_commands::check::handle_command(CheckCmdArgs {
            target: target.clone(),
            exclude: exclude.clone(),
            only: only.clone(),
            command: c.clone(),
        })
    })?;

    // build
    super::build::handle_command(
        BuildCmdArgs {
            target: target.clone(),
            exclude: exclude.clone(),
            only: only.clone(),
        },
        ExecutionEnvironment::Std,
    )?;

    // tests
    super::test::handle_command(
        TestCmdArgs {
            target: target.clone(),
            exclude: exclude.clone(),
            only: only.clone(),
            threads: None,
            command: TestSubCommand::All,
        },
        ExecutionEnvironment::Std,
    )?;

    // documentation
    [DocSubCommand::Build, DocSubCommand::Tests]
        .iter()
        .try_for_each(|c| {
            super::doc::handle_command(DocCmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
                command: c.clone(),
            })
        })?;

    // =================
    // no-std validation
    // =================

    #[cfg(target_os = "linux")]
    {
        // build
        super::build::handle_command(
            BuildCmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
            },
            ExecutionEnvironment::NoStd,
        )?;

        // tests
        super::test::handle_command(
            TestCmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
                threads: None,
                command: TestSubCommand::All,
            },
            ExecutionEnvironment::NoStd,
        )?;
    }

    Ok(())
}
