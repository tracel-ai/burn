use xtask_common::{
    anyhow,
    commands::{
        build::BuildCmdArgs,
        check::{self, CheckCmdArgs, CheckCommand},
        doc::{DocCmdArgs, DocCommand},
        test::{TestCmdArgs, TestCommand},
        Target,
    },
    ExecutionEnvironment,
};

pub fn handle_command() -> anyhow::Result<()> {
    let target = Target::Workspace;
    let exclude = vec![];
    let only = vec![];

    // ==============
    // std validation
    // ==============

    // checks
    [
        CheckCommand::Audit,
        CheckCommand::Format,
        CheckCommand::Lint,
        CheckCommand::Typos,
    ]
    .iter()
    .try_for_each(|c| {
        check::handle_command(CheckCmdArgs {
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
            command: TestCommand::All,
        },
        ExecutionEnvironment::Std,
    )?;

    // documentation
    [DocCommand::Build, DocCommand::Tests]
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
                command: TestCommand::All,
            },
            ExecutionEnvironment::NoStd,
        )?;
    }

    Ok(())
}
