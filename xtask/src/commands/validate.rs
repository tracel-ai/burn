use tracel_xtask::prelude::*;

pub fn handle_command(exec_env: &ExecutionEnvironment) -> anyhow::Result<()> {
    let target = Target::Workspace;
    let exclude = vec![];
    let only = vec![];

    if *exec_env == ExecutionEnvironment::Std || *exec_env == ExecutionEnvironment::All {
        // ==============
        // std validation
        // ==============
        info!("Run validation for std execution environment...");

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
                command: Some(c.clone()),
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
                command: Some(TestSubCommand::All),
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
                    command: Some(c.clone()),
                })
            })?;
    }

    if *exec_env == ExecutionEnvironment::NoStd || *exec_env == ExecutionEnvironment::All {
        // =================
        // no-std validation
        // =================
        info!("Run validation for no-std execution environment...");

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
                    command: Some(TestSubCommand::All),
                },
                ExecutionEnvironment::NoStd,
            )?;
        }
    }

    Ok(())
}
