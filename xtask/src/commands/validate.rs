use tracel_xtask::prelude::*;

use crate::commands::{
    build::BurnBuildCmdArgs,
    test::{BurnTestCmdArgs, CiTestType},
};

pub fn handle_command(
    args: &ValidateCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    let target = Target::Workspace;
    let exclude = vec![];
    let only = vec![];

    if context == Context::NoStd || context == Context::All {
        // =================
        // no-std validation
        // =================
        info!("Run validation for no-std execution environment...");

        #[cfg(target_os = "linux")]
        {
            // build
            super::build::handle_command(
                BurnBuildCmdArgs {
                    target: target.clone(),
                    exclude: exclude.clone(),
                    only: only.clone(),
                    ci: true,
                    release: args.release,
                },
                env.clone(),
                Context::NoStd,
            )?;

            // tests
            super::test::handle_command(
                BurnTestCmdArgs {
                    target: target.clone(),
                    exclude: exclude.clone(),
                    only: only.clone(),
                    threads: None,
                    jobs: None,
                    command: Some(TestSubCommand::All),
                    ci: CiTestType::GithubRunner,
                    features: None,
                    no_default_features: false,
                    force: false,
                    no_capture: false,
                    release: args.release,
                    test: None,
                },
                env.clone(),
                Context::NoStd,
            )?;
        }
    }

    if context == Context::Std || context == Context::All {
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
            base_commands::check::handle_command(
                CheckCmdArgs {
                    target: target.clone(),
                    exclude: exclude.clone(),
                    only: only.clone(),
                    command: Some(c.clone()),
                    ignore_audit: args.ignore_audit,
                },
                env.clone(),
                context.clone(),
            )
        })?;

        // build
        super::build::handle_command(
            BurnBuildCmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
                ci: true,
                release: args.release,
            },
            env.clone(),
            Context::Std,
        )?;

        // tests
        super::test::handle_command(
            BurnTestCmdArgs {
                target: target.clone(),
                exclude: exclude.clone(),
                only: only.clone(),
                threads: None,
                jobs: None,
                command: Some(TestSubCommand::All),
                ci: CiTestType::GithubRunner,
                features: None,
                no_default_features: false,
                release: args.release,
                test: None,
                force: false,
                no_capture: false,
            },
            env.clone(),
            Context::Std,
        )?;

        // documentation
        [DocSubCommand::Build, DocSubCommand::Tests]
            .iter()
            .try_for_each(|c| {
                super::doc::handle_command(
                    DocCmdArgs {
                        target: target.clone(),
                        exclude: exclude.clone(),
                        only: only.clone(),
                        command: Some(c.clone()),
                    },
                    env.clone(),
                    context.clone(),
                )
            })?;
    }

    Ok(())
}
