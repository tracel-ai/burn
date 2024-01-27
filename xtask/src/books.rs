use std::{collections::HashMap, path::Path, time::Instant};

use clap::{Args, Subcommand};
use derive_more::Display;

use crate::{
    endgroup, group,
    logging::init_logger,
    utils::{
        cargo::ensure_cargo_crate_is_installed, mdbook::run_mdbook_with_path,
        time::format_duration, Params, process::random_port,
    },
};

#[derive(Args)]
pub(crate) struct BookArgs {
    #[command(subcommand)]
    pub book: BookArg,
}

#[derive(Subcommand)]
pub(crate) enum BookArg {
    ///  Burn Book, a.k.a. the guide, made for the Burn users.
    Burn { command: BookCommand },
    /// Contributor book, made for people willing to get all the technical understanding and advices to contribute actively to the project.
    Contributor { command: BookCommand },
}

#[derive(clap::ValueEnum, Default, Display, Clone)]
pub(crate) enum BookCommand {
    /// Build the book
    Build,
    /// Open the book on a random port and rebuild it automatically upon changes
    #[default]
    Open,
}

pub(crate) struct Book {
    name: &'static str,
    path: &'static Path,
}

impl BookArgs {
    pub(crate) fn parse(&self) -> anyhow::Result<()> {
        init_logger().init();
        let start = Instant::now();
        Book::run(&self.book)?;
        let duration = start.elapsed();
        info!(
            "\x1B[32;1mTime elapsed for the current execution: {}\x1B[0m",
            format_duration(&duration)
        );
        Ok(())
    }
}

impl Book {
    const BURN_BOOK_NAME: &'static str = "Burn Book";
    const BURN_BOOK_PATH: &'static str = "./burn-book";

    const CONTRIBUTOR_BOOK_NAME: &'static str = "Contributor Book";
    const CONTRIBUTOR_BOOK_PATH: &'static str = "./burn-book";

    pub(crate) fn run(book_arg: &BookArg) -> anyhow::Result<()> {
        let (book, command) = match book_arg {
            BookArg::Burn { command } => (
                Self {
                    name: Self::BURN_BOOK_NAME,
                    path: Path::new(Self::BURN_BOOK_PATH),
                },
                command,
            ),
            BookArg::Contributor { command } => (
                Self {
                    name: Self::CONTRIBUTOR_BOOK_NAME,
                    path: Path::new(Self::CONTRIBUTOR_BOOK_PATH),
                },
                command,
            ),
        };
        book.execute(command);
        Ok(())
    }

    fn execute(&self, command: &BookCommand) {
        ensure_cargo_crate_is_installed("mdbook");
        group!("{}: {}", self.name, command);
        match command {
            BookCommand::Build => self.build(),
            BookCommand::Open => self.open(),
        };
        endgroup!();
    }

    fn build(&self) {
        run_mdbook_with_path(
            "build",
            Params::from([]),
            HashMap::new(),
            Some(self.path),
            "mdbook should build the book successfully",
        );
    }

    fn open(&self) {
        run_mdbook_with_path(
            "serve",
            Params::from([
                "--open",
                "--port",
                &random_port().to_string(),
            ]),
            HashMap::new(),
            Some(self.path),
            "mdbook should build the book successfully",
        );
    }

}
