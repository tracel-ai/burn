use std::{collections::HashMap, path::Path, time::Instant};

use clap::{Args, Subcommand};
use derive_more::Display;

use crate::{
    endgroup, group,
    logging::init_logger,
    utils::{
        cargo::ensure_cargo_crate_is_installed, mdbook::run_mdbook_with_path, process::random_port,
        time::format_duration, Params,
    },
};

#[derive(Args)]
pub(crate) struct BooksArgs {
    #[command(subcommand)]
    book: BookKind,
}

#[derive(Subcommand)]
pub(crate) enum BookKind {
    ///  Burn Book, a.k.a. the guide, made for the Burn users.
    Burn(BookKindArgs),
    /// Contributor book, made for people willing to get all the technical understanding and advices to contribute actively to the project.
    Contributor(BookKindArgs),
}

#[derive(Args)]
pub(crate) struct BookKindArgs {
    #[command(subcommand)]
    command: BookCommand,
}

#[derive(Subcommand, Display)]
pub(crate) enum BookCommand {
    /// Build the book
    Build,
    /// Open the book on the specified port or random port and rebuild it automatically upon changes
    Open(OpenArgs),
}

#[derive(Args, Display)]
pub(crate) struct OpenArgs {
    /// Specify the port to open the book on (defaults to a random port if not specified)
    #[clap(long, default_value_t = random_port())]
    port: u16,
}

/// Book information
pub(crate) struct Book {
    name: &'static str,
    path: &'static Path,
}

impl BooksArgs {
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
    const CONTRIBUTOR_BOOK_PATH: &'static str = "./contributor-book";

    pub(crate) fn run(book_arg: &BookKind) -> anyhow::Result<()> {
        let (book, command) = match book_arg {
            BookKind::Burn(args) => (
                Self {
                    name: Self::BURN_BOOK_NAME,
                    path: Path::new(Self::BURN_BOOK_PATH),
                },
                &args.command,
            ),
            BookKind::Contributor(args) => (
                Self {
                    name: Self::CONTRIBUTOR_BOOK_NAME,
                    path: Path::new(Self::CONTRIBUTOR_BOOK_PATH),
                },
                &args.command,
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
            BookCommand::Open(args) => self.open(args),
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

    fn open(&self, args: &OpenArgs) {
        run_mdbook_with_path(
            "serve",
            Params::from(["--open", "--port", &args.port.to_string()]),
            HashMap::new(),
            Some(self.path),
            "mdbook should build the book successfully",
        );
    }
}
