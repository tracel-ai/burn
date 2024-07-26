use std::{collections::HashMap, path::Path};

use xtask_common::{
    anyhow,
    clap::{self, Args, Subcommand},
    derive_more::Display,
    endgroup, group,
    utils::{
        cargo::ensure_cargo_crate_is_installed, mdbook::run_mdbook_with_path, process::random_port,
        Params,
    },
};

#[derive(clap::Args)]
pub struct BooksArgs {
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
        Book::run(&self.book)
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
        book.execute(command)
    }

    fn execute(&self, command: &BookCommand) -> anyhow::Result<()> {
        ensure_cargo_crate_is_installed("mdbook", None, None, false)?;
        group!("{}: {}", self.name, command);
        match command {
            BookCommand::Build => self.build(),
            BookCommand::Open(args) => self.open(args),
        }?;
        endgroup!();
        Ok(())
    }

    fn build(&self) -> anyhow::Result<()> {
        run_mdbook_with_path(
            "build",
            Params::from([]),
            HashMap::new(),
            Some(self.path),
            "mdbook should build the book successfully",
        )
    }

    fn open(&self, args: &OpenArgs) -> anyhow::Result<()> {
        run_mdbook_with_path(
            "serve",
            Params::from(["--open", "--port", &args.port.to_string()]),
            HashMap::new(),
            Some(self.path),
            "mdbook should build the book successfully",
        )
    }
}
