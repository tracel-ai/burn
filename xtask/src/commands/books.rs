use std::path::Path;

use tracel_xtask::prelude::*;

#[derive(clap::Args)]
pub struct BooksArgs {
    #[command(subcommand)]
    book: BookKind,
}

#[derive(clap::Subcommand)]
pub(crate) enum BookKind {
    ///  Burn Book, a.k.a. the guide, made for the Burn users.
    Burn(BookKindArgs),
    /// Contributor book, made for people willing to get all the technical understanding and advice to contribute actively to the project.
    Contributor(BookKindArgs),
}

#[derive(clap::Args)]
pub(crate) struct BookKindArgs {
    #[command(subcommand)]
    command: BookSubCommand,
}

#[derive(clap::Subcommand, strum::Display)]
pub(crate) enum BookSubCommand {
    /// Build the book
    Build,
    /// Open the book on the specified port or random port and rebuild it automatically upon changes
    Open(OpenArgs),
}

#[derive(clap::Args)]
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

    fn execute(&self, command: &BookSubCommand) -> anyhow::Result<()> {
        ensure_cargo_crate_is_installed("mdbook", None, None, false)?;
        group!("{}: {}", self.name, command);
        match command {
            BookSubCommand::Build => self.build(),
            BookSubCommand::Open(args) => self.open(args),
        }?;
        endgroup!();
        Ok(())
    }

    fn build(&self) -> anyhow::Result<()> {
        run_process(
            "mdbook",
            &vec!["build"],
            None,
            Some(self.path),
            "mdbook should build the book successfully",
        )
    }

    fn open(&self, args: &OpenArgs) -> anyhow::Result<()> {
        run_process(
            "mdbook",
            &vec!["serve", "--open", "--port", &args.port.to_string()],
            None,
            Some(self.path),
            "mdbook should open the book successfully",
        )
    }
}
