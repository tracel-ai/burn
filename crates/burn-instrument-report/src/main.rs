//! `burn-instrument-report <pass.json> [--against <saved.json>]`
//!
//! Print the report of a pass `burn-instrument` collected and saved, and with `--against`, the
//! pass held against a saved one.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn_instrument::CollectedPass;
use burn_instrument_report::Report;

const USAGE: &str = "usage: burn-instrument-report <pass.json> [--against <saved.json>]";

/// What the command line asked for.
struct Arguments {
    /// The pass to report.
    pass: PathBuf,
    /// The saved pass to hold it against.
    against: Option<PathBuf>,
}

impl Arguments {
    /// Parse `arguments`, the command line without the program name.
    fn parse(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut pass = None;
        let mut against = None;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--against" => match arguments.next() {
                    Some(saved) => against = Some(PathBuf::from(saved)),
                    None => return Err("`--against` needs the saved pass's file".to_string()),
                },
                flag if flag.starts_with("--") => return Err(format!("unknown flag `{flag}`")),
                path if pass.is_none() => pass = Some(PathBuf::from(path)),
                extra => return Err(format!("one pass at a time, not also `{extra}`")),
            }
        }
        Ok(Self {
            pass: pass.ok_or_else(|| "which pass?".to_string())?,
            against,
        })
    }
}

/// The pass saved at `path`.
fn read(path: &Path) -> Result<CollectedPass, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    CollectedPass::read(file).map_err(|error| format!("{}: {error}", path.display()))
}

fn run(arguments: Arguments) -> Result<(), String> {
    let pass = read(&arguments.pass)?;
    let report = Report::new(&pass);
    print!("{report}");
    if let Some(against) = &arguments.against {
        let saved = read(against)?;
        print!("{}", report.against(&saved));
    }
    Ok(())
}

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1);
    match Arguments::parse(arguments).and_then(run) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}\n{USAGE}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parsed(line: &[&str]) -> Result<Arguments, String> {
        Arguments::parse(line.iter().map(|argument| argument.to_string()))
    }

    #[test]
    fn a_pass_alone_or_against_a_saved_one() {
        let alone = parsed(&["now.json"]).expect("parsed");
        assert_eq!(alone.pass, PathBuf::from("now.json"));
        assert_eq!(alone.against, None);

        // The flag may come before or after the pass.
        for line in [
            &["now.json", "--against", "was.json"][..],
            &["--against", "was.json", "now.json"][..],
        ] {
            let against = parsed(line).expect("parsed");
            assert_eq!(against.pass, PathBuf::from("now.json"));
            assert_eq!(against.against, Some(PathBuf::from("was.json")));
        }
    }

    /// What the line cannot mean is refused rather than guessed at: no pass, a second pass, a
    /// flag nobody defined, or `--against` with nothing after it.
    #[test]
    fn a_line_that_names_no_single_pass_is_refused() {
        for line in [
            &[][..],
            &["now.json", "other.json"][..],
            &["now.json", "--save"][..],
            &["now.json", "--against"][..],
        ] {
            assert!(parsed(line).is_err(), "{line:?}");
        }
    }
}
