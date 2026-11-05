//! `burn-instrument-report <measurement.json> [--against <saved.json>]`
//!
//! Print the report of a measurement `burn-instrument` took and saved, and with `--against`, the
//! measurement held against a saved one.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use burn_instrument::Measurement;
use burn_instrument_report::Report;

const USAGE: &str = "usage: burn-instrument-report <measurement.json> [--against <saved.json>]";

/// What the command line asked for.
struct Arguments {
    /// The measurement to report.
    measurement: PathBuf,
    /// The saved measurement to hold it against.
    against: Option<PathBuf>,
}

impl Arguments {
    /// What `arguments`, the command line without the program name, asks for.
    ///
    /// # Errors
    ///
    /// When the line names no measurement, two of them, a flag nobody defined, or `--against` with
    /// nothing after it.
    fn new(mut arguments: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut measurement = None;
        let mut against = None;
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--against" => match arguments.next() {
                    Some(saved) => against = Some(PathBuf::from(saved)),
                    None => {
                        return Err("`--against` needs the saved measurement's file".to_string());
                    }
                },
                flag if flag.starts_with("--") => return Err(format!("unknown flag `{flag}`")),
                path if measurement.is_none() => measurement = Some(PathBuf::from(path)),
                extra => return Err(format!("one measurement at a time, not also `{extra}`")),
            }
        }
        Ok(Self {
            measurement: measurement.ok_or_else(|| "which measurement?".to_string())?,
            against,
        })
    }

    /// Print the report the line asked for.
    fn run(self) -> Result<(), String> {
        let measurement = read(&self.measurement)?;
        let report = Report::new(&measurement);
        print!("{report}");
        if let Some(against) = &self.against {
            let saved = read(against)?;
            print!("{}", report.against(&saved));
        }
        Ok(())
    }
}

/// The measurement saved at `path`.
fn read(path: &Path) -> Result<Measurement, String> {
    let file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    Measurement::read(file).map_err(|error| format!("{}: {error}", path.display()))
}

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1);
    match Arguments::new(arguments).and_then(Arguments::run) {
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
        Arguments::new(line.iter().map(|argument| argument.to_string()))
    }

    #[test]
    fn a_measurement_alone_or_against_a_saved_one() {
        let alone = parsed(&["now.json"]).expect("parsed");
        assert_eq!(alone.measurement, PathBuf::from("now.json"));
        assert_eq!(alone.against, None);

        // The flag may come before or after the measurement.
        for line in [
            &["now.json", "--against", "was.json"][..],
            &["--against", "was.json", "now.json"][..],
        ] {
            let against = parsed(line).expect("parsed");
            assert_eq!(against.measurement, PathBuf::from("now.json"));
            assert_eq!(against.against, Some(PathBuf::from("was.json")));
        }
    }

    /// What the line cannot mean is refused rather than guessed at: no measurement, a second one,
    /// a flag nobody defined, or `--against` with nothing after it.
    #[test]
    fn a_line_that_names_no_single_measurement_is_refused() {
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
