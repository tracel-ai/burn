//! A breakdown as a file: versioned JSON that reads back on any machine, with or without a
//! runtime.

use std::io::{BufWriter, Read, Write};

use serde::{Deserialize, Serialize};

use super::Breakdown;

/// What a breakdown file says it is, so a reader can refuse anything else before decoding it.
const FORMAT: &str = "burn-instrument breakdown";

/// The version of the breakdown file format this build writes and reads.
///
/// Every change to what a breakdown serializes as (a field added, removed, renamed or
/// re-encoded) is a new version, so a saved pass is read back as it was collected or not at all.
pub const BREAKDOWN_FILE_VERSION: u32 = 1;

/// A breakdown as it is written to a file.
#[derive(Serialize)]
struct FileOut<'a> {
    format: &'static str,
    version: u32,
    breakdown: &'a Breakdown,
}

/// A breakdown file as it is read: the header, and the breakdown parsed but not yet decoded, so
/// a file of another format or version is refused before its contents are.
///
/// Every header field may be missing here, and is checked in [`Breakdown::read`]: a file with no
/// format is not a breakdown, and a version too wide for this build's is still one it does not
/// read.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIn {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    version: Option<u64>,
    #[serde(default)]
    breakdown: serde_json::Value,
}

/// Why a breakdown could not be written or read back.
#[derive(Debug, thiserror::Error)]
pub enum BreakdownFileError {
    /// The file could not be written or read.
    #[error("breakdown file: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not JSON, or does not hold what its header says.
    #[error("breakdown file is malformed: {0}")]
    Malformed(#[from] serde_json::Error),
    /// The file is not a breakdown.
    #[error("not a breakdown file: its format is `{found}`")]
    NotABreakdown {
        /// The format the file declares, empty for a file that declares none.
        found: String,
    },
    /// The file is a breakdown in a version this build does not read.
    #[error(
        "breakdown file version {found} is not supported; this build reads version {supported}"
    )]
    UnsupportedVersion {
        /// The version the file declares.
        found: u64,
        /// The version this build reads, [`BREAKDOWN_FILE_VERSION`].
        supported: u32,
    },
}

impl Breakdown {
    /// Write this breakdown as JSON, under a header naming the format and its
    /// [`BREAKDOWN_FILE_VERSION`], so a pass collected on one machine can be read on another.
    pub fn write(&self, writer: impl Write) -> Result<(), BreakdownFileError> {
        let file = FileOut {
            format: FORMAT,
            version: BREAKDOWN_FILE_VERSION,
            breakdown: self,
        };
        // Buffered, so a file handed in as is is not written a token at a time; flushed here so
        // a failed write is an error rather than lost in a drop.
        let mut writer = BufWriter::new(writer);
        serde_json::to_writer_pretty(&mut writer, &file)?;
        writer.flush()?;
        Ok(())
    }

    /// Read back a breakdown [`write`](Self::write) wrote, refusing a file of another format or
    /// version before decoding its contents, and one whose contents are not exactly what that
    /// version writes: a field missing or one it does not know.
    pub fn read(mut reader: impl Read) -> Result<Self, BreakdownFileError> {
        // Read whole rather than parsed from the reader, which would read a file a byte at a
        // time.
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let file: FileIn = serde_json::from_slice(&bytes)?;
        let format = file.format.unwrap_or_default();
        if format != FORMAT {
            return Err(BreakdownFileError::NotABreakdown { found: format });
        }
        let Some(version) = file.version else {
            return Err(<serde_json::Error as serde::de::Error>::missing_field("version").into());
        };
        if version != u64::from(BREAKDOWN_FILE_VERSION) {
            return Err(BreakdownFileError::UnsupportedVersion {
                found: version,
                supported: BREAKDOWN_FILE_VERSION,
            });
        }
        Ok(Breakdown::deserialize(file.breakdown)?)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::breakdown::{Component, Kernel, Unplaced, Variant};
    use crate::{Bounds, ComponentPath};

    /// A breakdown written and read back is the breakdown that was written, so one collected on a
    /// device can be read on a machine without it.
    #[test]
    fn a_written_breakdown_reads_back_unchanged() {
        let breakdown = breakdown();
        let read = Breakdown::read(written(&breakdown).as_bytes()).expect("read back");
        assert_eq!(read, breakdown);
    }

    /// The format version 1 is, field for field. A change that fails this test is a new
    /// [`BREAKDOWN_FILE_VERSION`], not an edit to the expected text.
    #[test]
    fn version_one_is_pinned() {
        let expected = r#"{
  "format": "burn-instrument breakdown",
  "version": 1,
  "breakdown": {
    "tree": [
      {
        "label": "attention",
        "variants": [
          {
            "name": "flash",
            "entries": 2
          }
        ],
        "sites": [
          "model::attention::forward"
        ],
        "bounds": {
          "read": 64,
          "write": 16,
          "flops": 128
        },
        "observed": null,
        "entries": 2,
        "children": [],
        "kernels": [
          {
            "name": "matmul",
            "launches": 2,
            "total": 1500,
            "fuses": [
              [
                "query"
              ]
            ],
            "fuses_own": true
          }
        ],
        "waited": 20
      }
    ],
    "untagged": [],
    "end_of_pass": [],
    "waited_outside": 7,
    "host_timed": false,
    "timed_alone": false,
    "unplaced": {
      "timings": 0,
      "total": 0
    },
    "unfused": false
  }
}"#;
        assert_eq!(written(&breakdown()), expected);
        assert_eq!(
            Breakdown::read(expected.as_bytes()).expect("version 1 reads"),
            breakdown(),
            "and reads back as the breakdown it pins"
        );
    }

    /// A file is read as its version writes it or not at all: a field this build does not know
    /// would be dropped unseen, and a missing one read as a value nobody wrote.
    #[test]
    fn a_file_unlike_what_its_version_writes_is_refused() {
        let pinned = written(&breakdown());
        let unknown = pinned.replace(r#""unfused": false"#, r#""unfused": false, "extra": 1"#);
        let missing = pinned.replace(r#""observed": null,"#, "");
        assert_ne!(unknown, pinned);
        assert_ne!(missing, pinned);

        assert!(matches!(
            Breakdown::read(unknown.as_bytes()),
            Err(BreakdownFileError::Malformed(_))
        ));
        assert!(matches!(
            Breakdown::read(missing.as_bytes()),
            Err(BreakdownFileError::Malformed(_))
        ));
    }

    /// The header is told apart before anything else: a file declaring no format is not a
    /// breakdown, and a version wider than this build's is one it does not read.
    #[test]
    fn a_header_is_refused_for_what_it_says() {
        assert!(matches!(
            Breakdown::read(r#"{"version": 1}"#.as_bytes()),
            Err(BreakdownFileError::NotABreakdown { found }) if found.is_empty()
        ));
        let wide = r#"{"format": "burn-instrument breakdown", "version": 4294967296}"#;
        assert!(matches!(
            Breakdown::read(wide.as_bytes()),
            Err(BreakdownFileError::UnsupportedVersion {
                found: 4_294_967_296,
                supported: 1
            })
        ));
    }

    /// A file of another version is refused by its header, before contents this build may not know
    /// how to decode.
    #[test]
    fn another_version_is_refused_before_its_contents_are_decoded() {
        let file = r#"{"format": "burn-instrument breakdown", "version": 2, "breakdown": "?"}"#;
        assert!(matches!(
            Breakdown::read(file.as_bytes()),
            Err(BreakdownFileError::UnsupportedVersion {
                found: 2,
                supported: 1
            })
        ));
    }

    /// A file that is not a breakdown is refused by its header rather than misread as one.
    #[test]
    fn another_format_is_refused() {
        let file = r#"{"format": "something else", "version": 1}"#;
        assert!(matches!(
            Breakdown::read(file.as_bytes()),
            Err(BreakdownFileError::NotABreakdown { found }) if found == "something else"
        ));
    }

    /// A breakdown with one of everything the file carries.
    fn breakdown() -> Breakdown {
        Breakdown {
            tree: vec![Component {
                label: "attention".to_string(),
                variants: vec![Variant {
                    name: "flash".to_string(),
                    entries: 2,
                }],
                sites: vec!["model::attention::forward".to_string()],
                bounds: Some(Bounds {
                    read: 64,
                    write: 16,
                    flops: 128,
                }),
                observed: None,
                entries: 2,
                children: Vec::new(),
                kernels: vec![Kernel {
                    name: "matmul".to_string(),
                    launches: 2,
                    total: Duration::from_nanos(1_500),
                    fuses: vec![ComponentPath::new(["query"])],
                    fuses_own: true,
                }],
                own_wait: Duration::from_nanos(20),
            }],
            untagged: Vec::new(),
            end_of_pass: Vec::new(),
            outside_wait: Duration::from_nanos(7),
            host_timed: false,
            timed_alone: false,
            unplaced: Unplaced::default(),
            unfused: false,
        }
    }

    fn written(breakdown: &Breakdown) -> String {
        let mut bytes = Vec::new();
        breakdown.write(&mut bytes).expect("written");
        String::from_utf8(bytes).expect("JSON is UTF-8")
    }
}
