//! A collected pass as a file: versioned JSON that reads back on any machine, with or without a
//! runtime.

use std::io::{BufWriter, Read, Write};
use std::time::Duration;

use serde::de::Error as _;
use serde::{Deserialize, Serialize};

use super::Breakdown;
use crate::Peaks;

/// What a pass file says it is, so a reader can refuse anything else before decoding it.
const FORMAT: &str = "burn-instrument pass";

/// The version of the pass file format this build writes and reads.
///
/// Every change to what a file serializes as (a field added, removed, renamed or re-encoded) is a
/// new version, so a saved pass is read back as it was collected or not at all.
pub const PASS_FILE_VERSION: u32 = 2;

/// A collected pass and what it is scored against: what a pass file holds.
///
/// The breakdown alone says where the device time went. Scoring it needs the device's
/// [`Peaks`], and splitting a pass into the device's time and the host's needs the same pass run
/// with nothing collecting it; both are measured where the device is, so they travel with the
/// breakdown rather than being measured again wherever the file is read.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectedPass {
    /// The device the pass ran on, as its runtime names it.
    pub device: String,
    /// The device's peaks, measured before the pass, or `None` where it was not probed.
    pub peaks: Option<Peaks>,
    /// The wall time of the same pass run with nothing collecting it, where it was measured: the
    /// time a report splits into the device's and the host's.
    ///
    /// A collected pass times every launch, which costs the host a round trip per kernel, so its
    /// own wall time describes the instrumentation; this is the one that describes the model.
    pub clean_wall: Option<Duration>,
    /// The collected pass.
    pub breakdown: Breakdown,
}

/// A pass file as it is written.
#[derive(Serialize)]
struct FileOut<'a> {
    format: &'static str,
    version: u32,
    device: &'a str,
    peaks: Option<&'a Peaks>,
    #[serde(with = "super::nanoseconds::optional")]
    clean_wall: Option<Duration>,
    breakdown: &'a Breakdown,
}

/// A pass file as it is read: the header, and every other field parsed but not yet decoded,
/// so a file of another format or version is refused before its contents are.
///
/// Every field may be missing here, and is checked in [`CollectedPass::read`]: a file with no
/// format is not a pass, a version too wide for this build's is still one it does not read,
/// and a field the version writes is never read as a default nobody wrote.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIn {
    #[serde(default)]
    format: Option<String>,
    #[serde(default)]
    version: Option<u64>,
    #[serde(default, deserialize_with = "present")]
    device: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "present")]
    peaks: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "present")]
    clean_wall: Option<serde_json::Value>,
    #[serde(default, deserialize_with = "present")]
    breakdown: Option<serde_json::Value>,
}

/// Why a pass file could not be written or read back.
#[derive(Debug, thiserror::Error)]
pub enum PassFileError {
    /// The file could not be written or read.
    #[error("pass file: {0}")]
    Io(#[from] std::io::Error),
    /// The file is not JSON, or does not hold what its header says.
    #[error("pass file is malformed: {0}")]
    Malformed(#[from] serde_json::Error),
    /// The file is not a pass.
    #[error("not a pass file: its format is `{found}`")]
    NotAPass {
        /// The format the file declares, empty for a file that declares none.
        found: String,
    },
    /// The file is a pass in a version this build does not read.
    #[error("pass file version {found} is not supported; this build reads version {supported}")]
    UnsupportedVersion {
        /// The version the file declares.
        found: u64,
        /// The version this build reads, [`PASS_FILE_VERSION`].
        supported: u32,
    },
}

impl CollectedPass {
    /// Write this pass as JSON, under a header naming the format and its
    /// [`PASS_FILE_VERSION`], so a pass collected on one machine can be read on another.
    pub fn write(&self, writer: impl Write) -> Result<(), PassFileError> {
        let file = FileOut {
            format: FORMAT,
            version: PASS_FILE_VERSION,
            device: &self.device,
            peaks: self.peaks.as_ref(),
            clean_wall: self.clean_wall,
            breakdown: &self.breakdown,
        };
        // Buffered, so a file handed in as is is not written a token at a time; flushed here so
        // a failed write is an error rather than lost in a drop.
        let mut writer = BufWriter::new(writer);
        serde_json::to_writer_pretty(&mut writer, &file)?;
        writer.flush()?;
        Ok(())
    }

    /// Read back a pass [`write`](Self::write) wrote, refusing a file of another format or
    /// version before decoding its contents, and one whose contents are not exactly what that
    /// version writes: a field missing or one it does not know.
    pub fn read(mut reader: impl Read) -> Result<Self, PassFileError> {
        // Read whole rather than parsed from the reader, which would read a file a byte at a
        // time.
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let file: FileIn = serde_json::from_slice(&bytes)?;
        let format = file.format.unwrap_or_default();
        if format != FORMAT {
            return Err(PassFileError::NotAPass { found: format });
        }
        let version = required(file.version, "version")?;
        if version != u64::from(PASS_FILE_VERSION) {
            return Err(PassFileError::UnsupportedVersion {
                found: version,
                supported: PASS_FILE_VERSION,
            });
        }
        Ok(Self {
            device: serde_json::from_value(required(file.device, "device")?)?,
            peaks: serde_json::from_value(required(file.peaks, "peaks")?)?,
            clean_wall: super::nanoseconds::optional::deserialize(required(
                file.clean_wall,
                "clean_wall",
            )?)?,
            breakdown: serde_json::from_value(required(file.breakdown, "breakdown")?)?,
        })
    }
}

/// A field that is in the file, `null` included: serde reads a `null` into an `Option` as
/// `None`, which would make a pass written unprobed indistinguishable from a file missing the
/// field.
fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error> {
    serde_json::Value::deserialize(deserializer).map(Some)
}

/// A field the version writes, or the error for a file that left it out.
fn required<T>(field: Option<T>, name: &'static str) -> Result<T, PassFileError> {
    field.ok_or_else(|| serde_json::Error::missing_field(name).into())
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use burn_tensor::FloatDType;

    use super::*;
    use crate::breakdown::{Component, Kernel, Unplaced, Variant};
    use crate::{Bounds, ComponentPath, ComputePeak};

    /// A pass written and read back is the pass that was written, so one collected on a device
    /// can be read on a machine without it.
    #[test]
    fn a_written_pass_reads_back_unchanged() {
        let pass = pass();
        let read = CollectedPass::read(written(&pass).as_bytes()).expect("read back");
        assert_eq!(read, pass);
    }

    /// What a pass was not measured against reads back as not measured, rather than as a zero
    /// a report would divide by.
    #[test]
    fn an_unprobed_pass_reads_back_unprobed() {
        let pass = CollectedPass {
            peaks: None,
            clean_wall: None,
            ..pass()
        };
        let read = CollectedPass::read(written(&pass).as_bytes()).expect("read back");
        assert_eq!(read.peaks, None);
        assert_eq!(read.clean_wall, None);
    }

    /// The format version 2 is, field for field. A change that fails this test is a new
    /// [`PASS_FILE_VERSION`], not an edit to the expected text.
    #[test]
    fn version_two_is_pinned() {
        let expected = r#"{
  "format": "burn-instrument pass",
  "version": 2,
  "device": "Rocm(0)",
  "peaks": {
    "copy_bytes_per_s": 161000000000.0,
    "read_bytes_per_s": 199000000000.0,
    "write_bytes_per_s": 137000000000.0,
    "flops": {
      "per_s": 49300000000000.0,
      "dtype": "F16"
    }
  },
  "clean_wall": 2000,
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
        assert_eq!(written(&pass()), expected);
        assert_eq!(
            CollectedPass::read(expected.as_bytes()).expect("version 2 reads"),
            pass(),
            "and reads back as the pass it pins"
        );
    }

    /// A file is read as its version writes it or not at all: a field this build does not know
    /// would be dropped unseen, and a missing one read as a value nobody wrote.
    #[test]
    fn a_file_unlike_what_its_version_writes_is_refused() {
        let pinned = written(&pass());
        let unknown = pinned.replace(r#""unfused": false"#, r#""unfused": false, "extra": 1"#);
        let missing = pinned.replace(r#""observed": null,"#, "");
        let no_peaks = pinned.replace(
            &pinned[pinned.find(r#""peaks""#).unwrap()..pinned.find(r#""clean_wall""#).unwrap()],
            "",
        );
        for file in [&unknown, &missing, &no_peaks] {
            assert_ne!(file, &pinned);
            assert!(
                matches!(
                    CollectedPass::read(file.as_bytes()),
                    Err(PassFileError::Malformed(_))
                ),
                "{file}"
            );
        }
    }

    /// The header is told apart before anything else: a file declaring no format is not a
    /// pass, and a version wider than this build's is one it does not read.
    #[test]
    fn a_header_is_refused_for_what_it_says() {
        assert!(matches!(
            CollectedPass::read(r#"{"version": 2}"#.as_bytes()),
            Err(PassFileError::NotAPass { found }) if found.is_empty()
        ));
        let wide = r#"{"format": "burn-instrument pass", "version": 4294967296}"#;
        assert!(matches!(
            CollectedPass::read(wide.as_bytes()),
            Err(PassFileError::UnsupportedVersion {
                found: 4_294_967_296,
                supported: 2
            })
        ));
    }

    /// A file of another version is refused by its header, before contents this build may not
    /// know how to decode: version 1 carried no device and no peaks.
    #[test]
    fn another_version_is_refused_before_its_contents_are_decoded() {
        let file = r#"{"format": "burn-instrument pass", "version": 1, "breakdown": "?"}"#;
        assert!(matches!(
            CollectedPass::read(file.as_bytes()),
            Err(PassFileError::UnsupportedVersion {
                found: 1,
                supported: 2
            })
        ));
    }

    /// A file that is not a pass is refused by its header rather than misread as one.
    #[test]
    fn another_format_is_refused() {
        let file = r#"{"format": "something else", "version": 2}"#;
        assert!(matches!(
            CollectedPass::read(file.as_bytes()),
            Err(PassFileError::NotAPass { found }) if found == "something else"
        ));
    }

    /// A pass with one of everything the file carries.
    fn pass() -> CollectedPass {
        CollectedPass {
            device: "Rocm(0)".to_string(),
            peaks: Some(Peaks {
                copy_bytes_per_s: Some(161.0e9),
                read_bytes_per_s: Some(199.0e9),
                write_bytes_per_s: Some(137.0e9),
                flops: Some(ComputePeak {
                    per_s: 49.3e12,
                    dtype: FloatDType::F16,
                }),
            }),
            clean_wall: Some(Duration::from_nanos(2_000)),
            breakdown: Breakdown {
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
            },
        }
    }

    fn written(pass: &CollectedPass) -> String {
        let mut bytes = Vec::new();
        pass.write(&mut bytes).expect("written");
        String::from_utf8(bytes).expect("JSON is UTF-8")
    }
}
