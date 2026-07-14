use std::path::PathBuf;

/// Workload family selected in the UI and passed to the runner.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    Matmul,
    Attention,
    FlashAttention,
    Reduce,
}

impl ProblemKind {
    pub const ALL: [Self; 4] = [Self::Matmul, Self::Attention, Self::FlashAttention, Self::Reduce];

    pub fn label(self) -> &'static str {
        match self {
            Self::Matmul => "Matmul",
            Self::Attention => "Attention",
            Self::FlashAttention => "Flash Attention",
            Self::Reduce => "Reduce",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Attention => "attention",
            Self::FlashAttention => "flash_attention",
            Self::Reduce => "reduce",
        }
    }

    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "matmul" => Ok(Self::Matmul),
            "attention" => Ok(Self::Attention),
            "flash_attention" => Ok(Self::FlashAttention),
            "reduce" => Ok(Self::Reduce),
            other => Err(format!("unknown problem '{other}'")),
        }
    }
    pub fn shape_labels(&self) -> &'static [&'static str] {
        match self {
            Self::Matmul => &["m", "k", "n"],
            Self::Attention => &["batch", "seq", "head"],
            Self::FlashAttention => &["batch", "seq", "head"],
            Self::Reduce => &["batch", "dim1", "dim2"],
        }
    }

    pub fn default_shape(&self) -> Vec<usize> {
        match self {
            Self::Matmul => vec![512, 512, 512],
            Self::Attention => vec![1, 256, 256],
            Self::FlashAttention => vec![1, 256, 256],
            Self::Reduce => vec![32, 512, 1024],
        }
    }
}

/// A message streamed from the worker thread running the `runner` subprocess.
pub(crate) enum RunMsg {
    /// A line appended to the console output.
    Line(String),
    /// A transient status update shown in place (e.g. sync progress), not appended to the log.
    Progress(String),
    Done { ok: bool },
}

/// One archived run: its directory, parsed events, and whether it is currently shown.
pub(crate) struct RunView {
    pub(crate) name: String,
    pub(crate) dir: PathBuf,
    pub(crate) events: Vec<crate::TuneEvent>,
    pub(crate) selected: bool,
    pub(crate) custom_name: Option<String>,
}
