use std::path::PathBuf;

/// Workload family selected in the UI and passed to the runner.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ProblemKind {
    Matmul,
    Attention,
}

impl ProblemKind {
    pub const ALL: [Self; 2] = [Self::Matmul, Self::Attention];

    pub fn label(self) -> &'static str {
        match self {
            Self::Matmul => "Matmul",
            Self::Attention => "Attention",
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::Attention => "attention",
        }
    }

    pub fn from_str(value: &str) -> Result<Self, String> {
        match value {
            "matmul" => Ok(Self::Matmul),
            "attention" => Ok(Self::Attention),
            other => Err(format!("unknown problem '{other}'")),
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

#[derive(Clone, Copy)]
pub(crate) struct MatmulShape {
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
}
