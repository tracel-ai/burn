use std::path::PathBuf;

/// A message streamed from the worker thread running the `runner` subprocess.
pub(crate) enum RunMsg {
    Line(String),
    Done { ok: bool },
}

/// One archived run: its directory, parsed events, and whether it is currently shown.
pub(crate) struct RunView {
    pub(crate) name: String,
    pub(crate) dir: PathBuf,
    pub(crate) events: Vec<crate::TuneEvent>,
    pub(crate) selected: bool,
}

#[derive(Clone, Copy)]
pub(crate) struct MatmulShape {
    pub(crate) m: usize,
    pub(crate) k: usize,
    pub(crate) n: usize,
}
