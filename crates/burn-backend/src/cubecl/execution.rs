//! The execution vocabulary in cubecl's words, for a backend that applies it
//! on cubecl.

use crate::{
    AutotuneStatistics, CompilationStatistics, ExecutionStatistics, ProcessMode, StreamMode,
};
use cubecl::execution as cube;

impl From<ProcessMode> for cube::ProcessMode {
    fn from(mode: ProcessMode) -> Self {
        match mode {
            ProcessMode::Execute => Self::Execute,
            ProcessMode::CompileOnly => Self::CompileOnly,
            ProcessMode::CompileAndAutotune => Self::CompileAndAutotune,
        }
    }
}

impl From<StreamMode> for cube::StreamMode {
    fn from(mode: StreamMode) -> Self {
        match mode {
            StreamMode::Execute => Self::Execute,
            StreamMode::Discard => Self::Discard,
        }
    }
}

impl From<cube::ExecutionStatistics> for ExecutionStatistics {
    fn from(statistics: cube::ExecutionStatistics) -> Self {
        let compilation = statistics.compilation;
        let autotune = statistics.autotune;
        Self {
            compilation: CompilationStatistics {
                registered: compilation.registered as u64,
                compiled: compilation.compiled as u64,
                loaded: compilation.loaded as u64,
                failed: compilation.failed as u64,
                abandoned: compilation.abandoned as u64,
                refused: compilation.refused as u64,
                stored: compilation.stored as u64,
            },
            autotune: AutotuneStatistics {
                registered: autotune.registered as u64,
                measured: autotune.measured as u64,
                loaded: autotune.loaded as u64,
                failed: autotune.failed as u64,
                abandoned: autotune.abandoned as u64,
                persisted: autotune.persisted as u64,
            },
        }
    }
}
