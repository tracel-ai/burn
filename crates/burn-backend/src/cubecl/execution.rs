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
                registered: compilation.registered_count as u64,
                compiled: compilation.compiled_count as u64,
                loaded: compilation.loaded_count as u64,
                failed: compilation.failed_count as u64,
                abandoned: compilation.abandoned_count as u64,
                refused: compilation.refused_count as u64,
                stored: compilation.stored_count as u64,
            },
            autotune: AutotuneStatistics {
                registered: autotune.registered_count as u64,
                measured: autotune.measured_count as u64,
                loaded: autotune.loaded_count as u64,
                failed: autotune.failed_count as u64,
                abandoned: autotune.abandoned_count as u64,
                persisted: autotune.persisted_count as u64,
            },
        }
    }
}
