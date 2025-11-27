/// Macro for running a function in parallel.
#[cfg(feature = "multi-threads")]
#[macro_export(local_inner_macros)]
macro_rules! run_par {
    (
        $func:expr
    ) => {{
        use rayon::prelude::*;

        #[allow(clippy::redundant_closure_call)]
        rayon::scope(|_| $func())
    }};
}

/// Macro for running a function in parallel.
#[cfg(not(feature = "multi-threads"))]
#[macro_export(local_inner_macros)]
macro_rules! run_par {
    (
        $func:expr
    ) => {{ $func() }};
}

/// Macro for iterating in parallel.
#[cfg(not(feature = "multi-threads"))]
#[macro_export(local_inner_macros)]
macro_rules! iter_par {
    (
        $iter:expr
    ) => {{ $iter }};
}

/// Macro for iterating in parallel.
#[cfg(feature = "multi-threads")]
#[macro_export(local_inner_macros)]
macro_rules! iter_par {
    (
        $iter:expr
    ) => {{ $iter.into_par_iter() }};
}

/// Macro for iterating in parallel.
#[cfg(feature = "multi-threads")]
#[macro_export(local_inner_macros)]
macro_rules! iter_slice_par {
    (
        $slice:expr
    ) => {{ $slice.into_par_iter() }};
}

/// Macro for iterating in parallel.
#[cfg(not(feature = "multi-threads"))]
#[macro_export(local_inner_macros)]
macro_rules! iter_slice_par {
    (
        $slice:expr
    ) => {{ $slice.iter() }};
}

/// Macro for iterating over a range in parallel.
#[cfg(feature = "multi-threads")]
#[macro_export(local_inner_macros)]
macro_rules! iter_range_par {
    (
        $start:expr, $end:expr
    ) => {{ ($start..$end).into_par_iter() }};
}

/// Macro for iterating over a range in parallel.
#[cfg(not(feature = "multi-threads"))]
#[macro_export(local_inner_macros)]
macro_rules! iter_range_par {
    (
        $start:expr, $end:expr
    ) => {{ ($start..$end) }};
}
