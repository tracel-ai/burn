/// Macro for running a function in parallel.
#[macro_export(local_inner_macros)]
macro_rules! run_par {
    (
        $func:expr
    ) => {{
        #[cfg(feature = "rayon")]
        use rayon::prelude::*;

        #[cfg(feature = "rayon")]
        #[allow(clippy::redundant_closure_call)]
        let output = rayon::scope(|_| $func());

        #[cfg(not(feature = "rayon"))]
        let output = $func();

        output
    }};
}

/// Macro for iterating in parallel.
#[macro_export(local_inner_macros)]
macro_rules! iter_par {
    (
        $iter:expr
    ) => {{
        #[cfg(feature = "rayon")]
        let output = $iter.into_par_iter();

        #[cfg(not(feature = "rayon"))]
        let output = $iter;

        output
    }};
}

/// Macro for iterating over a range in parallel.
#[macro_export(local_inner_macros)]
macro_rules! iter_range_par {
    (
        $start:expr, $end:expr
    ) => {{
        #[cfg(feature = "rayon")]
        let output = ($start..$end).into_par_iter();

        #[cfg(not(feature = "rayon"))]
        let output = ($start..$end);

        output
    }};
}
