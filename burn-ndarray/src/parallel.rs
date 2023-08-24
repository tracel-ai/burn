/// Macro for running a function in parallel.
#[macro_export(local_inner_macros)]
macro_rules! run_par {
    (
        $func:expr
    ) => {{
        #[cfg(feature = "std")]
        use rayon::prelude::*;

        #[cfg(feature = "std")]
        #[allow(clippy::redundant_closure_call)]
        let output = rayon::scope(|_| $func());

        #[cfg(not(feature = "std"))]
        let output = $func();

        output
    }};
}

/// Macro for iterating over a range in parallel.
#[macro_export(local_inner_macros)]
macro_rules! iter_par {
    (
        $start:expr, $end:expr
    ) => {{
        #[cfg(feature = "std")]
        let output = ($start..$end).into_par_iter();

        #[cfg(not(feature = "std"))]
        let output = ($start..$end);

        output
    }};
}
