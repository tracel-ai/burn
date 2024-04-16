# Burn Common

The `burn-common` package hosts code that _must_ be shared between burn packages (with `std` or
`no_std` enabled). No other code should be placed in this package unless unavoidable.

The package must build with `cargo build --no-default-features` as well.
