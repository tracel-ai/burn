# Burn Benchmark

This crate is used with `cargo bench --features <backend>`
to compare backend computation times, from tensor operations to complex models.

Note: in order to compare different backend-specific tensor operation
implementations (for autotuning purposes, for instance), this should be done
within the corresponding backend crate.
