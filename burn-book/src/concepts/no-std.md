## Support for `no_std`

Burn, including its `burn-ndarray` backend, can work in a `no_std` environment, provided `alloc` is
available for the inference mode. To accomplish this, simply turn off the default features in `burn`
and `burn-ndarray` (which is the minimum requirement for running the inference mode). You can find a
reference example in
[burn-no-std-tests](https://github.com/burn-rs/burn/tree/main/burn-no-std-tests).

The `burn-core` and `burn-tensor` crates also support `no_std` with `alloc`. These crates can be
directly added as dependencies if necessary, as they are reexported by the `burn` crate.

Please be aware that when using the `no_std` mode, a random seed will be generated at build time if
one hasn't been set using the `Backend::seed` method. Also, the
[spin::mutex::Mutex](https://docs.rs/spin/latest/spin/mutex/struct.Mutex.html) is used instead of
[std::sync::Mutex](https://doc.rust-lang.org/std/sync/struct.Mutex.html) in this mode.