# Nightly

Burn has nightly support by default. However, one may meet this error during compilation:

```
error[E0658]: use of unstable library feature `allocator_api`
```

To solve it, the `hashbrown` crate with version higher than `0.15` needs to be involved explicitly in `Cargo.toml`:

```toml
hashbrown = { version = "0.15", features = ["nightly"] }
```

even it's not required by your code.

See [here](https://github.com/rust-lang/hashbrown/issues/564) for track the issue and [here](https://github.com/zakarumych/allocator-api2/issues/19) for the status.