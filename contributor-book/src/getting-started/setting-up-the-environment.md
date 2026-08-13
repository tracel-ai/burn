# Setting up the environment

Depending on what part of the project you plan on contributing to, there are a couple of tools to
install and commands to be familiar with.

## General

During development, these commands can automatically address common formatting and lint issues:

1. `cargo fmt --all` runs `rustfmt` on all files in the project.
2. `cargo clippy --fix` runs [Clippy](https://github.com/rust-lang/rust-clippy) and applies fixes
   for supported lint findings. It requires a clean Git state unless you pass `--allow-dirty`.

Before submitting a PR, run:

```bash
cargo run-checks
```

This is the common local validation command. It runs, in order:

- formatting checks;
- typo checks;
- a dependency audit;
- Clippy across the workspace;
- a quick host no-std compilation check; and
- backend tests in release mode using the Flex backend.

If your changes target another backend, override the default:

```bash
cargo run-checks --backend <backend>
```

The command is intended as a fast common baseline rather than the entire CI test matrix. You should
also run tests relevant to the crates changed. CI runs the broader workspace, documentation,
platform, feature, and backend combinations.

> Want more detailed macro error diagnostics? This is especially useful for debugging tensor-related
> tests:
>
> ```bash
> RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zmacro-backtrace" cargo run-checks
> ```

## Updating the burn semver version

If for some reason you need to bump for the next version (though that should probably be left to the
maintainers), edit the semantic version number in `burn/Cargo.toml`, and then run `cargo update` to
update the lock file.

## Contributing to either the Burn Book or Contributor Book

Both the Burn Book and the Contributor Book are built with mdbook. To open the book locally, run
`mdbook serve <path/to/book>` or `cargo xtask books {burn|contributor} open` which will install and
use mdbook automatically.

Alternatively, if you want to install mdbook directly, run the following command[^update_note]:

```bash
cargo install mdbook
```

For documentation-only changes, you can run `cargo xtask check typos` to check for misspellings
without running the full local validation. This installs
[`typos`](https://crates.io/crates/typos-cli) when needed. To apply suggested corrections to a book,
run `typos -w /path/to/book`.

[^update_note]:
    You might also want to install [cargo-update](https://github.com/nabijaczleweli/cargo-update) to
    easily keep your tools up to date, though it is in no way required.
