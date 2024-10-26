# Setting up the environment

Depending on what part of the project you plan on contributing to, there are a couple of tools to
install and commands to be familiar with. This section should be up to date with current project
practices (as of 2024-04-15).

## General

There are a few commands you will want to run prior to any commit for a non-draft PR:

1. `cargo fmt --all` will run `rustfmt` on all files in the project.
2. `cargo clippy --fix` will run [Clippy](https://github.com/rust-lang/rust-clippy) and fix any
   coding issues it can. Clippy necessitates to be in a clean Git state, but this can be
   circumvented by adding the `--allow-dirty` flag.
3. `cargo xtask check all` is a script located in the project root that builds and tests the
   project. It is required to run successfully prior to merging a PR. Fair warning, running these
   tests can take a while[^linux_mem_note].

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

Also instead of running `cargo xtask check all`, you can run `cargo xtask check typos` to
only check for misspellings. This will install [typo](https://crates.io/crates/typos-cli), and if
any are encountered you should be able to run `typo -w /path/to/book` to fix them.

[^linux_mem_note]: If your system is running into issues with memory and you are on linux, you may want to switch
    to a [virtual console](https://wiki.archlinux.org/title/Linux_console#Virtual_consoles) to run
    the tests. To do this, press `ctrl+alt+f3` to switch to a virtual console (and log in), and
    either `ctrl+alt+f1` or `ctrl+alt+f2` to switch back to your graphical session.

[^update_note]: You might also want to install [cargo-update](https://github.com/nabijaczleweli/cargo-update) to
    easily keep your tools up to date, though it is in no way required.
