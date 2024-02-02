# setting up the environment

There are a couple of tools that need to be installed, and commands to be familiar with, depending
on what part of the project you plan on contributing to. This section should be up to date with
current project practices (as of 2024-01-26)

## General

There are a few commands you want to run prior to any commit for a non-draft PR:

1. `cargo clippy --fix --allow-dirty`, this will run clippy and fix any issues it can, the allow
   dirty flag is required whenever you have uncommitted changes
2. `cargo fmt --all`, this will run rustfmt on all files in the project
3. `./run_checks.sh all`, this is a script located in the project root that builds and tests the
   project. It is required that this passes prior to merging a PR. Fair warning, running these tests
   can take a while[^linux_mem_note].

## Updating the burn semver version

If for some reason you need to bump for the next version (though that should probably be left to the maintainers), edit the semantic version number in `burn/Cargo.toml`, and then run
`cargo update` to update the lock file.

## Contributing to either the Burn Book or Contributor Book


Both the Burn Book and the Contributor Book are built with mdbook. If in the process of adding or modifying a page in the books, if you need to inspect the generated output(such as when using mathjax which seems prone to breakage), run use `mdbook --open <path/to/book>` or run `cargo xtask books {burn|contributor} open` which will install and use mdbook automatically.

Alternatively, if you want to install mdbook directly, run the
following command[^update_note]:

```bash
cargo install mdbook
```

Also instead of running `./run_checks.sh all`, you can run `./run_checks.sh typo`, or `cargo xtasks run-checks typo`, to only check for
misspellings. This will install [typo](https://crates.io/crates/typos-cli), and if any are
encountered you should be able to run `typo -w /path/to/book` to fix them.

[^linux_mem_note]: If your system is running into issues with memory and you are on linux, you may want to switch to a [virtual console](https://wiki.archlinux.org/title/Linux_console#Virtual_consoles) to run the tests. To do this, press `ctrl+alt+f3` to switch to a virtual console (and log in), and either `ctrl+alt+f2` or `ctrl+alt+f1` to switch back to your graphical session.

[^update_note]: You might also want to install [cargo-update](https://github.com/nabijaczleweli/cargo-update) to easily keep your tools up to date, though it is in no way required.
