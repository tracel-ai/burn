# setting up the environment

There are a couple of tools that need to be installed, and commands to be familiar with, depending on what part of the project you plan on contributing to. This section should be up to date with current project practices (as of 2024-01-26)

## General

there are a few commands you want to run prior to any commit for a non-draft PR:

1. `cargo clippy --fix --allow-dirty`, this will run clippy and fix any issues it can, the allow dirty flag is required whenever you have uncommitted changes
2. `cargo fmt --all`, this will run rustfmt on all files in the project
3. `./run_checks.sh all`, this is a script located in the project root that builds and tests the project. It is required that this pass prior to merging a PR. Fair warning, running these tests can take a while[^2].

## Updating the burn semver version

To bump for the next version, install `cargo-edit` if its not on your system, and use this command:

```sh
cargo set-version --bump minor
```

## Contributing to the Burn (Developer) Book

Both the Burn book and the burn developer book are built with mdbook. To install mdbook, run the following command[^1]:

```bash
cargo install mdbook
```


[^1]: You might also want to install [cargo-update](https://github.com/nabijaczleweli/cargo-update) to easily keep your tools up to date, though it is in no way required.
[^2]: if your system is running into issues with memory and you are on linux  you may want to switch to a [virtual console](https://wiki.archlinux.org/title/Linux_console#Virtual_consoles) to run the tests. To do this, press `ctrl+alt+f3` to switch to a virtual console(and log in), and either `ctrl+alt+f2` or `ctrl+alt+f1` to switch back to your graphical session.