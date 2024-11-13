# Configuring your editor

These steps are not required, and most of this isn't specific to Burn, but it's definitely helpful
if you haven't already done it.

## VSCode

Install the following extensions:

- [rust-lang.rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
  for Rust syntax and semantic analysis
- [tamasfe.even-better-toml](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)
  for TOML syntax and semantic analysis
- [fill-labs.dependi](https://marketplace.visualstudio.com/items?itemName=fill-labs.dependi) for
  managing dependencies
- [vadimcn.vscode-lldb](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb) for
  debugging

### Setting up the Debugger

To use the debugger, follow these steps:

1. Open `Command Palette` with `Ctrl+Shift+P` or `F1` and type
   `LLDB: Generate Launch Configurations from Cargo.toml` then select it, this will generate a file
   that should be saved as `.vscode/launch.json`.
2. Select the configuration from the "run and debug" side panel, then select the target from the list.
   Since this repo has `debug = 0` in the root `Cargo.toml` to speed up compilation, you need replace it with `debug = true` in the root `Cargo.toml` when using a debugger and breakpoints with `launch.json` settings.
3. Now you can enable breakpoints on code through IDE then start debugging the library/binary you
   want, like in the following example:

![debug-options](debug-options-vscode.png)

If you're creating a new library or binary, keep in mind to repeat step 1 to always keep a fresh
list of targets.

## Have another editor? Open a PR!
