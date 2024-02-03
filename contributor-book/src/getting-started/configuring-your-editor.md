# Configuring your editor

These are not required, and most of this isn't specific to Burn, but it's definitely helpful if you
haven't already done it.

## VSCode

Install the following extensions:

- [rust-lang.rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
- [tamasfe.even-better-toml](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)
- [serayuzgur.crates](https://marketplace.visualstudio.com/items?itemName=serayuzgur.crates)
- [vadimcn.vscode-lldb](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)

### Setting up the Debugger

To use the debugger, follow these steps:
1. Open `Command Palette` with `Ctrl+Shift+P` or `F1` and type `LLDB: Generate Launch Configurations from Cargo.toml` then select it, this will generate a file that should be saved as `.vscode/launch.json`.
2. Select the configuration from the "run and debug" side panel (it have a infested play button), then select the target from
3. Now you can enable breakpoint on code through IDE and then start debugging the library/binary you want, such as the following example:


![debug-options](debug-options-vscode.png)


If you're creating a new library or binary, keep in mind to repeat the step 1. to always keep a fresh list of targets.

## Have another editor? Open a PR!
