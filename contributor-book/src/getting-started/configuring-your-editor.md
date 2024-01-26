# Configuring your editor

These are not required, and most of this isn't specific to Burn, but it's definitely helpful if you haven't already done it.

## VSCode

1. Install the following extensions:

- [rust-lang.rust-analyzer](https://marketplace.visualstudio.com/items?itemName=rust-lang.rust-analyzer)
- [tamasfe.even-better-toml](https://marketplace.visualstudio.com/items?itemName=tamasfe.even-better-toml)
- [serayuzgur.crates](https://marketplace.visualstudio.com/items?itemName=serayuzgur.crates)
- [vadimcn.vscode-lldb](https://marketplace.visualstudio.com/items?itemName=vadimcn.vscode-lldb)

2. Open `Command Palette` with Ctrl+Shift+P or F1 and type `LLDB: Generate Launch Configurations from Cargo.toml` then select it, this will generate a file that should be saved as `.vscode/launch.json`.

3. Now you can enable breakpoint on code through IDE and then start debugging the library/binary you want, such as the following example:

<div align="center">
<img src="./assets/debug-options-vscode.png" width="700px"/>
<div align="left">

4. If you're creating a new library or binary, keep in mind to repeat the step 2 to always keep a fresh list of targets.

## Have another editor? Open a PR!