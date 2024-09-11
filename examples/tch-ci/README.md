# CI Example

[This](https://github.com/tracel-ai/burn/tree/main/examples/tch-ci/release.yaml)
is a github action example for `burn` project with `tch` feature enabled for Linux, Windows, macOS.

Things the action does:
- creates a new release on github with the changelog
- retrieves setup scripts from dist for Linux, Windows, macOS users
- setup the environment and builds the project for Linux, Windows, macOS
- uploads the artifacts to the github release page

# Usage

- copy `CHANGELOG.md` file and `dist` folder to your project root
- copy `release.yaml` to `.github/workflows` folder and modify `NAME` `BIN1` `BIN2`
- allow actions and enable read and write permissions in your repository
- trigger the action by pushing a new tag

# Acknowledgements

- [setup burn env](https://github.com/tracel-ai/burn/tree/main/crates/burn-tch)
- [set up cuda action](https://github.com/Jimver/cuda-toolkit)
- [upload rust binary action](https://github.com/taiki-e/upload-rust-binary-action)
- [create github release](https://github.com/taiki-e/create-gh-release-action)
