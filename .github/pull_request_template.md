## Pull Request Template

### Checklist

- [ ] Confirmed that `cargo run-checks` command has been executed.
- [ ] Made sure the book is up to date with changes in this PR.


> [!TIP]
> Want more detailed macro error diagnostics? This is especially useful for debugging tensor-related tests:
>
> ```bash
> RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zmacro-backtrace" cargo run-checks
> ```

### Related Issues/PRs

_Provide links to relevant issues and dependent PRs._

### Changes

_Summarize the problem being addressed and your solution._

### Testing

_Describe how these changes have been tested._
