# Debugging Tips

This page provides techniques and tools to help you diagnose and fix issues when working with Burn.

## Macro Debugging

When working with macros in Burn, especially those related to tensor and module operations, you might encounter cryptic error messages. Here's how to get better diagnostics:

```bash
# Enable advanced macro debugging
RUSTC_BOOTSTRAP=1 RUSTFLAGS="-Zmacro-backtrace" cargo run-checks
```

This command reveals the complete macro expansion process, helping you pinpoint exactly where and why a macro is failing. Particularly useful for:

- Module operation debugging
- Complex derive macro issues
- Tracking down subtle syntax errors in macro invocations
