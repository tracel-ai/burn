# Contributing to Burn

Welcome to the Burn community! We're glad you're interested in contributing.

## How to Contribute

The best way to get started is to look at [open issues](https://github.com/tracel-ai/burn/issues)
and find one that interests you. Issues labeled `good first issue` are a great starting point for
new contributors.

If you have an idea that isn't covered by an existing issue, open one first to discuss the approach.
This helps align expectations and avoids wasted effort on both sides.

For questions, discussions, or just to say hello, join us on
[Discord](https://discord.gg/uPEBbYYDB6). The [Contributor Book](https://burn.dev/contributor-book/)
covers architecture, environment setup, and guides for common tasks.

## Change Ownership

The core principle behind all contributions: **PR authors must understand, justify, and explain
every change they propose.** After a PR is accepted, both the reviewer and the author should be
confident it improves the codebase.

This applies equally whether you wrote the code from scratch, adapted it from another project, or
used AI tools to help generate it. The origin of the code doesn't matter; what matters is that you
own it intellectually and can stand behind it during review.

## AI-Assisted Contributions

Using AI coding tools (Copilot, Cursor, Claude, ChatGPT, etc.) is fine. Many contributors use them,
and we don't ban or discourage their use.

That said, the [Change Ownership](#change-ownership) principle applies fully. You are the author,
not your AI tool. This means:

- Read and understand every line before submitting.
- Review AI-generated code for correctness, style consistency, and relevance.
- Test your changes locally and confirm they work as intended.
- Be prepared to explain the rationale behind any change during review.

Do not use "AI generated" as a justification for low-quality code.

## Before You Open a PR

1. **Check for an existing issue.** If there isn't one, open an issue first to discuss the approach.
   This is especially important for large changes or refactors.
2. **Read the codebase.** Understand the architecture and conventions already in place. The
   [Contributor Book](https://burn.dev/contributor-book/) covers architecture, environment setup,
   and guides for common tasks.
3. **Keep it focused.** One PR should address one concern. If you spot an unrelated issue while
   working, open a separate PR for it.
4. **Run validation.** Run `cargo run-checks` before submitting. This runs formatting, linting, and
   the full test suite. All checks must pass.

## PR Requirements

Every pull request should include:

- **A descriptive title** that summarizes the change.
- **A description** covering what you changed, why, how you tested it, and a link to the relevant
  issue.
- **Passing CI checks.** Please don't ask reviewers to look at a red build.
- **Minimal scope.** Avoid bundling unrelated changes together.

## Code Quality Standards

- Follow existing code style and project conventions.
- Write idiomatic Rust. If you are new to the codebase, study existing patterns before contributing.
- Keep dependencies minimal. Don't introduce new crates without discussion.
- Document public APIs. Non-trivial logic should have comments explaining _why_, not just _what_.
- Prefer clarity over cleverness.
- Bug fixes should include a regression test.

## Large Pull Requests

Large, complex PRs are harder to review effectively and carry more risk. To help both yourself and
reviewers, consider breaking substantial changes into smaller, incremental PRs. Each should be
valuable on its own, even if the full picture spans multiple PRs.

If you're planning a large effort, open an issue or start a discussion first so we can align on the
approach before you invest too much time.

## Review Process

- Maintainers review PRs as time allows. Please be patient.
- Be responsive to feedback. If changes are requested, address them or explain your reasoning.
- Reviewers may ask clarifying questions about any part of your PR. This is a normal part of
  collaborative review and helps ensure shared understanding.
- Don't force-push to rewrite history during an active review without notice.
- If a PR goes stale for more than 14 days without a response from the author, it may be closed.

## Getting Help

If you're stuck or unsure about something, don't hesitate to ask. Open an issue, start a discussion,
or reach out on [Discord](https://discord.gg/uPEBbYYDB6). We're happy to help.
