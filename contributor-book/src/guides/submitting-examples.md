# Submitting Examples to Burn

This guide explains how to create and submit new examples to the Burn repository. Examples are a great way to demonstrate Burn's capabilities and help users understand how to use the framework effectively.

For a minimal working example, see the [simple-regression](https://github.com/tracel-ai/burn/blob/main/examples/simple-regression/examples/regression.rs) example in the repository.

## Repository Structure

The Burn repository is set up as a workspace, with examples located in the `examples/` directory. Each example is a separate crate that can reuse workspace dependencies.

## Creating a New Example

1. Navigate to the examples directory:
   ```bash
   cd examples
   ```

2. Create a new library crate:
   ```bash
   cargo new --lib <my-example>
   ```

3. Update the example's `Cargo.toml`:
   ```toml
   [package]
   name = "<my-example>"
   version = "0.1.0"
   edition = "2021"
   readme = "README.md"
   # Remove this line if it exists
   # readme.workspace = true

   [dependencies]
   # Reuse workspace dependencies when available
   serde = { workspace = true }
   # Add example-specific dependencies
   burn = { path = "../../" }
   ```

## Required Files and Structure

### README.md
Each example must include a README.md file with:
- A brief description of what the example demonstrates
- A terminal command showing how to run the example
- Any prerequisites or setup instructions

Example README structure:
````markdown
# Example Name

Brief description of what this example demonstrates.

## Running the Example

```bash
cargo run -p <my-example> --example <my-example>
```

## Prerequisites

List any prerequisites here.
````

### Source Code Structure

- `src/` directory: Contains the main implementation code
- `examples/` directory: Contains example code
  - `<my-example>.rs`: Example implementation

## Resource Handling

- Resources (datasets, models, etc.) should be downloaded in the example code
- Do not track external files in the repository
- Include code to download and prepare resources when the example is run

## Best Practices

1. **Code Organization**
   - Keep the code modular and well-documented
   - Use clear, descriptive variable and function names
   - Include comments explaining complex operations

2. **Error Handling**
   - Implement proper error handling
   - Provide meaningful error messages
   - Handle resource download failures gracefully

3. **Performance**
   - Optimize for reasonable execution time
   - Include progress indicators for long-running operations
   - Consider adding configuration options for different hardware capabilities

4. **Documentation**
   - Document all public APIs
   - Include inline comments for complex logic
   - Explain any non-obvious implementation details

## Sharing Code with the Book

When a book chapter shows code from a maintained example, use an mdBook include so changes to the
example also update the chapter. Include the whole file when it is short, or select a named region
to preserve a tutorial's step-by-step presentation. Prefer named regions over line numbers, which
shift as the source changes.

Mark a region in the Rust source with ordinary comments:

```rust,ignore
// ANCHOR: example_model
#[derive(Module, Debug)]
pub struct Model {
    linear: Linear,
}
// ANCHOR_END: example_model
```

Inside the chapter's Rust code fence, reference the source path and region. For example, the
basic-workflow model chapter uses:

```text
\{{#include ../../../examples/guide/src/model.rs:model}}
```

Paths are relative to the Markdown file containing the include. The example above is relative to
`burn-book/src/basic-workflow/model.md`. mdBook removes the anchor comments from rendered snippets.
Keep conceptual or deliberately abbreviated snippets inline when the maintained implementation
would obscure the explanation. Update the surrounding prose when an included API changes.

Build both books from the repository root to check includes:

```sh
cargo xtask books burn build
cargo xtask books contributor build
```

These commands install mdBook if needed. To preview either book in your browser, replace `build`
with `open`.

An include keeps the displayed code synchronized with its source; compiling and testing the
example remains a separate check.

## Submitting Your Example

1. Ensure your example follows all the guidelines above
2. Test your example thoroughly
3. Create a pull request with:
   - A clear description of what the example demonstrates
   - Any relevant issue numbers
   - Screenshots or output examples (if applicable)

Feel free to ask questions in the pull request if you need clarification or guidance.
