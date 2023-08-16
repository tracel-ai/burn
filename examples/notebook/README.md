# Jupyter Notebook Examples with Burn

This directory includes Jupyter Notebook examples showcasing the usage of the Burn deep learning
framework in Rust through
[Evcxr Jupyter](https://github.com/evcxr/evcxr/blob/main/evcxr_jupyter/README.md). The examples are
systematically organized based on the specific Burn features they illustrate.

## Viewing Options

You can explore the examples in different ways:

- **Notebook Viewer:** If you prefer not to set up the entire crate package, you can view the
  examples in a notebook viewer or run them to see images and other media outputs.

- **Visual Studio Code (vscode):** If you're using vscode, you already have access to a built-in
  notebook viewer, enabling you to open and interact with the notebook files directly.

For other editors, you can utilize the [Jupyter Notebook Viewer](https://nbviewer.jupyter.org/).

## Getting Started with Rust and Evcxr

To execute the Rust code within the notebooks, you must install the Evcxr kernel. Here's how to get
started:

### Install Evcxr Kernel

1. **Build Evcxr Kernel:** Install the required package with the following command:

   ```shell
   cargo install evcxr_jupyter
   ```

2. **Install and Register the Kernel to Jupyter:**
   ```shell
   evcxr_jupyter --install
   ```

### Open and Run Notebooks

Once the kernel is installed, you can open the notebook files in your preferred editor and run the
code. Ensure that the kernel is set to `Rust` within the notebook for proper execution.

## Additional Reading Resources

- [Notebook Special Commands for Evcxr](https://github.com/evcxr/evcxr/blob/main/COMMON.md): Learn
  about the unique commands and functionalities offered by Evcxr for a more efficient workflow with
  Jupyter Notebooks.
