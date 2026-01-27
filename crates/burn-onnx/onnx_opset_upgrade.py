#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
# ]
# ///
#
# Learn more about Astral's UV tool at
# https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies

import os
import sys
import onnx
from onnx import shape_inference
from onnx import version_converter


def validate_model_path(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")


def load_onnx_model(model_path):
    try:
        model = onnx.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

    try:
        onnx.checker.check_model(model)
        print("Model loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {str(e)}")

    return model


def print_opset_version(model):
    try:
        print("Opset version:", model.opset_import[0].version)
    except (IndexError, AttributeError):
        print("Warning: Could not determine opset version")


def upgrade_model(model):
    try:
        current_opset = model.opset_import[0].version
        if current_opset >= 16:
            print(f"Current opset version {current_opset} is already >= 16, skipping upgrade.")
            return model

        upgraded_model = version_converter.convert_version(model, 16)
        print("Model upgraded to opset 16.")
        return upgraded_model
    except Exception as e:
        raise RuntimeError(f"Failed to upgrade model to opset 16: {str(e)}")


def apply_shape_inference(upgraded_model):
    try:
        inferred_model = shape_inference.infer_shapes(upgraded_model)
        print("Model shape inference applied.")
        return inferred_model
    except Exception as e:
        print(f"Warning: Shape inference partially applied: {str(e)}")
        return upgraded_model


def save_model(inferred_model, output_path):
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        onnx.save(inferred_model, output_path)
        print(f"Model saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {str(e)}")

def main():
    # Get input path from user prompt
    model_path = input("Enter the path to the input ONNX model: ")
    validate_model_path(model_path)

    # Process the model
    model = load_onnx_model(model_path)
    print_opset_version(model)
    upgraded_model = upgrade_model(model)
    inferred_model = apply_shape_inference(upgraded_model)

    # Get output path from user prompt
    default_output = model_path.replace('.onnx', '_opset16.onnx')
    output_path_input = input(f"Enter the path to save the output ONNX model (press Enter for default '{default_output}'): ")
    output_path = output_path_input if output_path_input else default_output

    save_model(inferred_model, output_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
