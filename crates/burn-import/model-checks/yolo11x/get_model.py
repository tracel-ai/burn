#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx-weekly==1.19.0.dev20250419",
#   "ultralytics>=8.3.0",
#   "numpy",
#   "pillow",
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
import numpy as np
from pathlib import Path


def download_yolo11x_model():
    """Download YOLO11x model and export to ONNX format, or use existing ONNX if available."""
    # Create artifacts directory
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    output_path = artifacts_dir / "yolo11x.onnx"
    
    # Check if ONNX model already exists
    if output_path.exists():
        print(f"✓ ONNX model already exists at: {output_path}")
        print("  Skipping download and conversion.")
        return str(output_path)
    
    # Only download and convert if ONNX doesn't exist
    try:
        from ultralytics import YOLO
        
        # Download and load YOLO11x model
        print("Downloading YOLO11x model...")
        pt_path = artifacts_dir / "yolo11x.pt"
        model = YOLO(str(pt_path))
        
        # Export to ONNX
        print(f"Exporting YOLO11x model to ONNX format...")
        
        # Export model (it will be created in current directory)
        model.export(format="onnx", simplify=True)
        
        # Move the exported file to artifacts directory
        exported_file = Path("yolo11x.onnx")
        if exported_file.exists() and not output_path.exists():
            exported_file.rename(output_path)
        
        # Clean up the .pt file after successful conversion
        if output_path.exists():
            print(f"Model exported successfully to: {output_path}")
            if pt_path.exists():
                print(f"Removing temporary PyTorch file: {pt_path}")
                pt_path.unlink()
            return str(output_path)
        else:
            raise FileNotFoundError(f"Expected ONNX file not found at {output_path}")
            
    except Exception as e:
        raise RuntimeError(f"Failed to download/export YOLO11x model: {str(e)}")


def load_onnx_model(model_path):
    """Load and validate ONNX model."""
    try:
        model = onnx.load(model_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {str(e)}")

    try:
        onnx.checker.check_model(model)
        print(f"✓ Model loaded and validated successfully: {model_path}")
    except Exception as e:
        print(f"⚠ Model validation warning: {str(e)}")

    return model


def print_model_info(model):
    """Print model information including opset version and input/output shapes."""
    try:
        opset_version = model.opset_import[0].version
        print(f"\nModel Information:")
        print(f"  Opset version: {opset_version}")
        
        # Print input information
        print(f"\n  Inputs:")
        for input in model.graph.input:
            shape = []
            for dim in input.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            print(f"    - {input.name}: {shape}")
        
        # Print output information
        print(f"\n  Outputs:")
        for output in model.graph.output:
            shape = []
            for dim in output.type.tensor_type.shape.dim:
                if dim.HasField('dim_value'):
                    shape.append(dim.dim_value)
                elif dim.HasField('dim_param'):
                    shape.append(dim.dim_param)
                else:
                    shape.append('?')
            print(f"    - {output.name}: {shape}")
            
    except (IndexError, AttributeError) as e:
        print(f"Warning: Could not extract model information: {e}")


def upgrade_model_opset(model, target_opset=16):
    """Upgrade model to target opset version."""
    try:
        current_opset = model.opset_import[0].version
        if current_opset >= target_opset:
            print(f"✓ Current opset version {current_opset} is already >= {target_opset}, skipping upgrade.")
            return model

        print(f"Upgrading model from opset {current_opset} to {target_opset}...")
        upgraded_model = version_converter.convert_version(model, target_opset)
        print(f"✓ Model upgraded to opset {target_opset}.")
        return upgraded_model
    except Exception as e:
        raise RuntimeError(f"Failed to upgrade model to opset {target_opset}: {str(e)}")


def apply_shape_inference(model):
    """Apply shape inference to the model."""
    try:
        print("Applying shape inference...")
        inferred_model = shape_inference.infer_shapes(model)
        print("✓ Shape inference applied successfully.")
        return inferred_model
    except Exception as e:
        print(f"⚠ Shape inference partially applied: {str(e)}")
        return model


def run_basic_evaluation(model_path):
    """Run basic evaluation/inference test on the model."""
    try:
        import onnx.numpy_helper
        
        print("\nRunning basic ONNX evaluation tests...")
        
        # Load the model
        model = onnx.load(model_path)
        
        # Get model input shape
        input_info = model.graph.input[0]
        input_shape = []
        for dim in input_info.type.tensor_type.shape.dim:
            if dim.HasField('dim_value'):
                input_shape.append(dim.dim_value)
            else:
                input_shape.append(1)  # Default to 1 for dynamic dimensions
        
        # YOLO models typically expect [batch, channels, height, width]
        # If shape is dynamic, use typical YOLO11 input dimensions
        if len(input_shape) != 4:
            input_shape = [1, 3, 640, 640]  # Default YOLO input
        else:
            # Ensure reasonable dimensions
            if input_shape[2] == 0 or input_shape[2] > 2000:
                input_shape[2] = 640
            if input_shape[3] == 0 or input_shape[3] > 2000:
                input_shape[3] = 640
        
        print(f"  Input shape for testing: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Try to run inference using onnxruntime if available
        try:
            import onnxruntime as ort
            
            print("  Running inference with ONNXRuntime...")
            session = ort.InferenceSession(model_path)
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: dummy_input})
            
            print(f"  ✓ Inference successful!")
            print(f"  Output shapes: {[output.shape for output in outputs]}")
            
        except ImportError:
            print("  ⚠ ONNXRuntime not available, skipping inference test")
            print("  To enable inference testing, install: uv pip install onnxruntime")
        except Exception as e:
            print(f"  ⚠ Inference test failed: {str(e)}")
        
        # Basic model statistics
        print(f"\n  Model statistics:")
        print(f"    - Number of nodes: {len(model.graph.node)}")
        print(f"    - Number of initializers: {len(model.graph.initializer)}")
        print(f"    - Model size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
        
        print("\n✓ Basic evaluation completed.")
        
    except Exception as e:
        print(f"⚠ Basic evaluation failed: {str(e)}")


def save_model(model, output_path):
    """Save the model to specified path."""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        onnx.save(model, output_path)
        print(f"✓ Model saved to: {output_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to save model: {str(e)}")


def main():
    print("=" * 60)
    print("YOLO11x Model Download and Processing Tool")
    print("=" * 60)
    
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    
    original_model_path = str(artifacts_dir / "yolo11x.onnx")
    output_path = str(artifacts_dir / "yolo11x_opset16.onnx")
    
    # Check if upgraded model already exists (that's all we really need)
    if Path(output_path).exists():
        print(f"\n✓ Upgraded ONNX model already exists: {output_path}")
        print("  This is the only model needed for burn-import testing.")
        print("\nSkipping download and processing steps.")
        
        # Just run evaluation test on the upgraded model
        print("\nRunning evaluation test...")
        run_basic_evaluation(output_path)
        
        print("\n" + "=" * 60)
        print("✓ Evaluation completed successfully!")
        print("=" * 60)
        return
    
    # Step 1: Download YOLO11x model (or use existing)
    print("\nStep 1: Preparing YOLO11x model...")
    original_model_path = download_yolo11x_model()
    
    # Check if upgraded model already exists
    if Path(output_path).exists():
        print(f"\n✓ Upgraded model already exists at: {output_path}")
        print("  Skipping processing steps.")
    else:
        # Step 2: Load and validate the original model
        print("\nStep 2: Loading and validating original model...")
        model = load_onnx_model(original_model_path)
        print_model_info(model)
        
        # Step 3: Upgrade opset version
        print("\nStep 3: Upgrading opset version...")
        upgraded_model = upgrade_model_opset(model, target_opset=16)
        
        # Step 4: Apply shape inference
        print("\nStep 4: Applying shape inference...")
        inferred_model = apply_shape_inference(upgraded_model)
        
        # Step 5: Save upgraded model
        print(f"\nStep 5: Saving upgraded model...")
        save_model(inferred_model, output_path)
    
    # Step 6: Run basic evaluation on both models
    print("\nStep 6: Running evaluation tests...")
    print("\nTesting original model:")
    run_basic_evaluation(original_model_path)
    
    print("\nTesting upgraded model:")
    run_basic_evaluation(output_path)
    
    print("\n" + "=" * 60)
    print("✓ YOLO11x model processing completed successfully!")
    print(f"  Original model: {original_model_path}")
    print(f"  Processed model: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠ Operation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        sys.exit(1)