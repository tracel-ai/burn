#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#   "onnx>=1.15.0",
#   "numpy>=1.24.0",
# ]
# ///

"""Diagnostic: Load ONNX files and inspect their structure."""

import onnx

def inspect_model(path):
    print(f"\n{'='*60}")
    print(f"Inspecting: {path}")
    print('='*60)

    model = onnx.load(path)
    graph = model.graph

    print(f"\nInputs ({len(graph.input)}):")
    for inp in graph.input:
        print(f"  - {inp.name}: {inp.type}")

    print(f"\nOutputs ({len(graph.output)}):")
    for out in graph.output:
        print(f"  - {out.name}: {out.type}")

    print(f"\nInitializers ({len(graph.initializer)}):")
    for init in graph.initializer:
        print(f"  - {init.name}: dims={list(init.dims)}")

    print(f"\nNodes ({len(graph.node)}):")
    for node in graph.node:
        print(f"  - {node.op_type} '{node.name}': {list(node.input)} → {list(node.output)}")

if __name__ == '__main__':
    inspect_model('../fixtures/constant_lifting.onnx')
    inspect_model('../fixtures/constant_multiple_refs.onnx')
    inspect_model('../fixtures/matmul_dynamic_weights.onnx')
