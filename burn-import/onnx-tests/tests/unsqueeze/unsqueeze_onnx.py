from pathlib import Path
from typing import List, Optional, Tuple
from numpy.typing import ArrayLike, NDArray
import onnx
import onnxruntime
import numpy as np
from dataclasses import dataclass


def validate_sequence(arr: List | Tuple) -> None:
    """Function to validate that all elements in a sequence are of the same type.


    Args:
        arr (List | Tuple): Sequence to validate

    Raises:
        ValueError: raised if the elements in the sequence are not of the same type.
    """
    el_type = type(arr[0])
    for el in arr:
        if type(el) != el_type:
            raise ValueError(
                f"Expected all elements to be of type {el_type} but got {type(el)}"
            )


# extracting because match statement doesn't work nested
def array_handler(name: str, arr: np.ndarray) -> onnx.TensorProto:
    """Function for mapping different numpy dtypes to onnx tensor types.

    Args:
        name (str): name of the generated tensor
        arr (np.ndarray): data for the tensor, Note that data isn't stored in the tensor,
        constant nodes are WIP.

    Raises:
        ValueError: If the dtype is not a supported type (float32, float64, int32, int64) then an error is raised.

    Returns:
        onnx.TensorProto: a tensor proto with the data type and shape of the input array.
    """
    match arr.dtype:
        case np.float32:
            return onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.FLOAT, arr.shape
            )
        case np.float64:
            return onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.DOUBLE, arr.shape
            )

        case np.int32:
            return onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.INT32, arr.shape
            )
        case np.int64:
            return onnx.helper.make_tensor_value_info(
                name, onnx.TensorProto.INT64, arr.shape
            )

        case _:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")


def make_onnx_types(
    name: str, input_data: ArrayLike
) -> onnx.TensorProto | onnx.ValueInfoProto:
    """Function to map inputs to OnnxOpData to onnx types

    Args:
        name (str): The name of the input
        v (ArrayLike): The input data

    Raises:
        ValueError: If the input data is not a supported type (np.ndarray, list, tuple, int, float, bool) then an error is raised.

    Returns:
        onnx.TensorProto | onnx.ValueInfoProto: returns a tensor proto or value info proto based on the input data.
    """
    print(type(input_data))
    match input_data:
        case np.ndarray():
            return array_handler(name, input_data)  # type: ignore
        case list() | tuple():
            validate_sequence(input_data)  # type: ignore
            return array_handler(name, np.array(input_data))
        case int():
            return onnx.helper.make_value_info(name, onnx.ValueInfoProto.INT64)
        case float():
            return onnx.helper.make_value_info(name, onnx.ValueInfoProto.FLOAT)
        case bool():
            return onnx.helper.make_value_info(name, onnx.ValueInfoProto.BOOL)
        case _:
            raise ValueError(f"Unsupported type: {type(input_data)}")


@dataclass
class OnnxOpData:
    """helper for generating and validating models for onnx operators.

    Attributes:
        name (str): The name of the operator. Must match the name of the operator in onnx.
        inputs (dict[str, ArrayLike]): The inputs to the operator
        output (dict[str, ArrayLike]): The expected output of the operator
    """

    name: str
    inputs: dict[str, ArrayLike]
    output: dict[str, ArrayLike]

    @property
    def input_names(self) -> List[str]:
        return list(self.inputs.keys())

    @property
    def output_names(self) -> List[str]:
        return list(self.output.keys())

    @property
    def graph_inputs(self) -> List[onnx.TensorProto | onnx.ValueInfoProto]:
        return [make_onnx_types(k, v) for k, v in self.inputs.items()]

    @property
    def graph_outputs(self) -> List[onnx.TensorProto | onnx.ValueInfoProto]:
        return [make_onnx_types(k, v) for k, v in self.output.items()]

    def make_onnx_graph(self, op_name: Optional[str] = None) -> onnx.GraphProto:
        """Create a graph with a single node for testing.

        Args:
            op_inputs: The input tensor to the node."""
        node = onnx.helper.make_node(
            op_name or self.name,  # you learn something new every day
            inputs=self.input_names,
            outputs=self.output_names,
        )

        graph: onnx.GraphProto = onnx.helper.make_graph(
            [node],
            f"{self.name}_test",
            self.graph_inputs,
            outputs=self.graph_outputs,
        )

        return graph

    def save_model(self, path: Optional[Path | str] = None):
        """Converts the generated graph to an onnx model and saves it to a file.

        Args:
            path (Optional[Path  |  str], optional): desired path to the output. if unspecified, defaults to {op_name}.onnx

        Raises:
            ValueError: If you provide a path and it doesn't end with .onnx then an error is raised
        """
        model = onnx.helper.make_model(self.make_onnx_graph())
        if path and Path(path).suffix != ".onnx":
            raise ValueError(
                f"Provide path {path} must include the model name and end with .onnx extension"
            )
        onnx.save(model, str(path) if path else f"{self.name.lower()}.onnx")
        print(f"Model saved to {path}")

    def validate_model(self):
        """Loads the generated model and runs it with the provided inputs to validate the output.
        More of a sanity check than anything else.

        Returns:
            Outputs (Any): returns the outputs of the model in case there is a need to inspect them.
        """
        sess = onnxruntime.InferenceSession(f"{self.name.lower()}.onnx")
        sess_inputs = [inp.name for inp in sess.get_inputs()]
        outputs = sess.run(self.output_names, self.inputs)
        for i, k in enumerate(self.output.keys()):
            assert np.allclose(self.output[k], outputs[i])
        print("Output is the same as expected. Test passed.")
        return outputs

    def model_to_txt(self, path: Optional[Path | str] = None):
        """load the generated model and save it to a txt file for debugging purposes.

        Args:
            path (Optional[Path  |  str], optional): desired path to the output. if unspecified, defaults to {op_name}.txt
            in the current directory. Defaults to None.

        Raises:
            ValueError: If you provide a path and it doesn't end with .txt then an error is raised
        """
        model = onnx.helper.make_model(self.make_onnx_graph())
        if path and Path(path).suffix != ".txt":
            raise ValueError(
                f"Provide path {path} must include the model name and end with .txt extension"
            )
        with open(path if path else f"{self.name}.txt", "w") as f:
            f.write(str(model))
        print(f"Model saved to {path}")


if __name__ == "__main__":
    axes = [0, 4]
    x = np.array(np.random.randn(3, 4, 5))
    y = np.expand_dims(x, axis=axes)
    if y.shape != (1, 3, 4, 5, 1):
        raise ValueError(f"Expected shape (1,3,4,5,1) but got {y.shape}")

    data = OnnxOpData(
        name="Unsqueeze",
        inputs={"x": x, "axes": axes},
        output={"output": y},
    )

    # graph = data.make_onnx_graph("Unsqueeze")
    data.model_to_txt()
    # data.save_model()
    # result = data.validate_model()
    # print(result[0].shape)
    # print(data.output["output"])
    # assert np.allclose(result[0], data.output["output"])
    # print("Test passed")
