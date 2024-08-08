# Originally copied and modified from:
# https://github.com/tensorflow/tensorflow/blob/e0b19f6ef223af40e2e6d1d21b8464c1b2ebee8f/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb
# under the following license: Apache License 2.0
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tf2onnx
import onnx
import math
from pathlib import Path


def main():
    # Define paths to model files
    MODELS_DIR = '../src/model/'
    os.makedirs(MODELS_DIR, exist_ok=True)
    MODEL_ONNX = MODELS_DIR + 'sine.onnx'

    np.random.seed(1)

    # Number of sample datapoints
    SAMPLES = 1000

    # Generate a uniformly distributed set of random numbers in the range from
    # 0 to 2π, which covers a complete sine wave oscillation
    x_values = np.random.uniform(
        low=0, high=2*math.pi, size=SAMPLES).astype(np.float32)

    # Shuffle the values to guarantee they're not in order
    np.random.shuffle(x_values)

    # Calculate the corresponding sine values
    y_values = np.sin(x_values).astype(np.float32)

    # Add a small random number to each y value to mimic real world data
    y_values += 0.1 * np.random.randn(*y_values.shape)

    # We'll use 60% of our data for training and 20% for testing. The remaining
    # 20% will be used for validation. Calculate the indices of each section.
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

    # Use np.split to chop our data into three parts.
    # The second argument to np.split is an array of indices where the data
    # will be split. We provide two indices, so the data will be divided into
    # three chunks.
    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

    # Double check that our splits add up correctly
    assert (x_train.size + x_validate.size + x_test.size) == SAMPLES

    model = tf.keras.Sequential()

    # First layer takes a scalar input and feeds it through 16 "neurons". The
    # neurons decide whether to activate based on the 'relu' activation
    # function.
    model.add(keras.layers.Dense(16, activation='relu', input_shape=(1,)))

    # The new second layer may help the network learn more complex
    # representations
    model.add(keras.layers.Dense(16, activation='relu'))

    # Final layer is a single neuron, since we want to output a single value
    model.add(keras.layers.Dense(1))

    # Compile the model using a standard optimizer and loss function for
    # regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(x_train, y_train, epochs=500, batch_size=64,
                        validation_data=(x_validate, y_validate))

    # Use from_function for tf functions
    onnx_model, _ = tf2onnx.convert.from_keras(model, opset=16)
    onnx.save(onnx_model, MODEL_ONNX)
    print("Onnx model generated at", Path(MODEL_ONNX).absolute())


if __name__ == '__main__':
    main()
