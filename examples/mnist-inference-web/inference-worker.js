// This demo is part of Burn: https://github.com/tracel-ai/burn
// Released under the MIT OR Apache-2.0 license.

import init, { Mnist } from "./pkg/mnist_inference_web.js";

async function start() {
    await init();
    const mnist = new Mnist();

    // The page sends one request at a time so the model is never borrowed concurrently.
    self.onmessage = async ({ data: { input, generation } }) => {
        try {
            const output = await mnist.inference(input);
            self.postMessage({ type: "result", generation, output });
        } catch (error) {
            self.postMessage({ type: "error", message: String(error) });
        }
    };
    self.postMessage({ type: "ready" });
}

start().catch((error) => {
    self.postMessage({ type: "error", message: String(error) });
});
