#!/usr/bin/env bash

# Opening index.html file directly by a browser does not work because of
# the security restrictions by the browser. Viewing the HTML file will fail with 
# this error message:

# ```
# Access to script at
#  'file:///Users/user/Projects/burn-mac/examples/mnist-inference-web/pkg/mnist_inference_web.js' 
# from origin 'null' has been blocked by CORS policy: 
# Cross origin requests are only supported for protocol schemes: 
# http, data, isolated-app, chrome-extension, chrome, https, chrome-untrusted.
# ```
#  So that's why running a local HTTP server is needed. 

if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Running server requires python3."
    exit
fi

echo "Running local python HTTP server on port 8000 ..."
python3 -m http.server 8000
