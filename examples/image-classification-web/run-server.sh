# Opening index.html file directly by a browser does not work because of
# the security restrictions by the browser.

if ! command -v python3 &>/dev/null; then
    echo "python3 could not be found. Running server requires python3."
    exit
fi

python3 https_server.py
