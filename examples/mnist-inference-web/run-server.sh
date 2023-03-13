if ! command -v python3 &> /dev/null
then
    echo "python3 could not be found. Running server requires python3."
    exit
fi

echo "Running python HTTP server ..."
python3 -m http.server 8000
