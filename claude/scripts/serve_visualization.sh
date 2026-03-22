#!/bin/bash
# Serve the latent visualization locally
# Usage: ./serve_visualization.sh [port]

PORT=${1:-8080}
cd /workspace-vast/sshlin/codi/outputs

echo "Serving visualization at http://localhost:$PORT/latent_visualization.html"
echo "Press Ctrl+C to stop"

python3 -m http.server $PORT
