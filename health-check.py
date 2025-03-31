import asyncio
import json
import http
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

class HealthHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler for health check endpoint."""
    
    def do_GET(self):
        if self.path == '/health':
            self.send_response(http.HTTPStatus.OK)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            health_data = {
                'status': 'ok',
                'version': '1.0.0',
                'connections': len(active_connections)
            }
            self.wfile.write(json.dumps(health_data).encode('utf-8'))
        else:
            self.send_response(http.HTTPStatus.NOT_FOUND)
            self.end_headers()

def start_health_server(host='0.0.0.0', port=8000):
    """Start a simple HTTP server for health checks."""
    server = HTTPServer((host, port), HealthHandler)
    thread = Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    logging.info(f"Health check server started on http://{host}:{port}/health")
    return server

# Add these lines to the start_server() function in server.py
# Just before the WebSocket server is started:
"""
# Start health check server for Docker and monitoring systems
health_server = start_health_server(host=host, port=int(port))

# Start the WebSocket server
async with websockets.serve(handle_client, host, int(port)):
    # ... rest of the code ...

# Shutdown health server on exit
health_server.shutdown()
"""