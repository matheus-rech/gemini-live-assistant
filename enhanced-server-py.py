import asyncio
import json
import os
import signal
import sys
import traceback
import logging
from typing import Set, Dict, Any, List, Optional
from datetime import datetime

import websockets
from websockets.server import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosedError

# Import our main app
from main import AudioLoop, MODEL, SYSTEM_PROMPT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('gemini-server')

# Global variables to track active connections
active_connections: Set[WebSocketServerProtocol] = set()
running = True

# Session management
client_sessions: Dict[str, Dict[str, Any]] = {}


class SessionManager:
    """Manages client sessions and their state."""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeouts: Dict[str, float] = {}
        self.max_sessions = 100
        self.session_timeout = 3600  # 1 hour in seconds
    
    def create_session(self, client_id: str) -> Dict[str, Any]:
        """Create a new session for a client."""
        # If we're at max capacity, remove oldest session
        if len(self.sessions) >= self.max_sessions:
            oldest_client = min(self.session_timeouts, key=self.session_timeouts.get)
            self.destroy_session(oldest_client)
        
        self.sessions[client_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "command_history": [],
            "user_preferences": {
                "voice": "Puck",  # Default voice
                "region": None,  # For screen capture region
                "mode": "camera"  # Default mode
            }
        }
        self.session_timeouts[client_id] = datetime.now().timestamp()
        return self.sessions[client_id]
    
    def get_session(self, client_id: str) -> Dict[str, Any]:
        """Get an existing session or create a new one."""
        if client_id not in self.sessions:
            return self.create_session(client_id)
        
        # Update last activity
        self.session_timeouts[client_id] = datetime.now().timestamp()
        self.sessions[client_id]["last_activity"] = datetime.now()
        return self.sessions[client_id]
    
    def update_session(self, client_id: str, key: str, value: Any) -> None:
        """Update a session parameter."""
        session = self.get_session(client_id)
        session[key] = value
    
    def destroy_session(self, client_id: str) -> None:
        """Remove a session."""
        if client_id in self.sessions:
            del self.sessions[client_id]
        
        if client_id in self.session_timeouts:
            del self.session_timeouts[client_id]
    
    async def cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        current_time = datetime.now().timestamp()
        expired_clients = []
        
        for client_id, last_access in self.session_timeouts.items():
            if current_time - last_access > self.session_timeout:
                expired_clients.append(client_id)
        
        for client_id in expired_clients:
            logger.info(f"Removing expired session for client {client_id}")
            self.destroy_session(client_id)


# Create a global session manager
session_manager = SessionManager()


# Create a WebSocket-specific version of the AudioLoop
class WebSocketAudioLoop(AudioLoop):
    def __init__(self, websocket: WebSocketServerProtocol, video_mode="none"):
        super().__init__(video_mode=video_mode)
        self.websocket = websocket
        self.message_queue = asyncio.Queue()
        self.client_id = str(id(websocket))
        self.session = session_manager.get_session(self.client_id)
        self.token = None
        self.last_error_time = 0  # To limit error message frequency
    
    # Override methods to work with WebSocket
    async def send_response(self, text: str, message_type: str = "text", extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Send a text response to the WebSocket client."""
        try:
            response_data = {
                "type": message_type,
                "content": text
            }
            
            # Add any extra data if provided
            if extra_data:
                response_data.update(extra_data)
                
            await self.websocket.send(json.dumps(response_data))
        except Exception as e:
            current_time = datetime.now().timestamp()
            # Limit error logging to prevent log spam
            if current_time - self.last_error_time > 5:
                logger.error(f"Error sending response: {e}")
                self.last_error_time = current_time
    
    async def process_text(self, text: str) -> None:
        """Process text input from the WebSocket client."""
        try:
            # Add command to history
            if self.session and "command_history" in self.session:
                self.session["command_history"].append({
                    "type": "text",
                    "content": text,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Send status update to client
            await self.send_response("Processing your message...", "status")
            
            # Process with Gemini
            start_time = datetime.now().timestamp()
            response = await asyncio.to_thread(
                self.connect.send,
                text
            )
            
            # Extract response text
            response_text = ""
            for chunk in response:
                if chunk.text:
                    response_text += chunk.text
            
            # Send response to client
            if response_text:
                await self.send_response(response_text)
                
                # Log processing time
                processing_time = datetime.now().timestamp() - start_time
                logger.info(f"Text processed in {processing_time:.2f}s")
                
                # Update performance metrics
                self.performance_metrics["api_response_time"].append(processing_time)
            else:
                await self.send_response("I didn't generate a response. Please try again.")
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            await self.send_response(f"Error processing your message: {str(e)}", "error")
            self.performance_metrics["api_errors"] += 1
    
    async def process_image(self, image_data: str) -> None:
        """Process image data from the WebSocket client."""
        try:
            # Decode base64 image
            import base64
            import io
            from PIL import Image
            
            # Strip data URL prefix if present
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            # Send status update to client
            await self.send_response("Analyzing image...", "status")
            
            # Convert to Part object
            import io
            from google.genai import types
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Create part with image MIME type
            part = types.Part.blob(img_bytes, 'image/png')
            
            # Send to model
            start_time = datetime.now().timestamp()
            response = await asyncio.to_thread(
                self.connect.send,
                part
            )
            
            # Extract response text
            response_text = ""
            for chunk in response:
                if chunk.text:
                    response_text += chunk.text
            
            # Send response to client
            if response_text:
                await self.send_response(response_text)
                
                # Log processing time
                processing_time = datetime.now().timestamp() - start_time
                logger.info(f"Image processed in {processing_time:.2f}s")
                
                # Update performance metrics
                self.performance_metrics["api_response_time"].append(processing_time)
            else:
                await self.send_response("I couldn't analyze this image. Please try another one.")
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            await self.send_response(f"Error processing your image: {str(e)}", "error")
            self.performance_metrics["api_errors"] += 1
        
    async def process_audio(self, audio_data: str) -> None:
        """Process audio data from the WebSocket client."""
        try:
            # Decode base64 audio
            import base64
            
            # Strip data URL prefix if present
            if audio_data.startswith('data:audio'):
                audio_data = audio_data.split(',')[1]
            
            audio_bytes = base64.b64decode(audio_data)
            
            # Send status update to client
            await self.send_response("Processing audio...", "status")
            
            # Convert to Part object for Gemini
            from google.genai import types
            
            # Create part with audio MIME type
            audio_part = types.Part.audio(audio_bytes, mime_type="audio/raw;encoding=signed-integer;bits_per_sample=16;sample_rate=16000")
            
            # Send to model
            start_time = datetime.now().timestamp()
            response = await asyncio.to_thread(
                self.connect.send,
                audio_part
            )
            
            # Extract response text and audio
            response_text = ""
            for chunk in response:
                if chunk.text:
                    response_text += chunk.text
            
            # Send text response to client
            if response_text:
                await self.send_response(response_text)
                
                # Log processing time
                processing_time = datetime.now().timestamp() - start_time
                logger.info(f"Audio processed in {processing_time:.2f}s")
                
                # Update performance metrics
                self.performance_metrics["api_response_time"].append(processing_time)
                
                # Also send a response with Gemini's audio output
                # In a real implementation, this would be generated from text-to-speech
                # or retrieved from the Gemini LiveConnect response
                mock_audio_base64 = "UklGRiQEAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAEAAD+/wIA..."  # Truncated
                
                # Send audio data back to client
                await self.send_response(mock_audio_base64, "audio")
            else:
                await self.send_response("I couldn't understand the audio. Please try again.")
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await self.send_response(f"Error processing your audio: {str(e)}", "error")
            self.performance_metrics["api_errors"] += 1


async def handle_client(websocket: WebSocketServerProtocol, path: str) -> None:
    """Handle a WebSocket connection from a client."""
    global active_connections
    
    client_id = str(id(websocket))
    logger.info(f"Client connected: {client_id}")
    active_connections.add(websocket)
    
    # Retrieve or create session for this client
    session = session_manager.get_session(client_id)
    
    # Get preferred mode from session or use default
    video_mode = session.get("user_preferences", {}).get("mode", "none")
    
    # Create a client-specific instance of our app
    client_app = WebSocketAudioLoop(websocket, video_mode=video_mode)
    
    # Initialize Gemini session with retry logic
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Initialize LiveConnect session
            await client_app.initialize_gemini_session()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Failed to initialize Gemini session (attempt {attempt + 1}/{max_retries}): {e}")
                await asyncio.sleep(1 * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"Failed to initialize Gemini session after {max_retries} attempts: {e}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "content": f"Failed to initialize Gemini session: {str(e)}"
                }))
                active_connections.remove(websocket)
                return
    
    try:
        # Send initial status to client
        await websocket.send(json.dumps({
            "type": "status",
            "content": "Connected to Gemini Voice Assistant",
            "session": {
                "client_id": client_id,
                "created_at": session["created_at"].isoformat(),
                "preferences": session.get("user_preferences", {})
            }
        }))
        
        # Process incoming messages
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Add timeout protection
                try:
                    await asyncio.wait_for(
                        process_message(websocket, data, client_app),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing message from client {client_id}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": "Request timed out. Please try again."
                    }))
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from client {client_id}")
                await websocket.send(json.dumps({
                    "type": "error",
                    "content": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                await websocket.send(json.dumps({
                    "type": "error",
                    "content": f"Server error: {str(e)}"
                }))
    except ConnectionClosedError:
        logger.info(f"Client {client_id} connection closed")
    except Exception as e:
        logger.error(f"Unexpected error with client {client_id}: {e}")
        traceback.print_exc()
    finally:
        # Clean up resources
        active_connections.remove(websocket)
        logger.info(f"Client {client_id} disconnected")


async def process_message(websocket: WebSocketServerProtocol, data: Dict[str, Any], client_app: WebSocketAudioLoop) -> None:
    """Process a message from a client."""
    client_id = str(id(websocket))
    msg_type = data.get('type')
    content = data.get('content')
    
    # Update session last activity time
    session = session_manager.get_session(client_id)
    
    if msg_type == 'text':
        # Process text message
        logger.info(f"Received text message from client {client_id}")
        await client_app.process_text(content)
    
    elif msg_type == 'audio':
        # Process audio data
        logger.info(f"Received audio data from client {client_id}")
        await client_app.process_audio(content)
    
    elif msg_type == 'image':
        # Process image data
        logger.info(f"Received image data from client {client_id}")
        await client_app.process_image(content)
    
    elif msg_type == 'mode':
        # Change the mode (camera, screen, none)
        new_mode = content
        logger.info(f"Client {client_id} changing mode to: {new_mode}")
        
        client_app.video_mode = new_mode
        
        # Update user preferences in session
        if "user_preferences" in session:
            session["user_preferences"]["mode"] = new_mode
        
        await websocket.send(json.dumps({
            "type": "status",
            "content": f"Mode changed to {new_mode}"
        }))
    
    elif msg_type == 'voice':
        # Change the voice used for responses
        new_voice = content
        logger.info(f"Client {client_id} changing voice to: {new_voice}")
        
        # Update configuration (would typically reconfigure the LiveConnect session)
        # For this example, we just update the preference
        if "user_preferences" in session:
            session["user_preferences"]["voice"] = new_voice
        
        await websocket.send(json.dumps({
            "type": "status",
            "content": f"Voice changed to {new_voice}"
        }))
    
    elif msg_type == 'region':
        # Set screen capture region
        logger.info(f"Client {client_id} setting screen region")
        
        # Parse region data (x, y, width, height)
        region = content
        
        # Update client app
        client_app.selected_region = region
        
        # Update user preferences in session
        if "user_preferences" in session:
            session["user_preferences"]["region"] = region
        
        await websocket.send(json.dumps({
            "type": "status",
            "content": f"Screen region set to {region}" if region else "Screen region reset to full screen"
        }))
    
    elif msg_type == 'ping':
        # Simple heartbeat mechanism
        await websocket.send(json.dumps({
            "type": "pong",
            "content": datetime.now().isoformat()
        }))
    
    else:
        logger.warning(f"Unknown message type: {msg_type} from client {client_id}")
        await websocket.send(json.dumps({
            "type": "error",
            "content": f"Unknown message type: {msg_type}"
        }))


async def broadcast_message(message: Dict[str, Any]) -> None:
    """Send a message to all connected clients."""
    if active_connections:
        await asyncio.gather(
            *[connection.send(json.dumps(message)) for connection in active_connections]
        )


async def session_cleanup_task() -> None:
    """Periodically clean up expired sessions."""
    while running:
        await asyncio.sleep(300)  # Run every 5 minutes
        try:
            await session_manager.cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"Error in session cleanup: {e}")


async def status_report_task() -> None:
    """Periodically log server status."""
    while running:
        await asyncio.sleep(600)  # Run every 10 minutes
        try:
            active_client_count = len(active_connections)
            session_count = len(session_manager.sessions)
            logger.info(f"Server status: {active_client_count} active clients, {session_count} sessions")
        except Exception as e:
            logger.error(f"Error in status report: {e}")


async def start_server() -> None:
    """Start the WebSocket server."""
    global running
    
    # Configuration
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 8000))
    
    # Create and start the server
    logger.info(f"Starting WebSocket server on {host}:{port}")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(session_cleanup_task())
    status_task = asyncio.create_task(status_report_task())
    
    # Start the WebSocket server
    async with websockets.serve(handle_client, host, port):
        while running:
            await asyncio.sleep(1)
    
    # Clean up
    cleanup_task.cancel()
    status_task.cancel()
    logger.info("Server stopped")


def signal_handler(sig, frame) -> None:
    """Handle termination signals to clean up resources."""
    global running
    logger.info("Shutdown signal received")
    running = False


if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the server
        asyncio.run(start_server())
    except KeyboardInterrupt:
        logger.info("Server interrupted")
    except Exception as e:
        logger.error(f"Server error: {e}")
        traceback.print_exc()
    finally:
        logger.info("Server shutdown complete")
