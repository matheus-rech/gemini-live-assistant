import asyncio
import json
import os
import signal
import sys
import traceback
import logging
import datetime
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
                "mode": "camera",  # Default mode
                "adhd_support": True,  # Enable ADHD support by default
                "is_medical_context": False,  # Medical content flag
                "reading_pace": "moderate"  # Reading pace for ADHD support
            },
            "flashcards": []  # Store flashcards for study sessions
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
    
    def check_session_duration(self, client_id: str) -> bool:
        """
        Check if a session has exceeded the maximum duration limit.
        
        Args:
            client_id: The client ID to check
            
        Returns:
            True if session is still valid, False if it has expired
        """
        if client_id not in self.sessions:
            return False
        
        session = self.sessions[client_id]
        
        # Get session start time
        created_at = session["created_at"]
        
        # Check if the session has video mode
        has_video = session["user_preferences"].get("mode") in ("camera", "screen")
        
        # Max duration: 15 minutes for audio, 2 minutes for audio+video
        max_duration = datetime.timedelta(minutes=2 if has_video else 15)
        
        # Check if session has exceeded maximum duration
        current_time = datetime.now()
        if current_time - created_at > max_duration:
            logger.warning(f"Session {client_id} has exceeded maximum duration limit of {max_duration}")
            return False
        
        return True
    
    def add_flashcard(self, client_id: str, flashcard: Dict[str, Any]) -> None:
        """Add a flashcard to the user's collection."""
        session = self.get_session(client_id)
        if "flashcards" not in session:
            session["flashcards"] = []
        
        # Add an ID to the flashcard
        if "id" not in flashcard:
            flashcard["id"] = str(len(session["flashcards"]) + 1)
            
        # Add timestamp
        flashcard["created_at"] = datetime.now().isoformat()
        
        session["flashcards"].append(flashcard)
        logger.info(f"Added flashcard to session {client_id}: {flashcard.get('front', '')[:30]}...")
    
    def get_flashcards(self, client_id: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get flashcards for a client, optionally filtered by category."""
        session = self.get_session(client_id)
        flashcards = session.get("flashcards", [])
        
        if category:
            return [f for f in flashcards if f.get("category") == category]
        
        return flashcards


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
        
        # Initialize ADHD support settings from session
        self.adhd_support_enabled = self.session["user_preferences"].get("adhd_support", True)
        self.is_medical_context = self.session["user_preferences"].get("is_medical_context", False)
        self.reading_pace = self.session["user_preferences"].get("reading_pace", "moderate")
    
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
            
            # Check for specific commands related to ADHD and medical support
            lower_text = text.lower()
            
            # Handle educational mode commands
            if 'medical mode' in lower_text or 'usmle mode' in lower_text:
                self.is_medical_context = True
                self.session["user_preferences"]["is_medical_context"] = True
                await self.send_response("Medical education mode activated. Text processing optimized for medical terminology.", "status")
                return
                
            elif 'standard mode' in lower_text:
                self.is_medical_context = False
                self.session["user_preferences"]["is_medical_context"] = False
                await self.send_response("Standard mode activated.", "status")
                return
                
            elif 'flashcard mode' in lower_text:
                self.flashcard_mode = True
                await self.send_response("Flashcard mode activated. I'll help you study with flashcards.", "status")
                return
                
            elif 'adjust reading pace' in lower_text:
                if 'slow' in lower_text:
                    self.reading_pace = "slow"
                    self.session["user_preferences"]["reading_pace"] = "slow"
                    await self.send_response("Reading pace adjusted to slow.", "status")
                elif 'fast' in lower_text:
                    self.reading_pace = "fast"
                    self.session["user_preferences"]["reading_pace"] = "fast"
                    await self.send_response("Reading pace adjusted to fast.", "status")
                else:
                    self.reading_pace = "moderate"
                    self.session["user_preferences"]["reading_pace"] = "moderate"
                    await self.send_response("Reading pace adjusted to moderate.", "status")
                return
                
            # Send status update to client
            await self.send_response("Processing your message...", "status")
            
            # Process with Gemini
            start_time = datetime.now().timestamp()
            
            # Initialize Gemini session if needed
            if not self.connect:
                await self.initialize_gemini_session()
            
            # Send input to Gemini
            await self.connect.send(input=text, end_of_turn=True)
            
            # Process response
            async for response in self.connect.receive():
                if response.text:
                    await self.send_response(response.text)
                elif hasattr(response, 'tool_call'):
                    await self.process_tool_call(response.tool_call)
                elif hasattr(response, 'server_content') and hasattr(response.server_content, 'interrupted') and response.server_content.interrupted:
                    await self.send_response("Generation interrupted by user", "status")
            
            # Log processing time
            processing_time = datetime.now().timestamp() - start_time
            logger.info(f"Text processed in {processing_time:.2f}s")
            
            # Update performance metrics
            self.performance_metrics["api_response_time"].append(processing_time)
            
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
            
            # For medical content, add special preprocessing
            if self.is_medical_context:
                # Apply medical text enhancement if needed
                if self.last_command_context.get('command_type') == 'read_text':
                    image = self.screen_capture.enhance_medical_text(image)
                    await self.send_response("Applied specialized medical text enhancement.", "status")
            
            # Apply focus enhancement if needed for ADHD support
            if self.adhd_support_enabled and self.focus_region:
                image = self.screen_capture.enhance_focus_region(image, self.focus_region)
            
            # Convert to Part object
            import io
            from google.genai import types
            
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            # Create part with image MIME type
            part = types.Part.blob(img_bytes, 'image/png')
            
            # Initialize Gemini session if needed
            if not self.connect:
                await self.initialize_gemini_session()
                
            # Send to model
            start_time = datetime.now().timestamp()
            
            # Add context about medical content if needed
            if self.is_medical_context:
                context = "This is medical educational content for USMLE study. "
                if self.last_command_context.get('command_type') == 'read_text':
                    context += "Please read the medical text precisely, maintaining accuracy of all terminology."
                await self.connect.send(input=context)
            
            await self.connect.send(input=part, end_of_turn=True)
            
            # Process response
            async for response in self.connect.receive():
                if response.text:
                    await self.send_response(response.text)
                elif hasattr(response, 'tool_call'):
                    await self.process_tool_call(response.tool_call)
            
            # Log processing time
            processing_time = datetime.now().timestamp() - start_time
            logger.info(f"Image processed in {processing_time:.2f}s")
            
            # Update performance metrics
            self.performance_metrics["api_response_time"].append(processing_time)
            
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
            
            # Initialize Gemini session if needed
            if not self.connect:
                await self.initialize_gemini_session()
                
            # Send to model
            start_time = datetime.now().timestamp()
            await self.connect.send(input=audio_part, end_of_turn=True)
            
            # Process response
            text_response = ""
            async for response in self.connect.receive():
                if response.text:
                    text_response += response.text
                    await self.send_response(response.text)
                elif response.data:
                    # Send audio data back to client
                    await self.send_response(base64.b64encode(response.data).decode(), "audio")
                elif hasattr(response, 'tool_call'):
                    await self.process_tool_call(response.tool_call)
            
            # Log processing time
            processing_time = datetime.now().timestamp() - start_time
            logger.info(f"Audio processed in {processing_time:.2f}s")
            
            # Update performance metrics
            self.performance_metrics["api_response_time"].append(processing_time)
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await self.send_response(f"Error processing your audio: {str(e)}", "error")
            self.performance_metrics["api_errors"] += 1
    
    async def process_tool_call(self, tool_call):
        """Process tool call from Gemini and relay to client."""
        try:
            logger.info(f"Received tool call from Gemini: {tool_call}")
            
            # Extract function calls information
            function_calls = []
            if hasattr(tool_call, 'function_calls'):
                function_calls = tool_call.function_calls
            elif isinstance(tool_call, dict) and 'function_calls' in tool_call:
                function_calls = tool_call['function_calls']
            
            if not function_calls:
                logger.warning("Received empty tool call")
                return
            
            # Check for medical education specific tool calls
            for call in function_calls:
                # Get function details
                function_name = call.name if hasattr(call, 'name') else call.get('name', '')
                args = call.args if hasattr(call, 'args') else call.get('args', '{}')
                function_id = call.id if hasattr(call, 'id') else call.get('id', '')
                
                # Handle flashcard creation
                if function_name == "create_flashcard":
                    try:
                        args_dict = json.loads(args)
                        flashcard = {
                            "front": args_dict.get("front", ""),
                            "back": args_dict.get("back", ""),
                            "category": args_dict.get("category", "General")
                        }
                        
                        # Store flashcard in session
                        session_manager.add_flashcard(self.client_id, flashcard)
                        
                        # Notify user about the flashcard
                        await self.send_response(
                            f"Created flashcard in category: {flashcard['category']}", 
                            "flashcard_created",
                            {"flashcard": flashcard}
                        )
                    except Exception as e:
                        logger.error(f"Error processing flashcard creation: {e}")
            
            # Send tool call to client
            await self.websocket.send(json.dumps({
                "type": "tool_call",
                "content": {
                    "function_calls": [
                        {
                            "id": call.id if hasattr(call, 'id') else call.get('id'),
                            "name": call.name if hasattr(call, 'name') else call.get('name'),
                            "args": call.args if hasattr(call, 'args') else call.get('args')
                        } for call in function_calls
                    ]
                }
            }))
            
        except Exception as e:
            logger.error(f"Error processing tool call: {e}")
            traceback.print_exc()
            await self.send_response(f"Error processing tool call: {str(e)}", "error")
    
    async def process_tool_response(self, function_responses):
        """Process tool responses from client and send to Gemini."""
        try:
            logger.info(f"Processing tool responses: {function_responses}")
            
            if not self.connect:
                await self.initialize_gemini_session()
            
            # Send tool responses to Gemini
            await self.connect.send_tool_response(function_responses=function_responses)
            
            # Continue processing responses
            async for response in self.connect.receive():
                if response.text:
                    await self.send_response(response.text)
                elif response.data:
                    # Send audio data back to client
                    import base64
                    await self.send_response(base64.b64encode(response.data).decode(), "audio")
                elif hasattr(response, 'tool_call'):
                    await self.process_tool_call(response.tool_call)
            
        except Exception as e:
            logger.error(f"Error processing tool response: {e}")
            traceback.print_exc()
            await self.send_response(f"Error processing tool response: {str(e)}", "error")


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
                logger.warning(f"Failed to initialize Gemini session (attempt {attempt+1}/{max_retries}): {e}")
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
                
                # Check session duration first
                if not session_manager.check_session_duration(client_id):
                    await websocket.send(json.dumps({
                        "type": "error",
                        "content": "Session duration limit exceeded. Please reconnect to start a new session."
                    }))
                    break
                
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
    
    elif msg_type == 'tool_response':
        # Process tool response from client
        logger.info(f"Received tool response from client {client_id}")
        
        try:
            # Extract function responses
            function_responses = []
            if isinstance(content, list):
                function_responses = content
            elif isinstance(content, dict) and 'function_responses' in content:
                function_responses = content['function_responses']
            
            if not function_responses:
                logger.warning(f"Received empty tool response from client {client_id}")
                return
            
            # Send tool response to Gemini
            await client_app.process_tool_response(function_responses)
                
        except Exception as e:
            logger.error(f"Error processing tool response: {e}")
            traceback.print_exc()
            await websocket.send(json.dumps({
                "type": "error",
                "content": f"Error processing tool response: {str(e)}"
            }))
    
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
    
    elif msg_type == 'focus_region':
        # Set focus region for ADHD support
        logger.info(f"Client {client_id} setting ADHD focus region")
        
        # Parse region data (x, y, width, height)
        region = content
        
        # Update client app
        client_app.focus_region = region
        
        # Update user preferences in session
        if "user_preferences" in session:
            session["user_preferences"]["focus_region"] = region
        
        await websocket.send(json.dumps({
            "type": "status",
            "content": "Focus region set for ADHD support"
        }))
    
    elif msg_type == 'adhd_settings':
        # Update ADHD support settings
        logger.info(f"Client {client_id} updating ADHD settings")
        
        settings = content
        if isinstance(settings, dict):
            # Update client app settings
            if "enabled" in settings:
                client_app.adhd_support_enabled = settings["enabled"]
                session["user_preferences"]["adhd_support"] = settings["enabled"]
            
            if "reading_pace" in settings:
                client_app.reading_pace = settings["reading_pace"]
                session["user_preferences"]["reading_pace"] = settings["reading_pace"]
            
            await websocket.send(json.dumps({
                "type": "status",
                "content": "ADHD support settings updated"
            }))
    
    elif msg_type == 'flashcard_request':
        # Handle flashcard requests
        action = content.get('action', '')
        
        if action == 'list':
            # Return flashcard list
            category = content.get('category')
            flashcards = session_manager.get_flashcards(client_id, category)
            
            await websocket.send(json.dumps({
                "type": "flashcards",
                "content": flashcards
            }))
        
        elif action == 'add':
            # Add a new flashcard
            flashcard = content.get('flashcard', {})
            if flashcard and 'front' in flashcard and 'back' in flashcard:
                session_manager.add_flashcard(client_id, flashcard)
                
                await websocket.send(json.dumps({
                    "type": "status",
                    "content": "Flashcard added successfully"
                }))
    
    elif msg_type == 'medical_mode':
        # Set medical education mode
        is_medical = content.get('enabled', False)
        client_app.is_medical_context = is_medical
        
        # Update user preferences in session
        if "user_preferences" in session:
            session["user_preferences"]["is_medical_context"] = is_medical
        
        mode_name = "Medical education mode" if is_medical else "Standard mode"
        await websocket.send(json.dumps({
            "type": "status",
            "content": f"{mode_name} {'activated' if is_medical else 'deactivated'}"
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
