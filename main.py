import asyncio
import os
import sys
import time
import traceback
import hashlib
import argparse
import io
import logging
from typing import List, Optional, Tuple, Dict, Any, Union

import cv2
from google import genai
from google.genai import types
import mss
import numpy as np
import pyaudio
from PIL import Image, ImageEnhance, ImageFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('gemini-assistant')

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash")
DEFAULT_MODE = "camera"
CAPTURE_INTERVAL_MIN = 0.05  # min time between captures (seconds)
CAPTURE_INTERVAL_MAX = 0.5   # max time between captures (seconds)
DEFAULT_CAPTURE_INTERVAL = 0.2  # initial capture interval

# API key management
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize Gemini client with API key
client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=API_KEY)

# Configure audio response settings
CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "audio",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
)

# Initialize PyAudio
pya = pyaudio.PyAudio()

# User prompt that defines assistant behavior and constraints
SYSTEM_PROMPT = """
You are a helpful AI voice assistant. You can see what the user is showing you
through their camera or screen and respond to their questions about what you see.

For text you can see:
- If the user shows you a document or text on their screen, read it when asked.
- Summarize longer text content when appropriate.
- If asked to read something specific in the image, focus on that text.
- When reading text, present it in a logical order (top-to-bottom, left-to-right).
- For structured content like tables or forms, describe the structure first, then the content.

For images:
- Describe what you see when asked.
- Answer questions about objects, people, or scenes in the image.
- Be concise in your responses unless the user asks for more detail.

Text recognition commands:
- "Read this" or "What does it say?" - Read the text visible on screen
- "Summarize this page" - Provide a concise summary of visible text
- "Find [X] on this page" - Locate specific information in visible text

General guidelines:
- Be helpful, accurate, and concise in your responses.
- If you can't see something clearly, it's okay to say so.
- Don't make up information if you're uncertain.
- Respond conversationally and naturally.
- If the user asks you to remember something, do your best to keep it in mind.
"""

class ScreenCapture:
    """Enhanced screen capture with text recognition optimization."""
    
    def __init__(self, 
                 capture_interval: float = 0.2,
                 min_interval: float = 0.05,
                 max_interval: float = 0.5,
                 max_width: int = 1280):
        """
        Initialize screen capture with configurable parameters.
        
        Args:
            capture_interval: Initial interval between captures (seconds)
            min_interval: Minimum capture interval (seconds)
            max_interval: Maximum capture interval (seconds)
            max_width: Maximum width for resizing large captures
        """
        self.capture_interval = capture_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.max_width = max_width
        self.last_capture_time = 0
        self.last_image_hash = None
        self.text_region_cache = {}
        self.performance_metrics = {
            "capture_times": [],
            "processing_times": [],
            "transmission_times": [],
            "response_times": []
        }
    
    def adjust_capture_interval(self, avg_latency: float) -> None:
        """
        Adjust capture interval based on latency feedback.
        
        Args:
            avg_latency: Average latency from recent captures (seconds)
        """
        if avg_latency > 0.7:
            # If latency is high, slow down capture rate
            self.capture_interval = min(self.max_interval, self.capture_interval * 1.2)
            logger.info(f"Increasing capture interval to {self.capture_interval:.2f}s due to high latency")
        elif avg_latency < 0.3:
            # If latency is low, speed up capture rate
            self.capture_interval = max(self.min_interval, self.capture_interval * 0.9)
            logger.info(f"Decreasing capture interval to {self.capture_interval:.2f}s due to low latency")
    
    async def capture_screen(self, monitor_num: int = 1) -> Optional[Image.Image]:
        """
        Capture the screen and return as PIL Image.
        
        Args:
            monitor_num: Monitor number to capture (default: primary)
            
        Returns:
            PIL Image of the captured screen or None if skipped
        """
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.capture_interval:
            return None
            
        capture_start = time.time()
        
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[monitor_num]
                screenshot = sct.grab(monitor)
                img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
                
                # Resize if too large
                if img.width > self.max_width:
                    ratio = self.max_width / img.width
                    new_height = int(img.height * ratio)
                    img = img.resize((self.max_width, new_height))
                
                # Log capture time
                capture_time = time.time() - capture_start
                self.performance_metrics["capture_times"].append(capture_time)
                
                # Update last capture time
                self.last_capture_time = current_time
                
                # Check if screen has changed significantly
                img_array = np.array(img.resize((32, 32)))
                img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
                
                if img_hash == self.last_image_hash:
                    # Screen hasn't changed, capture less frequently
                    self.capture_interval = min(self.max_interval, self.capture_interval * 1.1)
                    return None
                    
                self.last_image_hash = img_hash
                return img
                
        except Exception as e:
            logger.error(f"Screen capture error: {e}")
            return None
    
    def enhance_text_visibility(self, image: Image.Image) -> Image.Image:
        """
        Enhance image to improve text recognition.
        
        Args:
            image: Original PIL Image
            
        Returns:
            Enhanced PIL Image
        """
        process_start = time.time()
        
        try:
            # Convert to OpenCV format for processing
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply slight blur to remove noise
            blurred = cv2.GaussianBlur(thresh, (3, 3), 0)
            
            # Sharpen the image
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(blurred, -1, kernel)
            
            # Convert back to PIL image
            enhanced = Image.fromarray(sharpened)
            
            # Log processing time
            process_time = time.time() - process_start
            self.performance_metrics["processing_times"].append(process_time)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing text visibility: {e}")
            # Return original if enhancement fails
            return image
    
    async def detect_text_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """
        Enhanced method to detect regions in the image that likely contain text.
        Uses a combination of MSER and improved OpenCV processing.
        
        Args:
            image: PIL Image
            
        Returns:
            List of (x, y, width, height) regions
        """
        # Use image hash for caching results
        img_array = np.array(image.resize((32, 32)))
        img_hash = hashlib.md5(img_array.tobytes()).hexdigest()
        
        if img_hash in self.text_region_cache:
            return self.text_region_cache[img_hash]
            
        try:
            # Convert to OpenCV format
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding for better text extraction
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Clean up the binary image with morphological operations
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Try using MSER for text detection
            try:
                mser = cv2.MSER_create()
                regions, _ = mser.detectRegions(gray)
                
                # Filter and merge text regions
                text_regions = []
                for region in regions:
                    x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                    
                    # Apply heuristics to identify potential text regions
                    aspect_ratio = w / h if h > 0 else 0
                    
                    if 0.1 < aspect_ratio < 15 and 10 < w < image.width * 0.8 and 8 < h < image.height * 0.2:
                        text_regions.append((x, y, w, h))
            except Exception:
                # Fallback to edge detection if MSER fails
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                text_regions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    if 0.1 < w/h < 10 and w > 20 and h > 10:
                        text_regions.append((x, y, w, h))
            
            # Merge overlapping regions
            text_regions = self._merge_overlapping_regions(text_regions)
            
            # Cache the results
            self.text_region_cache[img_hash] = text_regions
            
            # Manage cache size
            if len(self.text_region_cache) > 20:
                oldest_key = next(iter(self.text_region_cache))
                del self.text_region_cache[oldest_key]
                
            return text_regions
                
        except Exception as e:
            logger.error(f"Error detecting text regions: {e}")
            return []
    
    def _merge_overlapping_regions(self, regions: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping text regions.
        
        Args:
            regions: List of (x, y, width, height) regions
            
        Returns:
            List of merged regions
        """
        if not regions:
            return []
            
        # Sort by x-coordinate
        sorted_regions = sorted(regions, key=lambda r: r[0])
        merged_regions = [sorted_regions[0]]
        
        for current in sorted_regions[1:]:
            previous = merged_regions[-1]
            
            # Calculate overlap
            prev_x2 = previous[0] + previous[2]
            prev_y2 = previous[1] + previous[3]
            curr_x2 = current[0] + current[2]
            curr_y2 = current[1] + current[3]
            
            # Check for horizontal overlap
            h_overlap = (current[0] <= prev_x2) and (prev_x2 <= curr_x2)
            
            # Check for vertical overlap
            v_overlap = not ((current[1] >= prev_y2) or (curr_y2 <= previous[1]))
            
            if h_overlap and v_overlap:
                # Regions overlap, merge them
                x = min(previous[0], current[0])
                y = min(previous[1], current[1])
                w = max(prev_x2, curr_x2) - x
                h = max(prev_y2, curr_y2) - y
                
                merged_regions[-1] = (x, y, w, h)
            else:
                # No overlap, add current region
                merged_regions.append(current)
                
        return merged_regions
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get average performance metrics.
        
        Returns:
            Dictionary of average metric values
        """
        metrics = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                # Calculate average of last 10 values
                metrics[f"avg_{key}"] = sum(values[-10:]) / min(len(values), 10)
            else:
                metrics[f"avg_{key}"] = 0
                
        return metrics


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        """Initialize the audio loop with improved error handling."""
        self.video_mode = video_mode
        self.running = True
        self.audio_stream = None
        self.audio_error_count = 0  # Track audio errors for recovery
        
        try:
            self.audio_stream = pya.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SEND_SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE,
            )
        except OSError as e:
            logger.error(f"Error initializing audio stream: {e}")
            logger.info("Attempting to find alternative audio device...")
            # Try to find alternative audio device
            info = pya.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            for i in range(num_devices):
                if (pya.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels') > 0):
                    logger.info(f"Found input device: {pya.get_device_info_by_host_api_device_index(0, i).get('name')}")
                    try:
                        self.audio_stream = pya.open(
                            format=FORMAT,
                            channels=CHANNELS,
                            rate=SEND_SAMPLE_RATE,
                            input=True,
                            input_device_index=i,
                            frames_per_buffer=CHUNK_SIZE,
                        )
                        logger.info(f"Successfully initialized audio with device index {i}")
                        break
                    except OSError:
                        continue
        
        self.session = None
        self.audio_latencies: List[float] = []
        self.capture_interval = DEFAULT_CAPTURE_INTERVAL
        self.audio_in_queue = None
        self.last_capture_time = 0
        self.send_text_task = None
        
        # Initialize enhanced screen capture
        self.screen_capture = ScreenCapture(
            capture_interval=DEFAULT_CAPTURE_INTERVAL,
            min_interval=CAPTURE_INTERVAL_MIN,
            max_interval=CAPTURE_INTERVAL_MAX
        )
        
        # Store selected screen region (x, y, width, height) or None for full screen
        self.selected_region = None
        
        # Track command context
        self.last_command_context = {}

        # Initialize Gemini session with LiveConnect for audio
        self.connect = None
        self.chat_session = None
        
        # Command history
        self.command_history = CommandHistory()
        
        # Performance metrics
        self.performance_metrics = {
            "audio_latency": [],
            "frame_processing_time": [],
            "api_response_time": [],
            "audio_errors": 0,
            "video_errors": 0,
            "api_errors": 0
        }

    async def initialize_gemini_session(self):
        """Initialize Gemini LiveConnect session with audio support and retry mechanism."""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Initialize LiveConnect session with Gemini model
                self.connect = await asyncio.to_thread(
                    client.generate.live_connect, 
                    model=MODEL,
                    config=CONFIG
                )
                
                # Send initial system prompt
                response = await asyncio.to_thread(
                    self.connect.send,
                    SYSTEM_PROMPT
                )
                
                logger.info("Gemini: Ready to help with what you see!")
                return
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Error initializing Gemini session (attempt {attempt+1}/{max_retries}): {e}")
                    await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"Failed to initialize Gemini session after {max_retries} attempts: {e}")
                    traceback.print_exc()
                    raise

    # ---------------------------
    # Camera and Screen Capture
    # ---------------------------
    async def get_frames(self):
        """Capture frames from webcam and send to Gemini with improved error handling."""
        try:
            # Attempt to open camera with retry mechanism
            cap = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    cap = await asyncio.to_thread(cv2.VideoCapture, 0)
                    if cap.isOpened():
                        break
                    logger.warning(f"Failed to open webcam (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(1)
                except Exception as e:
                    logger.error(f"Error opening webcam: {e}")
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)
            
            if not cap or not cap.isOpened():
                logger.error("Could not open webcam after multiple attempts")
                return

            # Configure camera
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            consecutive_errors = 0
            while self.running:
                current_time = time.time()
                if current_time - self.last_capture_time < self.capture_interval:
                    await asyncio.sleep(0.01)  # Small pause to avoid busy-waiting
                    continue

                self.last_capture_time = current_time
                try:
                    ret, frame = cap.read()
                    if not ret:
                        consecutive_errors += 1
                        if consecutive_errors > 5:
                            logger.error("Multiple frame capture failures, attempting to reset camera")
                            cap.release()
                            cap = await asyncio.to_thread(cv2.VideoCapture, 0)
                            consecutive_errors = 0
                        continue
                    
                    consecutive_errors = 0
                    # Convert to RGB (from BGR) and create PIL Image
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    # Send to Gemini model
                    if self.connect:
                        # Convert PIL Image to bytes
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Create part with image MIME type
                        part = types.Part.blob(img_bytes, 'image/png')
                        
                        # Send to Gemini
                        start_time = time.time()
                        response = await asyncio.to_thread(
                            self.connect.send,
                            part
                        )
                        
                        # Track API response time
                        self.performance_metrics["api_response_time"].append(time.time() - start_time)
                        
                        # Print text response
                        for chunk in response:
                            if chunk.text:
                                print("\nGemini: ", chunk.text)
                                # Update command history
                                self.command_history.add_command("Image from camera", chunk.text[:100], {"type": "camera"})

                except Exception as e:
                    logger.error(f"Frame capture error: {e}")
                    self.performance_metrics["video_errors"] += 1
                    await asyncio.sleep(0.5)  # Brief pause before retry

        except Exception as e:
            logger.error(f"Camera capture error: {e}")
            traceback.print_exc()
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()

    async def get_screen(self):
        """Enhanced screen capture with improved error handling and performance monitoring."""
        try:
            last_error_time = 0
            consecutive_errors = 0
            
            while self.running:
                try:
                    # Capture screen with adaptive interval
                    img = await self.screen_capture.capture_screen()
                    
                    if img is None:
                        # No capture or no significant change
                        await asyncio.sleep(0.01)
                        continue
                    
                    # Process selected region if specified
                    if self.selected_region:
                        x, y, width, height = self.selected_region
                        img = img.crop((x, y, x + width, y + height))
                    
                    # Check for recent text reading commands
                    is_text_focused = False
                    if self.last_command_context.get('command_type') == 'read_text':
                        is_text_focused = True
                        # Clear the context after using it
                        if time.time() - self.last_command_context.get('timestamp', 0) > 10:
                            self.last_command_context = {}
                    
                    # Detect text regions for text-focused mode
                    text_regions = []
                    if is_text_focused:
                        text_regions = await self.screen_capture.detect_text_regions(img)
                    
                    # Get enhanced version optimized for text recognition
                    enhanced_img = None
                    if is_text_focused or 'read' in self.last_command_context.get('command', '').lower():
                        enhanced_img = self.screen_capture.enhance_text_visibility(img)
                    
                    # Track capture and processing time
                    capture_time = time.time()
                    
                    if self.connect:
                        # Convert images to bytes and send to Gemini
                        if is_text_focused and enhanced_img:
                            # Prepare text prompt and enhanced image for text reading
                            prompt = "This is a screen capture that contains text. I've processed it to make the text more readable. Please focus on reading and interpreting any text visible in the image."
                            
                            # Send text prompt first
                            await asyncio.to_thread(
                                self.connect.send,
                                prompt
                            )
                            
                            # Convert enhanced image to bytes
                            img_byte_arr = io.BytesIO()
                            enhanced_img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            
                            # Create part with image MIME type
                            part = types.Part.blob(img_bytes, 'image/png')
                            
                            # Send enhanced image
                            response = await asyncio.to_thread(
                                self.connect.send,
                                part
                            )
                            
                        elif enhanced_img:
                            # Send both original and enhanced images with explanation
                            prompt = "This is a screen capture. I'm including both the original image and an enhanced version to help you read any text. Please focus on any text content when responding."
                            
                            # Send text prompt first
                            await asyncio.to_thread(
                                self.connect.send,
                                prompt
                            )
                            
                            # Send original image
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            img_part = types.Part.blob(img_bytes, 'image/png')
                            
                            await asyncio.to_thread(
                                self.connect.send,
                                img_part
                            )
                            
                            # Send enhanced image
                            enhanced_byte_arr = io.BytesIO()
                            enhanced_img.save(enhanced_byte_arr, format='PNG')
                            enhanced_bytes = enhanced_byte_arr.getvalue()
                            enhanced_part = types.Part.blob(enhanced_bytes, 'image/png')
                            
                            response = await asyncio.to_thread(
                                self.connect.send,
                                enhanced_part
                            )
                            
                        else:
                            # Standard image capture
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            part = types.Part.blob(img_bytes, 'image/png')
                            
                            response = await asyncio.to_thread(
                                self.connect.send,
                                part
                            )
                        
                        # Print text response
                        response_text = ""
                        for chunk in response:
                            if chunk.text:
                                response_text += chunk.text
                        
                        if response_text:
                            print("\nGemini: ", response_text)
                            # Update command history
                            self.command_history.add_command("Screen capture", response_text[:100], {"type": "screen"})
                        
                        # Calculate response time
                        response_time = time.time() - capture_time
                        
                        # Update performance metrics
                        self.screen_capture.performance_metrics["response_times"].append(response_time)
                        self.audio_latencies.append(response_time)
                        
                        # Adjust capture interval based on performance
                        metrics = self.screen_capture.get_performance_metrics()
                        self.capture_interval = self.screen_capture.capture_interval
                        
                        # Reset consecutive errors on success
                        consecutive_errors = 0
                    
                except Exception as e:
                    current_time = time.time()
                    consecutive_errors += 1
                    self.performance_metrics["video_errors"] += 1
                    
                    # Limit logging to prevent log spam
                    if current_time - last_error_time > 5:  # Log only once every 5 seconds
                        logger.error(f"Screen capture error: {e}")
                        last_error_time = current_time
                    
                    # If we have persistent errors, slow down to reduce resource usage
                    if consecutive_errors > 5:
                        await asyncio.sleep(1.0)  # Longer pause on persistent errors
                    else:
                        await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Enhanced screen capture error: {e}")
            traceback.print_exc()

    # ---------------------------
    # Text Input Handling
    # ---------------------------
    async def send_text(self):
        """Handle user text input with improved UX and error handling."""
        try:
            # Initialize Gemini LiveConnect session
            await self.initialize_gemini_session()
            
            print("\nGemini Voice Assistant running. Enter 'q' to quit.")
            print("Speak or type your question (or 'q' to quit):\n")
            
            while self.running:
                try:
                    user_input = await asyncio.to_thread(input, "")
                    
                    if user_input.lower() == 'q':
                        self.running = False
                        raise asyncio.CancelledError("User requested quit")
                    
                    if not user_input:
                        continue
                    
                    print(f"\nYou typed: {user_input}")
                    
                    # Check for screen reading commands
                    lower_input = user_input.lower()
                    if any(cmd in lower_input for cmd in ['read this', 'read the text', 'what does it say']):
                        # Mark this as a text reading command
                        self.last_command_context = {
                            'command': user_input,
                            'command_type': 'read_text',
                            'timestamp': time.time()
                        }
                        
                        # For text reading commands, don't need to send text separately
                        # The next screen capture will include the enhanced text processing
                        print("\nFocusing on text in next screen capture...")
                        continue
                    
                    # For region selection commands
                    if 'focus on region' in lower_input:
                        print("\nType the coordinates as x,y,width,height or 'reset' to clear region:")
                        region_input = await asyncio.to_thread(input, "")
                        
                        if region_input.lower() == 'reset':
                            self.selected_region = None
                            print("Cleared region selection. Using full screen.")
                        else:
                            try:
                                coords = [int(x.strip()) for x in region_input.split(',')]
                                if len(coords) == 4:
                                    self.selected_region = tuple(coords)
                                    print(f"Set region to {self.selected_region}")
                                else:
                                    print("Invalid format. Expected x,y,width,height")
                            except ValueError:
                                print("Invalid coordinates. Expected integers.")
                        continue
                    
                    # Handle help command
                    if lower_input in ['help', '?', 'commands']:
                        print("\nAvailable commands:")
                        print("- 'read this' or 'what does it say': Read text from the screen")
                        print("- 'focus on region': Select a specific region of the screen")
                        print("- 'reset region': Reset to full screen")
                        print("- 'q': Quit the application")
                        continue
                    
                    # Send text to Gemini
                    if self.connect:
                        try:
                            start_time = time.time()
                            response = await asyncio.to_thread(
                                self.connect.send,
                                user_input
                            )
                            
                            # Print text response
                            response_text = ""
                            for chunk in response:
                                if chunk.text:
                                    response_text += chunk.text
                            
                            if response_text:
                                print("\nGemini: ", response_text)
                                # Update command history
                                self.command_history.add_command(user_input, response_text[:100], {"type": "text"})
                            
                            # Track response time for performance monitoring
                            self.performance_metrics["api_response_time"].append(time.time() - start_time)
                            
                        except Exception as e:
                            logger.error(f"Error sending text to Gemini: {e}")
                            self.performance_metrics["api_errors"] += 1
                            print(f"\nError communicating with Gemini: {e}")
                except EOFError:
                    logger.info("EOF received on input")
                    break
                except Exception as e:
                    logger.error(f"Error handling text input: {e}")
                    await asyncio.sleep(1)
                    
        except EOFError:
            self.running = False
            raise asyncio.CancelledError("EOF on input")
        except Exception as e:
            logger.error(f"Fatal error in text input handler: {e}")
            traceback.print_exc()
            self.running = False

    # ---------------------------
    # Audio Handling
    # ---------------------------
    async def send_realtime(self):
        """Stream audio to Gemini with improved voice detection and error handling."""
        try:
            # Buffer for collecting audio chunks
            audio_buffer = []
            is_speaking = False
            silence_count = 0
            consecutive_errors = 0
            
            # Adaptive silence threshold
            SILENCE_THRESHOLD = 500  # Initial value, will be adjusted based on environment
            background_noise = []
            
            while self.running:
                if self.audio_stream:
                    # Read audio chunk
                    try:
                        data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        
                        # Convert to numpy array for analysis
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        
                        # Calculate volume
                        volume = np.abs(audio_array).mean()
                        
                        # Calibrate background noise if we have few samples
                        if len(background_noise) < 10 and not is_speaking:
                            background_noise.append(volume)
                            if len(background_noise) == 10:
                                # Set threshold above background noise
                                bg_mean = sum(background_noise) / len(background_noise)
                                SILENCE_THRESHOLD = max(500, bg_mean * 2.5)
                                logger.info(f"Calibrated silence threshold: {SILENCE_THRESHOLD}")
                        
                        if volume > SILENCE_THRESHOLD:
                            # Reset silence counter when sound detected
                            silence_count = 0
                            consecutive_errors = 0  # Reset error counter on successful read
                            
                            if not is_speaking:
                                is_speaking = True
                                print("\nListening...")
                            
                            # Add to buffer
                            audio_buffer.append(data)
                            
                        elif is_speaking:
                            # Still add some silence for natural pauses
                            audio_buffer.append(data)
                            silence_count += 1
                            
                            # If silence persists, assume speaking ended
                            if silence_count > 20:  # About 0.5 seconds of silence
                                is_speaking = False
                                
                                # Process collected audio
                                if audio_buffer and self.connect:
                                    print("Processing audio...")
                                    
                                    # Combine audio chunks
                                    audio_data = b''.join(audio_buffer)
                                    
                                    try:
                                        # LiveConnect supports audio input
                                        audio_part = types.Part.audio(audio_data, mime_type="audio/raw;encoding=signed-integer;bits_per_sample=16;sample_rate=16000")
                                        
                                        # Send audio to Gemini
                                        start_time = time.time()
                                        response = await asyncio.to_thread(
                                            self.connect.send,
                                            audio_part
                                        )
                                        
                                        # Print text response
                                        response_text = ""
                                        for chunk in response:
                                            if chunk.text:
                                                response_text += chunk.text
                                        
                                        if response_text:
                                            print("\nGemini: ", response_text)
                                            # Update command history
                                            self.command_history.add_command("Voice input", response_text[:100], {"type": "voice"})
                                            
                                        # Track response time for performance monitoring
                                        self.performance_metrics["api_response_time"].append(time.time() - start_time)
                                            
                                    except Exception as e:
                                        logger.error(f"Error sending audio to Gemini: {e}")
                                        self.performance_metrics["api_errors"] += 1
                                
                                # Clear buffer for next utterance
                                audio_buffer = []
                    
                    except Exception as e:
                        consecutive_errors += 1
                        self.performance_metrics["audio_errors"] += 1
                        
                        # Limit logging to prevent log spam
                        if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                            logger.error(f"Audio reading error ({consecutive_errors}): {e}")
                        
                        # Implement progressive recovery for audio stream issues
                        if consecutive_errors > 5:
                            logger.warning("Too many audio errors, attempting to reset audio stream...")
                            try:
                                if self.audio_stream:
                                    self.audio_stream.close()
                                
                                # Try to reopen the audio stream
                                self.audio_stream = pya.open(
                                    format=FORMAT,
                                    channels=CHANNELS,
                                    rate=SEND_SAMPLE_RATE,
                                    input=True,
                                    frames_per_buffer=CHUNK_SIZE,
                                )
                                consecutive_errors = 0
                                logger.info("Audio stream reset successful")
                            except Exception as reset_error:
                                logger.error(f"Failed to reset audio stream: {reset_error}")
                                # Add exponential backoff
                                await asyncio.sleep(min(1 * (2 ** min(consecutive_errors - 5, 5)), 30))
                
                # Small sleep to avoid busy waiting
                await asyncio.sleep(0.01)
                    
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
            traceback.print_exc()
    
    async def listen_audio(self):
        """Monitoring for audio processing."""
        # Most audio handling is now in send_realtime
        while self.running:
            await asyncio.sleep(1.0)
    
    async def receive_audio(self):
        """
        Audio responses are handled automatically by LiveConnect.
        This is a placeholder for compatibility and for handling special cases.
        """
        while self.running:
            await asyncio.sleep(1.0)
    
    async def play_audio(self):
        """
        Audio playback is handled automatically by LiveConnect.
        This is a placeholder for compatibility and for handling special cases.
        """
        self.audio_in_queue = asyncio.Queue()  # For compatibility
        while self.running:
            await asyncio.sleep(1.0)

    # ---------------------------
    # Performance Monitor
    # ---------------------------
    async def monitor_performance(self):
        """
        Enhanced performance monitoring with adaptive behavior.
        Periodically checks metrics and adjusts settings for optimal performance.
        """
        while self.running:
            await asyncio.sleep(10.0)  # Check every 10 seconds
            try:
                # Calculate average metrics
                avg_metrics = {}
                for metric_name, values in self.performance_metrics.items():
                    if isinstance(values, list) and values:
                        # Keep only the most recent values
                        if len(values) > 50:
                            self.performance_metrics[metric_name] = values[-20:]
                        avg = sum(self.performance_metrics[metric_name]) / len(self.performance_metrics[metric_name])
                        avg_metrics[f"avg_{metric_name}"] = avg
                    elif isinstance(values, int):
                        avg_metrics[metric_name] = values
                
                # Print performance report
                if self.video_mode == "screen":
                    screen_metrics = self.screen_capture.get_performance_metrics()
                    metrics_str = ", ".join([f"{k}: {v:.3f}s" for k, v in screen_metrics.items() 
                                           if not k.startswith("_")])
                    logger.info(f"Performance metrics: {metrics_str}")
                    
                    # Report errors if any
                    error_metrics = ", ".join([f"{k}: {v}" for k, v in avg_metrics.items() 
                                             if "error" in k and v > 0])
                    if error_metrics:
                        logger.warning(f"Error metrics: {error_metrics}")
                else:
                    if "avg_api_response_time" in avg_metrics:
                        logger.info(f"Avg API response time: {avg_metrics['avg_api_response_time']:.2f}s")
                
                # Adaptive behavior
                if "avg_api_response_time" in avg_metrics:
                    avg_latency = avg_metrics["avg_api_response_time"]
                    
                    # Adjust capture interval based on latency
                    if (avg_latency > 0.7):
                        self.capture_interval = min(
                            CAPTURE_INTERVAL_MAX, self.capture_interval * 1.2
                        )
                        logger.info(f"Increasing capture interval to {self.capture_interval:.2f}s due to high latency")
                    elif avg_latency < 0.3:
                        self.capture_interval = max(
                            CAPTURE_INTERVAL_MIN, self.capture_interval * 0.9
                        )
                        logger.info(f"Decreasing capture interval to {self.capture_interval:.2f}s due to low latency")
                
                # Check for system issues
                if avg_metrics.get("audio_errors", 0) > 5:
                    logger.warning("High number of audio errors detected. Please check your audio device.")
                if avg_metrics.get("video_errors", 0) > 5:
                    logger.warning("High number of video errors detected. Please check your camera/screen settings.")
                if avg_metrics.get("api_errors", 0) > 5:
                    logger.warning("High number of API errors detected. Please check your network connection.")
                    
                # Reset counters for next period
                self.performance_metrics["audio_errors"] = 0
                self.performance_metrics["video_errors"] = 0
                self.performance_metrics["api_errors"] = 0
                
                # Clear old latency data
                if len(self.audio_latencies) > 50:
                    self.audio_latencies = self.audio_latencies[-20:]
                    
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

    # ---------------------------
    # Main Run Method
    # ---------------------------
    async def run(self):
        try:
            logger.info(f"Starting Gemini Voice Assistant in {self.video_mode} mode")
            
            # Create a task group to manage all concurrent tasks
            async with asyncio.TaskGroup() as tg:
                # Start text input handling
                self.send_text_task = tg.create_task(self.send_text())
                
                # Start audio streaming
                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())

                # Launch either camera or screen capture (or none)
                if self.video_mode == "camera":
                    logger.info("Initializing camera capture")
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    logger.info("Initializing screen capture")
                    tg.create_task(self.get_screen())
                else:
                    logger.info("Running in audio-only mode")

                # Start audio output handling
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                
                # Start performance monitoring
                tg.create_task(self.monitor_performance())

                # Wait for all tasks to complete
                await self.send_text_task

        except asyncio.CancelledError as c:
            # Graceful exit on user request
            logger.info(f"Shutting down: {c}")
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
            traceback.print_exc()
        finally:
            self.running = False
            logger.info("Cleaning up resources...")
            
            # Close audio stream
            if self.audio_stream is not None:
                try:
                    self.audio_stream.close()
                except Exception as e:
                    logger.error(f"Error closing audio stream: {e}")
            
            # Close Gemini LiveConnect session
            if self.connect:
                try:
                    self.connect.close()
                except Exception as e:
                    logger.error(f"Error closing Gemini connection: {e}")
                    
            logger.info("Cleaned up resources. Goodbye!")


class CommandHistory:
    """Tracks command history for context-aware responses."""
    
    def __init__(self, max_size=20):
        self.commands = []
        self.max_size = max_size
    
    def add_command(self, command, response, metadata=None):
        """Add a command to history."""
        entry = {
            "command": command,
            "response": response,
            "timestamp": time.time(),
            "metadata": metadata or {}
        }
        
        self.commands.append(entry)
        
        # Keep history within size limit
        if len(self.commands) > self.max_size:
            self.commands = self.commands[-self.max_size:]
    
    def get_recent_commands(self, count=5):
        """Get most recent commands."""
        return self.commands[-count:]
    
    def get_command_context(self):
        """Generate context from command history for the model."""
        if not self.commands:
            return ""
        
        context = "Here's what we've been discussing:\n"
        for i, entry in enumerate(self.commands[-3:]):  # Last 3 commands
            context += f"- You: {entry['command']}\n"
            context += f"- Assistant: {entry['response'][:100]}{'...' if len(entry['response']) > 100 else ''}\n"
        
        return context


# ---------------------------
# CLI Entry Point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemini Voice Reading Assistant")
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="Type of capture: camera, screen, or none",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log to file instead of console",
    )
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    if args.log_file:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=args.log_file,
            filemode='a'
        )
    else:
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
    
    logger.info(f"Starting Gemini Voice Reading Assistant in {args.mode} mode")
    
    main = AudioLoop(video_mode=args.mode)
    
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
