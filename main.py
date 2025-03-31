import asyncio
import os
import sys
import time
import traceback
import hashlib
import argparse
import io
import logging
import json
import re
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
MODEL = os.environ.get("GEMINI_MODEL", "models/gemini-2.0-flash-exp")  # Updated to exp model
DEFAULT_MODE = "camera"
CAPTURE_INTERVAL_MIN = 0.05  # min time between captures (seconds)
CAPTURE_INTERVAL_MAX = 0.5   # max time between captures (seconds)
DEFAULT_CAPTURE_INTERVAL = 0.2  # initial capture interval

# ADHD support settings
ADHD_SUPPORT_ENABLED = True
FOCUS_ENHANCEMENT_LEVEL = 2  # 1-3 scale for focus enhancement intensity
READING_PACE = "moderate"     # slow, moderate, fast
NOTIFICATION_STYLE = "subtle" # none, subtle, prominent

# API key management
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    logger.error("GEMINI_API_KEY environment variable not set")
    sys.exit(1)

# Initialize Gemini client with API key
client = genai.Client(http_options={"api_version": "v1alpha"}, api_key=API_KEY)

# Example tool definitions for function calling
TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g., San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            },
            {
                "name": "set_alarm",
                "description": "Set an alarm for a specific time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {
                            "type": "string",
                            "description": "The time to set the alarm for, in HH:MM format"
                        }
                    },
                    "required": ["time"]
                }
            },
            {
                "name": "create_flashcard",
                "description": "Create a flashcard for study purposes",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "front": {
                            "type": "string",
                            "description": "Content for the front of the flashcard (question)"
                        },
                        "back": {
                            "type": "string",
                            "description": "Content for the back of the flashcard (answer)"
                        },
                        "category": {
                            "type": "string",
                            "description": "Medical category for the flashcard"
                        }
                    },
                    "required": ["front", "back"]
                }
            }
        ]
    }
]

# Configure audio response settings with tool support
CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "audio",
    ],
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
        )
    ),
    tools=TOOLS
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

Special Educational Support (ADHD):
- When assisting with studying, maintain a structured, engaging approach
- For medical content, be exceptionally precise with terminology
- During flashcard sessions, provide clear feedback and encouragement
- Adjust reading pace and tone to maintain engagement
- Highlight important concepts and provide context when needed
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
        self.last_image = None
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
    
    def enhance_medical_text(self, image: Image.Image) -> Image.Image:
        """
        Enhance image to improve medical text recognition.
        
        Args:
            image: Original PIL Image
            
        Returns:
            Enhanced PIL Image optimized for medical terminology
        """
        process_start = time.time()
        
        try:
            # Convert to OpenCV format for processing
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply adaptive thresholding optimized for medical text
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Medical-specific noise reduction while preserving details
            denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
            
            # Convert back to PIL image
            enhanced = Image.fromarray(denoised)
            
            # Log processing time
            process_time = time.time() - process_start
            self.performance_metrics["processing_times"].append(process_time)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing medical text: {e}")
            # Return original if enhancement fails
            return image
    
    def enhance_focus_region(self, image: Image.Image, region_of_interest: Tuple[int, int, int, int]) -> Image.Image:
        """
        Apply visual highlighting to focus area for ADHD support.
        
        Args:
            image: Original PIL Image
            region_of_interest: (x, y, width, height) of focus region
            
        Returns:
            Image with enhanced focus region
        """
        try:
            # Convert to OpenCV format
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Create mask for focus area
            mask = np.zeros(img_cv.shape[:2], dtype=np.uint8)
            x, y, w, h = region_of_interest
            mask[y:y+h, x:x+w] = 255
            
            # Enhance contrast in focus area
            focused_region = cv2.bitwise_and(img_cv, img_cv, mask=mask)
            
            # Apply enhancement based on configured level
            if FOCUS_ENHANCEMENT_LEVEL >= 2:
                # Increase contrast in focus area
                lab = cv2.cvtColor(focused_region, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_lab = cv2.merge((cl, a, b))
                focused_region = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Reduce visual noise outside focus area
            background = cv2.bitwise_and(img_cv, img_cv, mask=cv2.bitwise_not(mask))
            
            # Adjust blur based on enhancement level
            blur_amount = 21
            if FOCUS_ENHANCEMENT_LEVEL == 1:
                blur_amount = 11
            elif FOCUS_ENHANCEMENT_LEVEL == 3:
                blur_amount = 31
                
            background = cv2.GaussianBlur(background, (blur_amount, blur_amount), 0)
            
            # Reduce opacity of background based on enhancement level
            if FOCUS_ENHANCEMENT_LEVEL >= 2:
                background = cv2.addWeighted(background, 0.7, np.zeros_like(background), 0, 0)
            
            # Combine for ADHD-optimized viewing
            result = cv2.add(focused_region, background)
            
            # Convert back to PIL
            enhanced = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing focus region: {e}")
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
    
    def detect_content_regions(self, image: Image.Image) -> List[Dict[str, Any]]:
        """
        Detect and classify content regions in medical study materials.
        
        Args:
            image: PIL Image of screen content
            
        Returns:
            List of region dictionaries with type classification
        """
        try:
            # Convert to OpenCV format
            img_cv = np.array(image)
            img_cv = img_cv[:, :, ::-1].copy()  # RGB to BGR
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing to improve region detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
            
            # Perform morphological operations to enhance text regions
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours for content blocks
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process and filter contours to identify content blocks
            content_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small regions
                if w < 100 or h < 20:
                    continue
                    
                # Classify region type based on position and characteristics
                region_type = self._classify_region_type(img_cv[y:y+h, x:x+w], x, y, image.width, image.height)
                
                content_regions.append({
                    "coords": (x, y, w, h),
                    "type": region_type,
                    "area": w * h
                })
            
            # Sort by vertical position (top to bottom)
            content_regions.sort(key=lambda r: r["coords"][1])
            
            return content_regions
            
        except Exception as e:
            logger.error(f"Error detecting content regions: {e}")
            return []
    
    def _classify_region_type(self, region_img, x, y, img_width, img_height):
        """
        Classify the type of content region (question, answer, figure, etc.)
        """
        # Simple heuristic classification based on position and content
        region_height = region_img.shape[0]
        region_width = region_img.shape[1]
        
        # Check for question markers (Q:, 1., etc.)
        question_markers = self._detect_question_markers(region_img)
        
        # Check for answer markers (A:, ans:, etc.)
        answer_markers = self._detect_answer_markers(region_img)
        
        # Calculate relative position (useful for flashcards/split screens)
        relative_y = y / img_height
        
        if question_markers:
            return "question"
        elif answer_markers:
            return "answer"
        elif region_height > 100 and region_width > 200:
            if relative_y < 0.4:
                return "header"
            elif relative_y > 0.7:
                return "footer"
            else:
                return "content"
        else:
            return "unknown"
    
    def _detect_question_markers(self, region_img):
        """Detect markers indicating a question"""
        # Convert region to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if len(region_img.shape) > 2 else region_img
        
        # Apply binary thresholding for clearer text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Simple template matching approach for common question markers
        # For a real application, you'd use a proper OCR here
        # This is a simplified implementation
        
        # Check for "Q" or "1." like patterns
        # Using simple structural analysis
        left_aligned = np.sum(thresh[:, :20] == 0) > np.sum(thresh[:, 20:40] == 0)
        
        # Numbers 1-9 followed by period or parenthesis pattern
        digit_pattern = self._has_digit_pattern(thresh)
        
        return left_aligned and digit_pattern

    def _detect_answer_markers(self, region_img):
        """Detect markers indicating an answer"""
        # Convert region to grayscale
        gray = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY) if len(region_img.shape) > 2 else region_img
        
        # Apply binary thresholding for clearer text
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Check for "A" or "Answer" patterns
        # Using simple structural analysis
        left_aligned = np.sum(thresh[:, :20] == 0) > np.sum(thresh[:, 20:40] == 0)
        
        # Check for specific patterns like "A:" or "Answer:"
        # This is a simplified implementation
        has_letter_patterns = self._has_letter_pattern(thresh)
        
        return left_aligned and has_letter_patterns
    
    def _has_digit_pattern(self, binary_image):
        """
        Check if the image likely contains a digit pattern 
        like '1.', '2)', etc. at the beginning
        
        This is a simplified approach without full OCR
        """
        # Simple structural analysis looking for digit-like patterns
        # in the first few columns of the image
        h, w = binary_image.shape
        
        # Extract left region where digits would be
        left_region = binary_image[:, :min(30, w//4)]
        
        # Count black pixels
        black_pixel_count = np.sum(left_region == 0)
        
        # Digit patterns typically have a moderate density of black pixels
        pixel_density = black_pixel_count / (left_region.shape[0] * left_region.shape[1])
        
        return 0.05 < pixel_density < 0.3
    
    def _has_letter_pattern(self, binary_image):
        """
        Check if the image likely contains typical answer markers
        like 'A)', 'B:', 'Answer:', etc. at the beginning
        
        This is a simplified approach without full OCR
        """
        # Simple structural analysis looking for letter-like patterns
        h, w = binary_image.shape
        
        # Extract left region where letters would be
        left_region = binary_image[:, :min(50, w//3)]
        
        # Count black pixels
        black_pixel_count = np.sum(left_region == 0)
        
        # Letter patterns typically have a moderate density of black pixels
        pixel_density = black_pixel_count / (left_region.shape[0] * left_region.shape[1])
        
        # Combined with vertical alignment patterns typical of letters
        # (This is highly simplified)
        vertical_pattern = np.mean(np.std(left_region, axis=0))
        
        return 0.05 < pixel_density < 0.25 and vertical_pattern > 40
    
    def detect_page_turn(self, current_frame, previous_frame=None):
        """
        Efficiently detect significant content changes like page turns.
        
        Args:
            current_frame: Current screen image
            previous_frame: Previous screen image
            
        Returns:
            Boolean indicating if a page turn was detected
        """
        if previous_frame is None or self.last_image is None:
            self.last_image = current_frame
            return False
            
        try:
            # Convert frames to numpy arrays if they're PIL images
            if isinstance(current_frame, Image.Image):
                current_np = np.array(current_frame)
            else:
                current_np = current_frame
                
            if isinstance(previous_frame, Image.Image):
                previous_np = np.array(previous_frame)
            else:
                previous_np = previous_frame
            
            # Convert frames to grayscale for faster processing
            current_gray = cv2.cvtColor(current_np, cv2.COLOR_RGB2GRAY) if len(current_np.shape) > 2 else current_np
            previous_gray = cv2.cvtColor(previous_np, cv2.COLOR_RGB2GRAY) if len(previous_np.shape) > 2 else previous_np
            
            # Resize for faster comparison
            current_small = cv2.resize(current_gray, (32, 32))
            previous_small = cv2.resize(previous_gray, (32, 32))
            
            # Compute difference
            diff = cv2.absdiff(current_small, previous_small)
            
            # Calculate change percentage
            change_percentage = np.sum(diff > 30) / diff.size
            
            # Update last image
            self.last_image = current_frame
            
            # Return True if change exceeds threshold
            return change_percentage > 0.3
            
        except Exception as e:
            logger.error(f"Error detecting page turn: {e}")
            return False
    
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
        self.last_captured_img = None
        
        # ADHD support settings
        self.adhd_support_enabled = ADHD_SUPPORT_ENABLED
        self.focus_region = None
        self.is_medical_context = False
        self.flashcard_mode = False
        self.current_flashcard = None
        
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
                # Initialize LiveConnect session with Gemini model - updated to use client.aio
                self.connect = await client.aio.live.connect(
                    model=MODEL,
                    config=CONFIG
                )
                
                # Send initial system prompt
                await self.connect.send(input=SYSTEM_PROMPT)
                
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
                    
                    # Store last captured image for page turn detection
                    self.last_captured_img = img

                    # Send to Gemini model
                    if self.connect:
                        # Apply focus enhancement if region is specified (ADHD support)
                        if self.adhd_support_enabled and self.focus_region:
                            img = self.screen_capture.enhance_focus_region(img, self.focus_region)
                        
                        # Convert PIL Image to bytes
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='PNG')
                        img_bytes = img_byte_arr.getvalue()
                        
                        # Create part with image MIME type
                        part = types.Part.blob(img_bytes, 'image/png')
                        
                        # Send to Gemini
                        start_time = time.time()
                        await self.connect.send(input=part)
                        
                        # Track API response time
                        self.performance_metrics["api_response_time"].append(time.time() - start_time)
                        
                        # Print text response
                        async for chunk in self.connect.receive():
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
                    
                    # Check for page turn (for ADHD support and medical education)
                    page_turn_detected = False
                    if self.adhd_support_enabled and self.last_captured_img is not None:
                        page_turn_detected = self.screen_capture.detect_page_turn(img, self.last_captured_img)
                        
                        if page_turn_detected and NOTIFICATION_STYLE != "none":
                            print("\nNew page detected.")
                            if self.connect and NOTIFICATION_STYLE == "prominent":
                                await self.connect.send(input="I notice you've turned to a new page. Would you like me to read it?")
                        
                    # Store current image for next comparison
                    self.last_captured_img = img
                    
                    # Process selected region if specified
                    if self.selected_region:
                        x, y, width, height = self.selected_region
                        img = img.crop((x, y, x + width, y + height))
                    
                    # For ADHD focus support
                    if self.adhd_support_enabled and self.focus_region:
                        img = self.screen_capture.enhance_focus_region(img, self.focus_region)
                    
                    # Check for recent text reading commands
                    is_text_focused = False
                    is_medical = self.is_medical_context
                    
                    if self.last_command_context.get('command_type') == 'read_text':
                        is_text_focused = True
                        # Clear the context after using it
                        if time.time() - self.last_command_context.get('timestamp', 0) > 10:
                            self.last_command_context = {}
                    
                    # For medical text detection
                    if self.last_command_context.get('content_type') == 'medical':
                        is_medical = True
                        
                    # Detect content regions for medical materials
                    content_regions = []
                    if is_medical:
                        content_regions = self.screen_capture.detect_content_regions(img)
                        
                        # Extract question/answer regions for flashcard mode
                        if self.flashcard_mode:
                            question_regions = [r for r in content_regions if r['type'] == 'question']
                            answer_regions = [r for r in content_regions if r['type'] == 'answer']
                            
                            if question_regions and self.last_command_context.get('command_type') == 'show_question':
                                # For flashcard question, focus on question region and blur answer
                                question_region = question_regions[0]['coords']
                                self.focus_region = question_region
                                img = self.screen_capture.enhance_focus_region(img, question_region)
                            elif answer_regions and self.last_command_context.get('command_type') == 'show_answer':
                                # For flashcard answer, focus on answer region
                                answer_region = answer_regions[0]['coords']
                                self.focus_region = answer_region
                                img = self.screen_capture.enhance_focus_region(img, answer_region)
                    
                    # Detect text regions for text-focused mode
                    text_regions = []
                    if is_text_focused:
                        text_regions = await self.screen_capture.detect_text_regions(img)
                    
                    # Get enhanced version optimized for text recognition
                    enhanced_img = None
                    if is_text_focused:
                        if is_medical:
                            # Use specialized medical text enhancement
                            enhanced_img = self.screen_capture.enhance_medical_text(img)
                        else:
                            # Use standard text enhancement
                            enhanced_img = self.screen_capture.enhance_text_visibility(img)
                    elif 'read' in self.last_command_context.get('command', '').lower():
                        enhanced_img = self.screen_capture.enhance_text_visibility(img)
                    
                    # Track capture and processing time
                    capture_time = time.time()
                    
                    if self.connect:
                        # Convert images to bytes and send to Gemini
                        if is_text_focused and enhanced_img:
                            # Prepare text prompt and enhanced image for text reading
                            prompt = "This is a screen capture that contains text. "
                            
                            if is_medical:
                                prompt += "This is medical educational content that requires precise reading of terminology. "
                            
                            prompt += "I've processed it to make the text more readable. Please focus on reading and interpreting any text visible in the image."
                            
                            # Send text prompt first
                            await self.connect.send(input=prompt)
                            
                            # Convert enhanced image to bytes
                            img_byte_arr = io.BytesIO()
                            enhanced_img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            
                            # Create part with image MIME type
                            part = types.Part.blob(img_bytes, 'image/png')
                            
                            # Send enhanced image
                            await self.connect.send(input=part)
                            
                        elif enhanced_img:
                            # Send both original and enhanced images with explanation
                            prompt = "This is a screen capture. I'm including both the original image and an enhanced version to help you read any text. Please focus on any text content when responding."
                            
                            # Send text prompt first
                            await self.connect.send(input=prompt)
                            
                            # Send original image
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            img_part = types.Part.blob(img_bytes, 'image/png')
                            
                            await self.connect.send(input=img_part)
                            
                            # Send enhanced image
                            enhanced_byte_arr = io.BytesIO()
                            enhanced_img.save(enhanced_byte_arr, format='PNG')
                            enhanced_bytes = enhanced_byte_arr.getvalue()
                            enhanced_part = types.Part.blob(enhanced_bytes, 'image/png')
                            
                            await self.connect.send(input=enhanced_part)
                            
                        else:
                            # Standard image capture
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            img_bytes = img_byte_arr.getvalue()
                            part = types.Part.blob(img_bytes, 'image/png')
                            
                            await self.connect.send(input=part)
                        
                        # Print text response
                        response_text = ""
                        async for chunk in self.connect.receive():
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
                    
                    # ADHD and medical education specific commands
                    if 'medical mode' in lower_input or 'usmle mode' in lower_input:
                        self.is_medical_context = True
                        print("\nMedical education mode activated. Text processing optimized for medical terminology.")
                        continue
                        
                    elif 'standard mode' in lower_input:
                        self.is_medical_context = False
                        print("\nStandard mode activated.")
                        continue
                        
                    elif 'flashcard mode' in lower_input:
                        self.flashcard_mode = True
                        print("\nFlashcard mode activated. I'll help you study with flashcards.")
                        continue
                        
                    elif 'next flashcard' in lower_input or 'show question' in lower_input:
                        # Mark for showing only question part
                        self.last_command_context = {
                            'command': user_input,
                            'command_type': 'show_question',
                            'timestamp': time.time()
                        }
                        print("\nShowcasing question. Answer is hidden.")
                        continue
                        
                    elif 'show answer' in lower_input or 'reveal answer' in lower_input:
                        # Mark for showing answer part
                        self.last_command_context = {
                            'command': user_input,
                            'command_type': 'show_answer',
                            'timestamp': time.time()
                        }
                        print("\nRevealing answer.")
                        continue
                        
                    # Focus region commands for ADHD support
                    elif 'focus on' in lower_input:
                        print("\nSelect region to focus on (format: x,y,width,height):")
                        region_input = await asyncio.to_thread(input, "")
                        
                        try:
                            coords = [int(x.strip()) for x in region_input.split(',')]
                            if len(coords) == 4:
                                self.focus_region = tuple(coords)
                                print(f"Focus region set to {self.focus_region}")
                                # Set context for ADHD focus support
                                self.last_command_context = {
                                    'command': user_input,
                                    'command_type': 'focus_region',
                                    'timestamp': time.time()
                                }
                            else:
                                print("Invalid format. Expected x,y,width,height")
                        except ValueError:
                            print("Invalid coordinates. Expected integers.")
                        continue
                        
                    elif 'clear focus' in lower_input or 'reset focus' in lower_input:
                        self.focus_region = None
                        print("\nFocus region cleared.")
                        continue
                    
                    # Standard reading commands
                    elif any(cmd in lower_input for cmd in ['read this', 'read the text', 'what does it say']):
                        # Mark this as a text reading command
                        self.last_command_context = {
                            'command': user_input,
                            'command_type': 'read_text',
                            'content_type': 'medical' if self.is_medical_context else 'standard',
                            'timestamp': time.time()
                        }
                        
                        # For text reading commands, don't need to send text separately
                        # The next screen capture will include the enhanced text processing
                        print("\nFocusing on text in next screen capture...")
                        continue
                    
                    # For region selection commands
                    elif 'focus on region' in lower_input:
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
                    elif lower_input in ['help', '?', 'commands']:
                        print("\nAvailable commands:")
                        print("- 'read this' or 'what does it say': Read text from the screen")
                        print("- 'focus on region': Select a specific region of the screen")
                        print("- 'reset region': Reset to full screen")
                        print("- 'medical mode': Optimize for medical content")
                        print("- 'flashcard mode': Enable flashcard study support")
                        print("- 'show question'/'show answer': Control flashcard visibility")
                        print("- 'focus on': Enhance a specific area (for ADHD support)")
                        print("- 'clear focus': Remove focus enhancement")
                        print("- 'q': Quit the application")
                        continue
                    
                    # Send text to Gemini
                    if self.connect:
                        try:
                            start_time = time.time()
                            await self.connect.send(input=user_input, end_of_turn=True)
                            
                            # Process response
                            async for chunk in self.connect.receive():
                                if chunk.text:
                                    print("\nGemini: ", chunk.text)
                                    # Update command history
                                    self.command_history.add_command(user_input, chunk.text[:100], {"type": "text"})
                                elif hasattr(chunk, 'tool_call'):
                                    await self.handle_tool_call(chunk.tool_call)
                                elif hasattr(chunk, 'server_content') and hasattr(chunk.server_content, 'interrupted') and chunk.server_content.interrupted:
                                    logger.info("Generation interrupted by user")
                            
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
    # Function Calling Handling
    # ---------------------------
    async def handle_tool_call(self, tool_call):
        """Handle tool calls from the model."""
        try:
            logger.info(f"Received tool call: {tool_call}")
            
            function_calls = tool_call.function_calls if hasattr(tool_call, 'function_calls') else []
            function_responses = []
            
            for call in function_calls:
                try:
                    # Extract function information
                    function_name = call.name if hasattr(call, 'name') else ""
                    function_args = json.loads(call.args) if hasattr(call, 'args') else {}
                    function_id = call.id if hasattr(call, 'id') else ""
                    
                    logger.info(f"Processing function call: {function_name} with args: {function_args}")
                    
                    # Process based on function name
                    if function_name == "get_weather":
                        result = {"temperature": 72, "condition": "sunny", "location": function_args.get("location", "unknown")}
                        print(f"\nCalled function: get_weather for {function_args.get('location', 'unknown location')}")
                        print(f"Result: Temperature 72°F, Sunny")
                        
                    elif function_name == "set_alarm":
                        time_value = function_args.get('time', '00:00')
                        result = {"success": True, "time": time_value}
                        print(f"\nCalled function: set_alarm for {time_value}")
                        print(f"Result: Alarm set for {time_value}")
                        
                    # Medical education specific functions
                    elif function_name == "create_flashcard":
                        front = function_args.get('front', '')
                        back = function_args.get('back', '')
                        category = function_args.get('category', 'General')
                        
                        # Create flashcard
                        self.current_flashcard = {
                            "front": front,
                            "back": back,
                            "category": category
                        }
                        
                        result = {
                            "success": True,
                            "flashcard_id": int(time.time()),
                            "message": "Flashcard created successfully"
                        }
                        
                        print(f"\nCreated flashcard - Category: {category}")
                        print(f"Front: {front}")
                        print(f"Back: {back}")
                    
                    else:
                        # Default for unimplemented functions
                        result = {"error": "Function not implemented"}
                    
                    # Create function response
                    function_responses.append({
                        "name": function_name,
                        "response": json.dumps(result),
                        "id": function_id
                    })
                    
                    logger.info(f"Function {function_name} returned: {result}")
                    
                except Exception as e:
                    logger.error(f"Error processing function call {call}: {e}")
                    # Still provide a response for this function
                    function_responses.append({
                        "name": call.name if hasattr(call, 'name') else "unknown",
                        "response": json.dumps({"error": str(e)}),
                        "id": call.id if hasattr(call, 'id') else "unknown"
                    })
            
            # Send function responses back to model
            if function_responses and self.connect:
                await self.connect.send_tool_response(function_responses=function_responses)
                logger.info(f"Sent {len(function_responses)} function responses back to model")
                
        except Exception as e:
            logger.error(f"Error handling tool call: {e}")
            traceback.print_exc()

    # ---------------------------
    # Audio Handling
    # ---------------------------
    async def send_realtime(self):
        """Stream audio to Gemini leveraging API's built-in VAD."""
        try:
            consecutive_errors = 0
            
            while self.running:
                if self.audio_stream:
                    # Read audio chunk
                    try:
                        data = self.audio_stream.read(CHUNK_SIZE, exception_on_overflow=False)
                        consecutive_errors = 0  # Reset error counter on successful read
                        
                        # Send audio data directly to Gemini - it will handle VAD internally
                        if self.connect:
                            # LiveConnect supports audio input
                            audio_part = types.Part.audio(data, mime_type="audio/raw;encoding=signed-integer;bits_per_sample=16;sample_rate=16000")
                            
                            # Send audio to Gemini without waiting for completion
                            await self.connect.send(input=audio_part, end_of_turn=False)
                    
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
        Handle audio responses from Gemini with improved interruption support.
        """
        self.audio_in_queue = asyncio.Queue()  # For compatibility
        
        while self.running:
            try:
                if self.connect:
                    # Process each response chunk
                    async for chunk in self.connect.receive():
                        # Check for interruptions
                        if hasattr(chunk, 'server_content') and hasattr(chunk.server_content, 'interrupted'):
                            if chunk.server_content.interrupted:
                                logger.info("Generation interrupted by user")
                                # Clear audio buffers to stop playback immediately
                                while not self.audio_in_queue.empty():
                                    try:
                                        self.audio_in_queue.get_nowait()
                                    except:
                                        pass
                                
                                # If any function calls were canceled, log them
                                if hasattr(chunk.server_content, 'cancelled_tool_calls'):
                                    cancelled_ids = chunk.server_content.cancelled_tool_calls
                                    logger.info(f"Function calls cancelled due to interruption: {cancelled_ids}")
                                
                                continue
                        
                        # Process audio data
                        if chunk.data:
                            await self.audio_in_queue.put(chunk.data)
                        
                        # Process text response
                        if chunk.text:
                            print("\nGemini: ", chunk.text)
                            # Update command history
                            self.command_history.add_command("Response", chunk.text[:100], {"type": "response"})
                        
                        # Process tool calls
                        if hasattr(chunk, 'tool_call'):
                            await self.handle_tool_call(chunk.tool_call)
            except Exception as e:
                logger.error(f"Error in receive_audio: {e}")
                await asyncio.sleep(1.0)
    
    async def play_audio(self):
        """
        Audio playback for responses from Gemini.
        """
        self.audio_in_queue = asyncio.Queue()  # Initialize if not already done
        
        try:
            # Open output stream
            stream = await asyncio.to_thread(
                pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            
            while self.running:
                try:
                    # Get audio data from queue
                    audio_data = await self.audio_in_queue.get()
                    
                    # Adjust playback rate for ADHD support if needed
                    if self.adhd_support_enabled:
                        # This is a simplified approach - real implementation would use a 
                        # proper audio processing library to adjust speed without pitch change
                        # For now, we'll just play at normal rate
                        pass
                    
                    # Play audio
                    await asyncio.to_thread(stream.write, audio_data)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in audio playback: {e}")
                    await asyncio.sleep(0.1)
            
            # Clean up
            stream.close()
            
        except Exception as e:
            logger.error(f"Error setting up audio playback: {e}")
            traceback.print_exc()

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
    parser.add_argument(
        "--adhd-support",
        type=str,
        default="enabled",
        help="Enable ADHD support features",
        choices=["enabled", "disabled"],
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
    
    # Set ADHD support based on argument
    ADHD_SUPPORT_ENABLED = (args.adhd_support == "enabled")
    
    logger.info(f"Starting Gemini Voice Reading Assistant in {args.mode} mode")
    logger.info(f"ADHD support features: {args.adhd_support}")
    
    main = AudioLoop(video_mode=args.mode)
    
    try:
        asyncio.run(main.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
