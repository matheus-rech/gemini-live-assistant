# Gemini Voice Reading Assistant

An interactive voice assistant powered by Google's Gemini AI that can see and respond to what you show through your camera or screen.

![Gemini Voice Assistant Demo](docs/demo-screenshot.png)

## Features

- **Real-time Audio Interaction**: Seamless voice communication with Gemini AI
- **Multiple Input Methods**: 
  - Camera capture for analyzing physical documents and objects
  - Screen capture for sharing digital content
  - Text input for typed queries
  - Voice recording for spoken questions
- **Text Recognition and Reading**: 
  - Read text from camera or screen feeds
  - Summarize documents 
  - Find specific information in visible text
- **Adaptive Performance**: Automatically monitors and adjusts settings for optimal performance
- **Voice Responses**: Natural-sounding responses with multiple voice options
- **Web Interface**: Clean, responsive UI for intuitive interaction
- **WebSocket Server**: Robust server with connection management and error recovery
- **Docker Support**: Simple deployment with Docker and Docker Compose

## System Requirements

- Python 3.10 or higher
- Google Gemini API key (obtain from [AI Studio](https://aistudio.google.com/apikey))
- Working microphone and speakers
- Webcam (for camera mode)
- 4GB RAM minimum (8GB+ recommended)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## Quick Start

### Docker Installation (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gemini-voice-assistant.git
   cd gemini-voice-assistant
   ```

2. Create a `.env` file with your API key:
   ```bash
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

3. Run with Docker Compose:
   ```bash
   docker-compose up
   ```

4. Open your browser and navigate to [http://localhost:3000](http://localhost:3000)

### Manual Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gemini-voice-assistant.git
   cd gemini-voice-assistant
   ```

2. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Set your Gemini API key:
   ```bash
   export GEMINI_API_KEY=your_api_key_here
   ```

4. Run the application:
   ```bash
   ./run.sh
   ```

## Usage

### Web Interface

The web interface provides the following options:

- **Start/Stop Camera**: Toggle webcam capture
- **Start/Stop Screen Share**: Toggle screen capturing
- **Capture Image**: Capture a still image from the video feed
- **Voice Recording**: Click and hold to record audio
- **Text Input**: Type your questions or commands
- **Voice Selection**: Choose from different voice options for AI responses

### Command Line

```bash
# Run with camera mode
./run.sh --mode camera

# Run with screen capture mode
./run.sh --mode screen

# Run without visual input (audio only)
./run.sh --mode none

# Run with debug logging
./run.sh --log-level DEBUG
```

### Common Commands

When interacting with the assistant, you can use these commands:

- **"Read this"** or **"What does it say?"**: Ask Gemini to read visible text
- **"Summarize this page"**: Get a concise summary of visible text
- **"Find [X] on this page"**: Locate specific information
- **"Describe what you see"**: Get a description of visible content

## Project Structure

```
gemini-voice-assistant/
├── main.py              # Core application logic
├── server.py            # WebSocket server
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── requirements.txt     # Python dependencies
├── setup.sh             # Setup script
├── run.sh               # Run script
├── logs/                # Log files
└── web-client/          # Web interface
    ├── src/             # React source code
    ├── public/          # Static assets
    └── Dockerfile       # Web client Docker configuration
```

## Advanced Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Google Gemini API key
- `GEMINI_MODEL`: Model to use (default: "models/gemini-2.0-flash")
- `HOST`: Host to bind the server to (default: "localhost")
- `PORT`: Port to run the server on (default: 8000)

### Performance Tuning

You can adjust these parameters in `main.py` to optimize performance:

- `CAPTURE_INTERVAL_MIN`: Minimum time between captures (seconds)
- `CAPTURE_INTERVAL_MAX`: Maximum time between captures (seconds)
- `DEFAULT_CAPTURE_INTERVAL`: Initial capture interval

### Custom Voice Options

Available voices:
- Puck (default)
- Charon
- Kore
- Fenrir
- Aoede

## Troubleshooting

### ALSA Library Issues

If you encounter errors related to the ALSA library (common on Linux):

1. **Check audio device availability**:
   ```bash
   arecord -l
   ```

2. **Configure ALSA**:
   Create or update `~/.asoundrc`:
   ```
   pcm.!default {
       type hw
       card 0
   }

   ctl.!default {
       type hw
       card 0
   }
   ```

3. **Check PulseAudio**:
   Start PulseAudio:
   ```bash
   pulseaudio --start
   ```

4. **Permissions**:
   Add your user to the audio group:
   ```bash
   sudo usermod -aG audio $USER
   ```
   Log out and back in for changes to take effect.

### WebSocket Connection Issues

1. Check that the WebSocket server is running:
   ```bash
   curl http://localhost:8000/health
   ```

2. Ensure no firewall is blocking port 8000

3. Check server logs:
   ```bash
   tail -f logs/gemini_assistant_*.log
   ```

## Security Considerations

- The API key grants access to the Gemini API and should be kept secure
- The application only processes media locally and doesn't store recordings
- Screen sharing should be used cautiously to avoid exposing sensitive information

## Development

### Prerequisites for Development

- Node.js 16+ for web client development
- Python 3.10+ for backend development

### Setting Up Development Environment

1. Set up the backend:
   ```bash
   ./setup.sh --venv
   ```

2. Set up the web client:
   ```bash
   cd web-client
   npm install
   ```

3. Run in development mode:
   ```bash
   # Terminal 1 (Backend)
   ./run.sh --server
   
   # Terminal 2 (Web Client)
   cd web-client
   npm start
   ```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project uses the [Google Gemini API](https://ai.google.dev/gemini-api)
- Inspired by the Multimodal Live API examples from the [Gemini cookbook](https://github.com/google-gemini/cookbook)

---

For questions or feedback, please open an issue on the GitHub repository.
