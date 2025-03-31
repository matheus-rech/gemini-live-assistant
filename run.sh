#!/bin/bash
# run.sh - Run the Gemini Voice Reading Assistant

# Function to display usage information
show_help() {
    echo "Gemini Voice Reading Assistant"
    echo "Usage: ./run.sh [options]"
    echo ""
    echo "Options:"
    echo "  --mode <mode>       Set capture mode: camera, screen, or none (default: camera)"
    echo "  --log-level <level> Set log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)"
    echo "  --docker            Run using Docker (requires Docker installed)"
    echo "  --server            Run only the WebSocket server"
    echo "  --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh --mode screen      # Run with screen capture mode"
    echo "  ./run.sh --log-level DEBUG  # Run with debug logging"
    echo "  ./run.sh --docker           # Run using Docker containers"
}

# Set default values
MODE="camera"
LOG_LEVEL="INFO"
RUN_DOCKER=false
RUN_SERVER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --docker)
            RUN_DOCKER=true
            shift
            ;;
        --server)
            RUN_SERVER=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Create logs directory if it doesn't exist
mkdir -p logs

# Set the log file with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/gemini_assistant_${TIMESTAMP}.log"

# If running in Docker mode
if [ "$RUN_DOCKER" = true ]; then
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "Error: Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        echo "Error: Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check for API key
    if [ -z "$GEMINI_API_KEY" ]; then
        if [ -f .env ]; then
            echo "Loading API key from .env file..."
            export $(grep -v '^#' .env | xargs)
        else
            echo "Error: GEMINI_API_KEY environment variable is not set and .env file not found."
            exit 1
        fi
    fi
    
    echo "Starting Gemini Voice Assistant with Docker..."
    echo "Mode: $MODE"
    
    # Export environment variables for docker-compose
    export GEMINI_MODE=$MODE
    export LOG_LEVEL=$LOG_LEVEL
    
    # Run docker-compose
    docker-compose up --build
    
    exit 0
fi

# If not running in Docker mode, run the Python application directly

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    if [ "$(uname)" == "Darwin" ] || [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
fi

# Check if API key is set
if [ -z "$GEMINI_API_KEY" ]; then
    if [ -f .env ]; then
        echo "Loading API key from .env file..."
        export $(grep -v '^#' .env | xargs)
    else
        echo "Error: GEMINI_API_KEY environment variable is not set and .env file not found."
        exit 1
    fi
fi

# Validate mode
if [[ "$MODE" != "camera" && "$MODE" != "screen" && "$MODE" != "none" ]]; then
    echo "Error: Invalid mode. Please use 'camera', 'screen', or 'none'."
    exit 1
fi

# Check if model is set
if [ -z "$GEMINI_MODEL" ]; then
    echo "Setting default model: models/gemini-2.0-flash-exp"
    export GEMINI_MODEL="models/gemini-2.0-flash-exp"
fi

# Run application
if [ "$RUN_SERVER" = true ]; then
    echo "Starting WebSocket server..."
    echo "Log file: $LOG_FILE"
    python server.py --log-level $LOG_LEVEL --log-file $LOG_FILE 2>&1 | tee -a $LOG_FILE
else
    echo "Starting Gemini Voice Assistant..."
    echo "Mode: $MODE"
    echo "Model: $GEMINI_MODEL"
    echo "Log Level: $LOG_LEVEL"
    echo "Log file: $LOG_FILE"
    python main.py --mode $MODE --log-level $LOG_LEVEL --log-file $LOG_FILE 2>&1 | tee -a $LOG_FILE
fi
# Please install the portaudio library before installing pyaudio
# On Ubuntu/Debian:
# sudo apt-get install portaudio19-dev
# On macOS:
# brew install portaudio
