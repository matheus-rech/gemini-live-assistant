#!/bin/bash
# setup.sh - Setup script for Gemini Voice Reading Assistant

echo "Gemini Voice Reading Assistant - Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Detected Python version: $python_version"
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || [ "$python_major" -eq 3 -a "$python_minor" -lt 10 ]; then
    echo "Error: Python 3.10 or higher is required."
    exit 1
fi

# Check for operating system and install dependencies
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    echo "Detected macOS system"
    
    # Check for Homebrew
    if ! command -v brew &> /dev/null; then
        echo "Homebrew is required but not installed. Please install Homebrew first."
        echo "Visit https://brew.sh for installation instructions."
        exit 1
    fi
    
    # Install dependencies with Homebrew
    echo "Installing dependencies with Homebrew..."
    brew install portaudio opencv ffmpeg
    
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux
    echo "Detected Linux system"
    
    # Check for apt (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "Installing dependencies with apt..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev libopencv-dev python3-opencv ffmpeg \
                            libsm6 libxext6 libgl1-mesa-glx curl
    # Check for dnf (Fedora/RHEL)
    elif command -v dnf &> /dev/null; then
        echo "Installing dependencies with dnf..."
        sudo dnf install -y portaudio-devel opencv-devel ffmpeg-devel curl
    # Check for pacman (Arch Linux)
    elif command -v pacman &> /dev/null; then
        echo "Installing dependencies with pacman..."
        sudo pacman -S portaudio opencv ffmpeg curl
    else
        echo "Warning: Unable to detect package manager. Please install required dependencies manually:"
        echo "  - PortAudio development libraries"
        echo "  - OpenCV development libraries"
        echo "  - FFmpeg"
        echo "  - cURL"
    fi
    
else
    # Windows or other OS
    echo "Warning: Operating system not recognized. You may need to install dependencies manually."
    echo "Required dependencies:"
    echo "  - PortAudio development libraries"
    echo "  - OpenCV development libraries"
    echo "  - FFmpeg"
    echo "  - cURL"
fi

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    if [ "$(uname)" == "Darwin" ] || [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
        source venv/bin/activate
    else
        source venv/Scripts/activate
    fi
    
    echo "Virtual environment created and activated."
fi

# Create logs directory
mkdir -p logs
echo "Created logs directory."

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Check for API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "Warning: GEMINI_API_KEY environment variable is not set."
    echo "You will need to set this before running the application."
    echo "Example: export GEMINI_API_KEY=your_api_key_here"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        echo "Creating .env file for environment variables..."
        echo "# Add your API key here" > .env
        echo "GEMINI_API_KEY=" >> .env
        echo "GEMINI_MODEL=models/gemini-2.0-flash-exp" >> .env
        echo ".env file created. Please add your API key there."
    fi
fi

# Set up web client if it exists
if [ -d "web-client" ]; then
    echo "Setting up web client..."
    
    # Check for Node.js
    if command -v node &> /dev/null; then
        echo "Node.js detected, installing web client dependencies..."
        cd web-client
        if command -v npm &> /dev/null; then
            npm install
            cd ..
        else
            echo "Warning: npm not found. Please install npm to build the web client."
            cd ..
        fi
    else
        echo "Warning: Node.js not found. Please install Node.js to build the web client."
    fi
fi

echo ""
echo "Setup complete! You can now run the application with:"
echo "./run.sh"
echo ""
echo "For more options, run: ./run.sh --help"

# Provide Docker instructions if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo ""
    echo "Docker and Docker Compose detected. To run with Docker:"
    echo "1. Make sure your .env file contains your GEMINI_API_KEY"
    echo "2. Run: docker-compose up"
fi
