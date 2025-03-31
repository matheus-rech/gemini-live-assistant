import React, { useState, useEffect, useRef, useCallback } from 'react';
import './GeminiAssistant.css';

const RECONNECT_INTERVAL = 2000; // 2 seconds
const MAX_RECONNECT_ATTEMPTS = 5;

const GeminiAssistant = () => {
  // State for UI elements
  const [connected, setConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isListening, setIsListening] = useState(false);
  const [selectedMode, setSelectedMode] = useState('camera');
  const [selectedVoice, setSelectedVoice] = useState('Puck');
  const [isProcessing, setIsProcessing] = useState(false);
  const [streamVideo, setStreamVideo] = useState(false);
  const [error, setError] = useState(null);
  const [reconnectMessage, setReconnectMessage] = useState(null); // Pa638
  
  // Refs
  const wsRef = useRef(null);
  const videoRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const screenCaptureRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const reconnectAttemptsRef = useRef(0);
  const messagesEndRef = useRef(null);
  
  const voices = ['Puck', 'Charon', 'Kore', 'Fenrir', 'Aoede'];
  
  // Scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Connect to WebSocket
  const connectWebSocket = useCallback(() => {
    // Get websocket URL from env or use default
    const wsUrl = process.env.REACT_APP_WS_URL || 'ws://localhost:8000';
    
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      console.log('WebSocket already connected');
      return;
    }
    
    try {
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        setError(null);
        setReconnectMessage(null); // Pa638
        reconnectAttemptsRef.current = 0;
        
        // Send initial mode preference to server
        ws.send(JSON.stringify({
          type: 'mode',
          content: selectedMode
        }));
        
        // Send voice preference to server
        ws.send(JSON.stringify({
          type: 'voice',
          content: selectedVoice
        }));
      };
      
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Received message:', data);
        
        switch (data.type) {
          case 'text':
            setMessages(prevMessages => [
              ...prevMessages, 
              { role: 'assistant', content: data.content, type: 'text' }
            ]);
            setIsProcessing(false);
            break;
            
          case 'audio':
            // Play audio response
            const audio = new Audio(`data:audio/wav;base64,${data.content}`);
            audio.play();
            break;
            
          case 'status':
            console.log('Status update:', data.content);
            // For important status updates, we can add them to the messages
            if (data.content.includes('Connected')) {
              setMessages(prevMessages => [
                ...prevMessages, 
                { role: 'system', content: data.content, type: 'status' }
              ]);
            }
            break;
            
          case 'error':
            console.error('Error from server:', data.content);
            setError(data.content);
            setIsProcessing(false);
            break;
          
          case 'tool_call':
            console.log('Tool call from assistant:', data.content);
            
            // Process each function call
            if (data.content && data.content.function_calls) {
                const functionResponses = data.content.function_calls.map(call => {
                    // Here we're providing a simple implementation - in a real app, 
                    // you would implement actual function handling
                    let result = { error: "Function not implemented" };
                    let args = {};
                    
                    try {
                        args = JSON.parse(call.args);
                    } catch (e) {
                        console.error('Error parsing function args:', e);
                    }
                    
                    // Example implementation for a few functions
                    if (call.name === "get_weather") {
                        result = { temperature: 72, condition: "sunny", location: args.location || "unknown" };
                    } else if (call.name === "search_web") {
                        result = { results: ["Result 1", "Result 2", "Result 3"] };
                    } else if (call.name === "set_alarm") {
                        result = { success: true, time: args.time || "00:00" };
                    }
                    
                    // Add message to UI to show function was called
                    setMessages(prevMessages => [
                        ...prevMessages, 
                        { 
                            role: 'system', 
                            content: `Function called: ${call.name}${args.location ? ` for ${args.location}` : ''}${args.time ? ` at ${args.time}` : ''}`, 
                            type: 'function' 
                        }
                    ]);
                    
                    return {
                        id: call.id,
                        name: call.name,
                        response: JSON.stringify(result)
                    };
                });
                
                // Send responses back to server
                if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({
                        type: 'tool_response',
                        content: functionResponses
                    }));
                }
            }
            break;
            
          default:
            console.log('Unknown message type:', data.type);
        }
      };
      
      ws.onclose = (event) => {
        console.log('WebSocket disconnected, code:', event.code);
        setConnected(false);
        
        // Attempt reconnect if not a normal closure
        if (event.code !== 1000 && event.code !== 1001) {
          reconnectAttemptsRef.current += 1;
          
          if (reconnectAttemptsRef.current <= MAX_RECONNECT_ATTEMPTS) {
            console.log(`Reconnecting in ${RECONNECT_INTERVAL}ms... (Attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`);
            setReconnectMessage(`Reconnecting... (Attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`); // Pa638
            setTimeout(connectWebSocket, RECONNECT_INTERVAL);
          } else {
            setError('Failed to reconnect after multiple attempts. Please refresh the page.');
          }
        }
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error. Please check your network.');
      };
      
      wsRef.current = ws;
      
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
      setError('Failed to connect. Please try again later.');
    }
  }, [selectedMode, selectedVoice]);
  
  // Connect on component mount
  useEffect(() => {
    connectWebSocket();
    
    // Clean up on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
      
      if (screenCaptureRef.current) {
        screenCaptureRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [connectWebSocket]);
  
  // Handle text input submission
  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!inputText.trim() || !connected) return;
    
    // Add message to UI
    setMessages(prevMessages => [
      ...prevMessages, 
      { role: 'user', content: inputText, type: 'text' }
    ]);
    
    // Send to server
    wsRef.current.send(JSON.stringify({
      type: 'text',
      content: inputText
    }));
    
    setInputText('');
    setIsProcessing(true);
  };
  
  // Start/stop webcam
  const toggleCamera = async () => {
    try {
      if (streamVideo) {
        // Stop camera
        if (mediaStreamRef.current) {
          mediaStreamRef.current.getTracks().forEach(track => track.stop());
          mediaStreamRef.current = null;
        }
        
        setStreamVideo(false);
        videoRef.current.srcObject = null;
        
      } else {
        // Start camera
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { width: 640, height: 480 }, 
          audio: false 
        });
        
        mediaStreamRef.current = stream;
        videoRef.current.srcObject = stream;
        setStreamVideo(true);
        
        // Change mode on server if needed
        if (selectedMode !== 'camera') {
          setSelectedMode('camera');
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'mode',
              content: 'camera'
            }));
          }
        }
      }
    } catch (error) {
      console.error('Error accessing camera:', error);
      setError('Failed to access camera. Please check permissions.');
    }
  };
  
  // Start/stop screen capture
  const toggleScreenCapture = async () => {
    try {
      if (streamVideo) {
        // Stop screen capture
        if (screenCaptureRef.current) {
          screenCaptureRef.current.getTracks().forEach(track => track.stop());
          screenCaptureRef.current = null;
        }
        
        setStreamVideo(false);
        videoRef.current.srcObject = null;
        
      } else {
        // Start screen capture
        const stream = await navigator.mediaDevices.getDisplayMedia({ 
          video: { cursor: 'always' },
          audio: false
        });
        
        // Handle user cancelling the screen selection
        stream.getVideoTracks()[0].onended = () => {
          setStreamVideo(false);
          screenCaptureRef.current = null;
        };
        
        screenCaptureRef.current = stream;
        videoRef.current.srcObject = stream;
        setStreamVideo(true);
        
        // Change mode on server if needed
        if (selectedMode !== 'screen') {
          setSelectedMode('screen');
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
              type: 'mode',
              content: 'screen'
            }));
          }
        }
      }
    } catch (error) {
      console.error('Error accessing screen capture:', error);
      setError('Failed to capture screen. Please check permissions.');
    }
  };
  
  // Start/stop voice recording
  const toggleVoiceRecording = async () => {
    if (isListening) {
      // Stop recording
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      
    } else {
      try {
        // Get audio stream
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Create media recorder
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        chunksRef.current = [];
        
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) {
            chunksRef.current.push(e.data);
          }
        };
        
        mediaRecorder.onstop = async () => {
          setIsListening(false);
          
          if (chunksRef.current.length > 0) {
            const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
            chunksRef.current = [];
            
            // Convert to base64
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            
            reader.onloadend = () => {
              const base64Audio = reader.result.split(',')[1];
              
              // Add message to UI
              setMessages(prevMessages => [
                ...prevMessages, 
                { role: 'user', content: '[Voice message]', type: 'audio' }
              ]);
              
              // Send to server
              if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({
                  type: 'audio',
                  content: base64Audio
                }));
                
                setIsProcessing(true);
              }
            };
          }
          
          // Stop all tracks
          stream.getTracks().forEach(track => track.stop());
        };
        
        // Start recording
        mediaRecorder.start();
        setIsListening(true);
        
        // Auto-stop after 10 seconds to prevent very long recordings
        setTimeout(() => {
          if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
          }
        }, 10000);
        
      } catch (error) {
        console.error('Error accessing microphone:', error);
        setError('Failed to access microphone. Please check permissions.');
      }
    }
  };
  
  // Capture image from camera/screen
  const captureImage = () => {
    if (!videoRef.current || !videoRef.current.srcObject) {
      setError('No camera or screen feed available.');
      return;
    }
    
    try {
      // Create a canvas element to capture the image
      const canvas = document.createElement('canvas');
      canvas.width = videoRef.current.videoWidth;
      canvas.height = videoRef.current.videoHeight;
      
      // Draw the current video frame on the canvas
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
      
      // Convert to base64
      const base64Image = canvas.toDataURL('image/jpeg');
      
      // Add message to UI
      setMessages(prevMessages => [
        ...prevMessages, 
        { role: 'user', content: '[Image captured]', type: 'image' }
      ]);
      
      // Send to server
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'image',
          content: base64Image.split(',')[1]
        }));
        
        setIsProcessing(true);
      }
    } catch (error) {
      console.error('Error capturing image:', error);
      setError('Failed to capture image.');
    }
  };
  
  // Handle voice selection
  const handleVoiceChange = (e) => {
    const newVoice = e.target.value;
    setSelectedVoice(newVoice);
    
    // Send to server
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'voice',
        content: newVoice
      }));
    }
  };
  
  // Render loading indicator during processing
  const renderLoadingIndicator = () => {
    return isProcessing && (
      <div className="loading-indicator">
        <div className="loading-spinner"></div>
        <p>Processing...</p>
      </div>
    );
  };

  // Custom message rendering based on message type
  const renderMessage = (msg, index) => {
    // Special rendering for function messages
    if (msg.type === 'function') {
      return (
        <div key={index} className="message system function">
          <div className="message-content">
            <span className="function-icon">⚙️</span> {msg.content}
          </div>
        </div>
      );
    }
    
    // Default message rendering
    return (
      <div key={index} className={`message ${msg.role}`}>
        <div className="message-content">
          {msg.content}
        </div>
      </div>
    );
  };

  return (
    <div className="gemini-assistant">
      <header className="assistant-header">
        <h1>Gemini Voice Reading Assistant</h1>
        <div className="connection-status">
          <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}></span>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </header>
      
      {error && (
        <div className="error-banner">
          <span>{error}</span>
          <button onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      
      {reconnectMessage && ( // Pa638
        <div className="reconnect-banner">
          <span>{reconnectMessage}</span>
        </div>
      )}
      
      <div className="main-container">
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <h2>Welcome to Gemini Voice Reading Assistant</h2>
              <p>I can help you read and interpret text from your camera or screen.</p>
              <p>Try saying "Read this" or "What does this say?" when showing me text.</p>
            </div>
          ) : (
            <div className="messages-list">
              {messages.map((msg, index) => renderMessage(msg, index))}
              <div ref={messagesEndRef} />
            </div>
          )}
          
          {renderLoadingIndicator()}
        </div>
        
        <div className="video-container">
          <video 
            ref={videoRef} 
            autoPlay 
            muted 
            className={streamVideo ? 'active' : 'hidden'}
          ></video>
          
          {!streamVideo && (
            <div className="video-placeholder">
              <span className="placeholder-text">No video stream active</span>
            </div>
          )}
          
          <div className="video-controls">
            <button 
              onClick={toggleCamera} 
              className={streamVideo && selectedMode === 'camera' ? 'active' : ''}
              disabled={!connected || (streamVideo && selectedMode !== 'camera')}
            >
              <i className="icon camera-icon"></i>
              {streamVideo && selectedMode === 'camera' ? 'Stop Camera' : 'Start Camera'}
            </button>
            
            <button 
              onClick={toggleScreenCapture} 
              className={streamVideo && selectedMode === 'screen' ? 'active' : ''}
              disabled={!connected || (streamVideo && selectedMode !== 'screen')}
            >
              <i className="icon screen-icon"></i>
              {streamVideo && selectedMode === 'screen' ? 'Stop Screen' : 'Share Screen'}
            </button>
            
            <button 
              onClick={captureImage} 
              disabled={!connected || !streamVideo}
            >
              <i className="icon capture-icon"></i>
              Capture Image
            </button>
          </div>
        </div>
      </div>
      
      <div className="input-container">
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Type your message..."
            disabled={!connected || isProcessing}
          />
          <button 
            type="submit" 
            disabled={!connected || !inputText.trim() || isProcessing}
          >
            <i className="icon send-icon"></i>
          </button>
        </form>
        
        <button 
          className={`voice-button ${isListening ? 'listening' : ''}`}
          onClick={toggleVoiceRecording}
          disabled={!connected || isProcessing}
        >
          <i className={`icon microphone-icon ${isListening ? 'active' : ''}`}></i>
          {isListening ? 'Stop Recording' : 'Start Recording'}
        </button>
      </div>
      
      <div className="settings-panel">
        <div className="voice-selection">
          <label htmlFor="voice-select">Voice:</label>
          <select 
            id="voice-select" 
            value={selectedVoice}
            onChange={handleVoiceChange}
            disabled={!connected}
          >
            {voices.map(voice => (
              <option key={voice} value={voice}>{voice}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
};

export default GeminiAssistant;
