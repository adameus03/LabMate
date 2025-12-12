#!/usr/bin/env python3
"""
RFID Real-Time Waterfall Server with WebSockets
Tails the log file and streams data via WebSocket for smooth visualization
"""

import time
import json
import asyncio
import websockets
import threading
import os
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import sys

class RFIDMonitor:
    def __init__(self, filepath, max_events=1000):
        self.filepath = filepath
        self.events = deque(maxlen=max_events)
        self.epcs = {}
        self.running = False
        self.file_pos = 0
        self.subscribers = set()
        
    def tail_file(self):
        """Continuously monitor file for new lines"""
        # Start from end of file
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r') as f:
                f.seek(0, 2)  # Go to end
                self.file_pos = f.tell()
        
        while self.running:
            try:
                with open(self.filepath, 'r') as f:
                    f.seek(self.file_pos)
                    lines = f.readlines()
                    self.file_pos = f.tell()
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            timestamp = int(parts[0])
                            epc = parts[1]
                            
                            # Assign column index to new EPCs
                            if epc not in self.epcs:
                                self.epcs[epc] = len(self.epcs)
                                # Notify subscribers of new EPC
                                asyncio.run(self.broadcast({
                                    'type': 'epc_update',
                                    'epcs': self.epcs
                                }))
                            
                            event = {
                                'type': 'event',
                                'timestamp': timestamp,
                                'epc': epc,
                                'col': self.epcs[epc]
                            }
                            
                            self.events.append(event)
                            # Broadcast to all subscribers
                            asyncio.run(self.broadcast(event))
                
                time.sleep(0.05)  # Poll every 50ms
                
            except FileNotFoundError:
                time.sleep(1)
            except Exception as e:
                print(f"Error reading file: {e}")
                time.sleep(1)
    
    async def broadcast(self, message):
        """Send message to all connected WebSocket clients"""
        if self.subscribers:
            await asyncio.gather(
                *[ws.send(json.dumps(message)) for ws in self.subscribers],
                return_exceptions=True
            )
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self.tail_file, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False

# Global monitor instance
monitor = None

async def websocket_handler(websocket, path):
    """Handle WebSocket connections"""
    monitor.subscribers.add(websocket)
    try:
        # Send initial EPC list
        await websocket.send(json.dumps({
            'type': 'epc_update',
            'epcs': monitor.epcs
        }))
        
        # Keep connection alive
        await websocket.wait_closed()
    finally:
        monitor.subscribers.remove(websocket)

class RFIDHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs
    
    def get_html(self):
        return '''<!DOCTYPE html>
<html>
<head>
    <title>RFID Piano Tiles Waterfall</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: #000;
            overflow: hidden;
            font-family: 'Courier New', monospace;
        }
        #canvas {
            display: block;
        }
        #info {
            position: absolute;
            top: 10px;
            left: 10px;
            color: #0f0;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border: 2px solid #0f0;
            font-size: 14px;
            border-radius: 5px;
            min-width: 200px;
            transition: opacity 0.3s, transform 0.3s;
        }
        #info.hidden {
            opacity: 0;
            pointer-events: none;
            transform: translateY(-10px);
        }
        #info input[type="number"] {
            width: 60px;
            background: #000;
            color: #0f0;
            border: 1px solid #0f0;
            padding: 2px;
            font-family: 'Courier New', monospace;
        }
        #info button {
            background: #0f0;
            color: #000;
            border: none;
            padding: 5px 15px;
            cursor: pointer;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            margin-top: 5px;
            width: 100%;
        }
        #info button:hover {
            background: #0c0;
        }
        #info button.paused {
            background: #f80;
        }
        #info button.active {
            background: #ff0;
            color: #000;
        }
        #legend {
            position: absolute;
            top: 10px;
            right: 10px;
            color: #0f0;
            background: rgba(0,0,0,0.8);
            padding: 15px;
            border: 2px solid #0f0;
            font-size: 11px;
            max-height: 90vh;
            overflow-y: auto;
            border-radius: 5px;
            min-width: 200px;
            transition: opacity 0.3s, transform 0.3s;
        }
        #legend.hidden {
            opacity: 0;
            pointer-events: none;
            transform: translateY(-10px);
        }
        .epc-item {
            display: flex;
            align-items: center;
            margin: 5px 0;
            padding: 3px;
        }
        .epc-color {
            width: 20px;
            height: 20px;
            margin-right: 8px;
            border: 1px solid #0f0;
            flex-shrink: 0;
        }
        .epc-text {
            font-size: 10px;
            word-break: break-all;
        }
        #status {
            position: absolute;
            bottom: 10px;
            left: 10px;
            color: #0f0;
            background: rgba(0,0,0,0.8);
            padding: 8px 12px;
            border: 2px solid #0f0;
            font-size: 12px;
            border-radius: 5px;
        }
        .connected { color: #0f0; }
        .disconnected { color: #f00; }
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <button id="togglePanels">HIDE PANELS (H)</button>
    <div id="info">
        <div><strong>RFID WATERFALL</strong></div>
        <div>Events: <span id="eventCount">0</span></div>
        <div>EPCs: <span id="epcCount">0</span></div>
        <div>Rate: <span id="rate">0</span> ev/s</div>
        <div style="margin-top: 10px; border-top: 1px solid #0f0; padding-top: 8px;">
            Time offset (min): <input type="number" id="timeOffset" value="-17" step="1">
        </div>
        <button id="pauseBtn">PAUSE</button>
        <button id="compactBtn">COMPACT MODE</button>
        <div style="font-size: 10px; margin-top: 5px; opacity: 0.7;">
            Spacebar: Pause | C: Compact
        </div>
    </div>
    <div id="legend">
        <strong>EPCs:</strong><br>
    </div>
    <div id="status">
        <span id="wsStatus" class="disconnected">Connecting...</span>
    </div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        
        let epcs = {};
        let events = [];
        let colors = {};
        let eventCount = 0;
        let lastEventCount = 0;
        let lastRateUpdate = Date.now();
        let ws = null;
        let timeOffsetMinutes = -17;
        let isPaused = false;
        let compactMode = false;
        
        const TILE_GAP = 3;
        const TILE_HEIGHT_NORMAL = 40;
        const TILE_HEIGHT_COMPACT = 5;
        const FALL_SPEED = 150; // pixels per second
        const TIME_MARGIN = 80;
        const TOP_MARGIN = 50;
        const PIXELS_PER_SECOND = 150; // Match FALL_SPEED for 1:1 time scale
        
        let gridLines = []; // Persistent grid lines with timestamps
        let nextGridLineY = TOP_MARGIN; // Next position to generate a grid line
        let gridLineCounter = 0; // Counter for generating timestamps
        let panelsVisible = true;
        
        function getTileHeight() {
            return compactMode ? TILE_HEIGHT_COMPACT : TILE_HEIGHT_NORMAL;
        }
        
        // Time offset handler
        document.addEventListener('DOMContentLoaded', () => {
            const offsetInput = document.getElementById('timeOffset');
            offsetInput.addEventListener('input', (e) => {
                timeOffsetMinutes = parseInt(e.target.value) || 0;
                // Reformat all existing events
                events.forEach(event => {
                    event.timeStr = formatTime(event.timestamp);
                });
            });
            
            // Pause/Play button
            const pauseBtn = document.getElementById('pauseBtn');
            pauseBtn.addEventListener('click', togglePause);
            
            // Compact mode button
            const compactBtn = document.getElementById('compactBtn');
            compactBtn.addEventListener('click', toggleCompact);
            
            // Toggle panels button
            const togglePanelsBtn = document.getElementById('togglePanels');
            togglePanelsBtn.addEventListener('click', togglePanels);
        });
        
        function togglePause() {
            isPaused = !isPaused;
            const pauseBtn = document.getElementById('pauseBtn');
            if (isPaused) {
                pauseBtn.textContent = 'PLAY';
                pauseBtn.classList.add('paused');
            } else {
                pauseBtn.textContent = 'PAUSE';
                pauseBtn.classList.remove('paused');
            }
        }
        
        function toggleCompact() {
            compactMode = !compactMode;
            const compactBtn = document.getElementById('compactBtn');
            if (compactMode) {
                compactBtn.textContent = 'NORMAL MODE';
                compactBtn.classList.add('active');
            } else {
                compactBtn.textContent = 'COMPACT MODE';
                compactBtn.classList.remove('active');
            }
        }
        
        function togglePanels() {
            console.log('togglePanels called, current state:', panelsVisible);
            panelsVisible = !panelsVisible;
            const info = document.getElementById('info');
            const legend = document.getElementById('legend');
            const toggleBtn = document.getElementById('togglePanels');
            
            console.log('Elements:', info, legend, toggleBtn);
            
            if (panelsVisible) {
                info.classList.remove('hidden');
                legend.classList.remove('hidden');
                toggleBtn.textContent = 'HIDE PANELS (H)';
            } else {
                info.classList.add('hidden');
                legend.classList.add('hidden');
                toggleBtn.textContent = 'SHOW PANELS (H)';
            }
            console.log('New state:', panelsVisible);
        }
        
        // Keyboard shortcut
        document.addEventListener('keydown', (e) => {
            console.log('Key pressed:', e.code, 'Target:', e.target.tagName);
            
            // Only block if typing in input field
            if (e.target.tagName === 'INPUT') {
                return;
            }
            
            if (e.code === 'Space') {
                e.preventDefault();
                togglePause();
            } else if (e.code === 'KeyC') {
                e.preventDefault();
                toggleCompact();
            } else if (e.key === 'h' || e.key === 'H' || e.code === 'KeyH') {
                e.preventDefault();
                console.log('H pressed, toggling panels');
                togglePanels();
            }
        });
        
        function hslToRgb(h, s, l) {
            const c = (1 - Math.abs(2 * l - 1)) * s;
            const x = c * (1 - Math.abs((h / 60) % 2 - 1));
            const m = l - c/2;
            let r, g, b;
            if (h < 60) [r,g,b] = [c,x,0];
            else if (h < 120) [r,g,b] = [x,c,0];
            else if (h < 180) [r,g,b] = [0,c,x];
            else if (h < 240) [r,g,b] = [0,x,c];
            else if (h < 300) [r,g,b] = [x,0,c];
            else [r,g,b] = [c,0,x];
            return `rgb(${Math.round((r+m)*255)},${Math.round((g+m)*255)},${Math.round((b+m)*255)})`;
        }
        
        function getColor(index) {
            const hue = (index * 137.5) % 360;
            return hslToRgb(hue, 0.8, 0.6);
        }
        
        function updateLegend() {
            const legend = document.getElementById('legend');
            const sorted = Object.entries(epcs).sort((a,b) => a[1] - b[1]);
            
            let html = '<strong>EPCs:</strong><br>';
            sorted.forEach(([epc, col]) => {
                html += `
                    <div class="epc-item">
                        <div class="epc-color" style="background-color: ${colors[epc]}"></div>
                        <div class="epc-text">${epc}</div>
                    </div>
                `;
            });
            legend.innerHTML = html;
        }
        
        function formatTime(timestamp) {
            // timestamp is in microseconds since epoch, convert to milliseconds
            const date = new Date(timestamp / 1000 + (timeOffsetMinutes * 60 * 1000));
            
            // Get local time components
            const h = String(date.getHours()).padStart(2, '0');
            const m = String(date.getMinutes()).padStart(2, '0');
            const s = String(date.getSeconds()).padStart(2, '0');
            const ms = String(date.getMilliseconds()).padStart(3, '0');
            
            return `${h}:${m}:${s}.${ms}`;
        }
        
        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8765');
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                document.getElementById('wsStatus').textContent = '[CONNECTED]';
                document.getElementById('wsStatus').className = 'connected';
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'epc_update') {
                    // Update EPCs
                    Object.entries(data.epcs).forEach(([epc, col]) => {
                        if (!(epc in epcs)) {
                            epcs[epc] = col;
                            colors[epc] = getColor(col);
                        }
                    });
                    updateLegend();
                    
                } else if (data.type === 'event') {
                    // Only add new events if not paused
                    if (!isPaused) {
                        events.push({
                            timestamp: data.timestamp,
                            epc: data.epc,
                            col: data.col,
                            y: TOP_MARGIN,
                            timeStr: formatTime(data.timestamp)
                        });
                        eventCount++;
                    }
                }
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                document.getElementById('wsStatus').textContent = '[DISCONNECTED]';
                document.getElementById('wsStatus').className = 'disconnected';
                // Try to reconnect after 2 seconds
                setTimeout(connectWebSocket, 2000);
            };
        }
        
        function draw() {
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const numCols = Object.keys(epcs).length || 1;
            const availableWidth = canvas.width - TIME_MARGIN;
            const tileWidth = (availableWidth / numCols) - TILE_GAP;
            
            // Use consistent frame time (cap at 100ms to prevent huge jumps when tab is inactive)
            const now = Date.now();
            const rawDt = 16 / 1000; // Target 60fps
            const dt = Math.min(rawDt, 0.1); // Cap at 100ms to prevent acceleration when lagging
            
            // Move next grid line position down
            if (!isPaused) {
                nextGridLineY += FALL_SPEED * dt;
            }
            
            // Generate new grid lines when needed (every PIXELS_PER_SECOND pixels)
            while (nextGridLineY >= TOP_MARGIN + PIXELS_PER_SECOND) {
                nextGridLineY -= PIXELS_PER_SECOND;
                
                // Calculate timestamp for this grid line with offset applied
                const currentTimeMs = Date.now();
                const adjustedTimeMs = currentTimeMs + (timeOffsetMinutes * 60 * 1000);
                const lineTimeMs = adjustedTimeMs - (gridLineCounter * 1000);
                gridLineCounter++;
                
                const gridDate = new Date(lineTimeMs);
                const h = String(gridDate.getHours()).padStart(2, '0');
                const m = String(gridDate.getMinutes()).padStart(2, '0');
                const s = String(gridDate.getSeconds()).padStart(2, '0');
                
                gridLines.push({
                    y: TOP_MARGIN,
                    label: `${h}:${m}:${s}`,
                    timestamp: lineTimeMs
                });
            }
            
            // Update and draw grid lines
            for (let i = gridLines.length - 1; i >= 0; i--) {
                const line = gridLines[i];
                
                // Move line down if not paused
                if (!isPaused) {
                    line.y += FALL_SPEED * dt;
                }
                
                // Remove if off screen
                if (line.y > canvas.height) {
                    gridLines.splice(i, 1);
                    continue;
                }
                
                // Draw line
                ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(TIME_MARGIN, line.y);
                ctx.lineTo(canvas.width, line.y);
                ctx.stroke();
                
                // Draw time label
                ctx.fillStyle = '#0f0';
                ctx.font = 'bold 11px Courier New';
                ctx.textAlign = 'right';
                ctx.fillText(line.label, TIME_MARGIN - 10, line.y + 4);
            }
            
            // Draw EPC labels at top
            ctx.textAlign = 'center';
            ctx.font = 'bold 10px Courier New';
            Object.entries(epcs).forEach(([epc, col]) => {
                const x = TIME_MARGIN + col * (tileWidth + TILE_GAP) + tileWidth / 2;
                ctx.fillStyle = colors[epc];
                const shortEpc = epc.length > 12 ? epc.substring(0, 6) + '...' + epc.substring(epc.length - 4) : epc;
                ctx.fillText(shortEpc, x, 20);
            });
            
            // Update and draw tiles
            ctx.textAlign = 'center';
            const tileHeight = getTileHeight();
            
            for (let i = events.length - 1; i >= 0; i--) {
                const event = events[i];
                
                // Only update position if not paused
                if (!isPaused) {
                    event.y += FALL_SPEED * dt;
                }
                
                const x = TIME_MARGIN + event.col * (tileWidth + TILE_GAP);
                const y = event.y;
                
                // Remove if off screen (only when not paused)
                if (!isPaused && y > canvas.height) {
                    events.splice(i, 1);
                    continue;
                }
                
                // Draw tile with glow
                ctx.shadowBlur = compactMode ? 5 : 15;
                ctx.shadowColor = colors[event.epc];
                ctx.fillStyle = colors[event.epc];
                ctx.fillRect(x, y, tileWidth, tileHeight);
                ctx.shadowBlur = 0;
                
                // Draw text only in normal mode
                if (!compactMode) {
                    // Draw EPC text on tile
                    ctx.fillStyle = '#000';
                    ctx.font = 'bold 9px Courier New';
                    const shortEpc = event.epc.length > 10 ? event.epc.substring(0, 10) + '...' : event.epc;
                    ctx.fillText(shortEpc, x + tileWidth/2, y + 15);
                    
                    // Draw timestamp on tile
                    ctx.font = '8px Courier New';
                    ctx.fillText(event.timeStr, x + tileWidth/2, y + 30);
                }
                
                ctx.textAlign = 'center';
            }
            
            // Update stats
            document.getElementById('eventCount').textContent = eventCount;
            document.getElementById('epcCount').textContent = Object.keys(epcs).length;
            
            if (now - lastRateUpdate > 1000) {
                const rate = eventCount - lastEventCount;
                document.getElementById('rate').textContent = rate;
                lastEventCount = eventCount;
                lastRateUpdate = now;
            }
            
            requestAnimationFrame(draw);
        }
        
        // Start
        connectWebSocket();
        draw();
        
        window.addEventListener('resize', () => {
            canvas.width = window.innerWidth;
            canvas.height = window.innerHeight;
        });
    </script>
</body>
</html>'''

async def start_websocket_server():
    """Start WebSocket server"""
    async with websockets.serve(websocket_handler, "localhost", 8765):
        await asyncio.Future()  # run forever

def main():
    if len(sys.argv) < 2:
        print("Usage: python rfid_server.py <path_to_all__epc_time.dat> [http_port]")
        print("Example: python rfid_server.py ../../../output/all__epc_time.dat 8000")
        sys.exit(1)
    
    filepath = sys.argv[1]
    http_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    global monitor
    monitor = RFIDMonitor(filepath)
    monitor.start()
    
    # Start WebSocket server in a thread
    def run_ws():
        asyncio.run(start_websocket_server())
    
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    
    # Start HTTP server
    server = HTTPServer(('localhost', http_port), RFIDHandler)
    print(f"╔════════════════════════════════════════════════════╗")
    print(f"║  RFID Piano Tiles Waterfall Server                ║")
    print(f"╠════════════════════════════════════════════════════╣")
    print(f"║  HTTP:      http://localhost:{http_port}/")
    print(f"║  WebSocket: ws://localhost:8765")
    print(f"║  File:      {filepath}")
    print(f"╚════════════════════════════════════════════════════╝")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nStopping server...")
        monitor.stop()
        server.shutdown()

if __name__ == '__main__':
    main()
