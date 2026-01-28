#!/bin/bash
# Video optimization script for demo.webm
#
# The original demo.webm is 72MB, too large for GitHub (25MB limit)
# This script provides several optimization options

echo "========================================"
echo "Video Optimization Options"
echo "========================================"
echo ""

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠ ffmpeg not found. Please install it:"
    echo "  Ubuntu/Debian: sudo apt-get install ffmpeg"
    echo "  macOS: brew install ffmpeg"
    echo ""
    exit 1
fi

# Option 1: Compress to smaller WebM (recommended)
echo "Option 1: Compress to smaller WebM (target: <10MB)"
echo "  Command:"
echo "  ffmpeg -i demo.webm -c:v libvpx-vp9 -crf 40 -b:v 0 -b:a 96k -c:a libopus demo_compressed.webm"
echo ""

# Option 2: Convert to optimized MP4
echo "Option 2: Convert to MP4 (better compatibility)"
echo "  Command:"
echo "  ffmpeg -i demo.webm -c:v libx264 -crf 28 -preset slow -c:a aac -b:a 96k demo.mp4"
echo ""

# Option 3: Create short GIF preview (first 10 seconds)
echo "Option 3: Create GIF preview (for README embedding)"
echo "  Command:"
echo "  ffmpeg -i demo.webm -t 10 -vf \"fps=10,scale=800:-1:flags=lanczos\" -loop 0 demo_preview.gif"
echo ""

# Option 4: Extract key frames as images
echo "Option 4: Extract key screenshots"
echo "  Command:"
echo "  ffmpeg -i demo.webm -vf \"select='eq(n,0)+eq(n,150)+eq(n,300)'\" -vsync 0 frame_%d.png"
echo ""

echo "========================================"
echo "Recommended workflow:"
echo "  1. Create compressed WebM for GitHub Release"
echo "  2. Create 10-second GIF for README preview"
echo "  3. Upload full video to YouTube for best experience"
echo "========================================"
echo ""
echo "Run this script with an option number (1-4):"
echo "  ./optimize_video.sh 1"
