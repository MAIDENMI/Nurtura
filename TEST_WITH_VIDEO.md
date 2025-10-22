# Testing with Baby Videos

## How to Test the Breathing Monitor with Actual Infant Videos

I've created a special version that works with video files so you can test it on real baby videos!

## 🎥 Quick Start

### Step 1: Get a Baby Video

You have several options:

#### Option A: Use Your Own Video
- Any video of a baby where the torso/chest is visible
- Works with: `.mp4`, `.avi`, `.mov`, `.mkv`, etc.
- Best quality: 480p or higher

#### Option B: Download a Test Video from YouTube

**Safe search terms:**
- "sleeping baby breathing"
- "newborn baby sleeping"
- "infant sleeping peacefully"
- "baby monitor footage"

**Using yt-dlp (recommended):**
```bash
# Install yt-dlp
brew install yt-dlp

# Download a video (replace URL with actual video)
yt-dlp -f 'best[height<=480]' -o baby_test.mp4 'https://youtube.com/watch?v=...'
```

#### Option C: Stock Video Sites
- Pexels Videos (free)
- Pixabay Videos (free)
- Search: "baby sleeping" or "infant breathing"

### Step 2: Run the Analysis

```bash
# Activate virtual environment
source venv/bin/activate

# Run with your video file
python breathing_monitor_video.py your_video.mp4
```

## 🎮 Controls

While video is playing:
- **SPACE** - Pause/Resume
- **'r'** - Restart video
- **'s'** - Take screenshot
- **'q'** - Quit

## 📊 What You'll See

### On-Screen Display:
1. **Breathing Rate** - Real-time breaths per minute
2. **Status** - NORMAL, LOW, HIGH, or DETECTING
3. **Pose Detection** - % of frames where baby detected
4. **Motion Value** - Raw breathing motion intensity
5. **Average BPM** - Running average (appears after 10+ frames)
6. **Skeleton Overlay** - Green lines showing detected pose
7. **Blue Torso Box** - Region being analyzed for breathing
8. **Progress Bar** - Shows video progress

### Color Coding:
- 🟢 **Green (NORMAL)**: 20-60 breaths/min
- 🔴 **Red (LOW)**: < 20 breaths/min
- 🔴 **Red (HIGH)**: > 60 breaths/min
- ⚪ **White (DETECTING)**: Initializing...

### Final Statistics:
When video ends, you'll see:
```
✓ Reached end of video
  Total frames processed: 450
  Pose detected in: 423 frames (94.0%)
  Average breathing rate: 34.2 BPM
  Min: 28.5, Max: 41.3
```

## ✅ What to Look For

### Good Results:
- ✓ Pose detection > 80%
- ✓ Breathing rate 20-60 BPM (for infants)
- ✓ Blue torso box visible and stable
- ✓ Green skeleton tracking the baby
- ✓ Consistent readings (not jumping wildly)

### Common Issues:

**"Pose not detected"**
- Baby not fully visible in frame
- Baby is too small in video
- Video quality too low
- Baby is covered (blanket over chest)

**Breathing rate = 0 or erratic**
- Wait 10-15 seconds for initialization
- Baby might be very still
- Lighting might be poor
- Try adjusting threshold in code

**Rate seems too high/low**
- Check if baby is actually moving (not breathing motion)
- Verify video FPS is correct
- Motion from camera movement can affect readings

## 🎯 Best Video Characteristics

For accurate breathing detection:

### Camera Setup:
- ✅ Fixed camera (not handheld)
- ✅ Baby's chest clearly visible
- ✅ Front or slight angle view
- ✅ 1-2 meters distance
- ❌ Too close or too far

### Lighting:
- ✅ Good, even lighting
- ✅ Soft natural light
- ❌ Harsh shadows
- ❌ Very dark/night vision

### Baby Position:
- ✅ Lying on back (supine)
- ✅ Chest exposed or thin clothing
- ✅ Relatively still (sleeping)
- ❌ Moving around
- ❌ Covered by blankets

### Video Quality:
- ✅ 480p or higher
- ✅ 24-30 FPS
- ✅ Minimal compression
- ❌ Very blurry
- ❌ Low resolution

## 📝 Example Test Session

```bash
$ python breathing_monitor_video.py baby_sleeping.mp4

============================================================
Infant Breathing Monitor - Video Analysis
============================================================

📹 Loading video: baby_sleeping.mp4
✓ Video loaded successfully
  Resolution: 640x480
  FPS: 30.0
  Frames: 900
  Duration: 30.0 seconds

Controls:
  SPACE - Pause/Resume
  'q' - Quit
  'r' - Restart video
  's' - Screenshot

Processing video...

# Video plays with real-time analysis...
# You see breathing rate updating, pose overlay, etc.

✓ Reached end of video
  Total frames processed: 900
  Pose detected in: 856 frames (95.1%)
  Average breathing rate: 36.4 BPM
  Min: 32.1, Max: 42.8

✓ Analysis complete
```

## 🔬 Interpreting Results

### Infant Breathing Rates (for reference):

| Age | Normal Range (breaths/min) |
|-----|---------------------------|
| Newborn (0-3 months) | 30-60 |
| Infant (3-12 months) | 24-40 |
| Toddler (1-3 years) | 20-30 |

If your results fall in these ranges, the detection is working well!

### Example Good Results:
```
Average breathing rate: 34.2 BPM  ✓
Pose detection: 94.0%            ✓
Min: 28.5, Max: 41.3             ✓
Status: NORMAL                    ✓
```

### Example Issues:
```
Average breathing rate: 85.3 BPM  ❌ Too high (motion detected)
Pose detection: 12.0%            ❌ Baby not visible
Min: 0, Max: 150                 ❌ Erratic readings
```

## 💡 Tips

1. **Pause and examine**
   - Press SPACE to pause
   - Check if torso box is correct
   - Verify skeleton aligns with baby

2. **Try different videos**
   - Test with multiple videos
   - Compare results
   - See what works best

3. **Save interesting frames**
   - Press 's' to screenshot
   - Capture good detections
   - Document results

4. **Adjust for your video**
   - Edit the code if needed
   - Tune `breathing_threshold`
   - Adjust `window_size`

## 🐛 Troubleshooting

**Video won't open:**
```bash
# Check if video file exists
ls -lh your_video.mp4

# Try a different format
ffmpeg -i input.mov -c:v libx264 output.mp4
```

**Pose not detected at all:**
- Try a different video
- Ensure baby is clearly visible
- Check video isn't corrupted

**Can't find good test videos:**
- Use your phone to record a test
- Point at yourself and breathe slowly
- Adult breathing works too for testing!

## 🎓 What This Proves

Successfully analyzing a baby video demonstrates:

✅ **The AI can detect infant poses**
- MediaPipe works on small body sizes
- Detects babies, not just adults

✅ **Breathing detection works in real scenarios**
- Not just lab conditions
- Handles real-world video

✅ **Non-contact monitoring is feasible**
- No sensors needed
- Just a camera

✅ **The algorithm can measure subtle movements**
- Breathing is tiny motion
- Optical flow is sensitive enough

## ⚠️ Remember

This is still **educational/research only**:
- NOT for medical use
- NOT for baby safety monitoring
- NOT a replacement for proper equipment
- Just demonstrates the technology works!

## 📧 Next Steps

After testing with video:
1. Try different baby videos
2. Compare results with video descriptions
3. Test edge cases (bad lighting, covered baby, etc.)
4. Document what works and what doesn't

This is great for:
- Validating your project
- Presentations/demos
- Understanding limitations
- Research and learning

Have fun testing! 🎉

