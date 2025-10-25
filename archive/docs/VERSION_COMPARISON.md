# Version Comparison

## Quick Reference: Which Version Should I Use?

| Version | Best For | Run Command |
|---------|----------|-------------|
| **Enhanced** ⭐ | Best accuracy, infant-specific | `./run.sh enhanced` |
| **Graphical** 📊 | Demonstrations, visual feedback | `./run.sh graph` |
| **Video** 🎥 | Testing with recordings | `./run.sh video file.mp4` |
| **Advanced** ⚙️ | Logging, configuration | `./run.sh advanced` |
| **Basic** 📚 | Learning, simplicity | `./run.sh` |

---

## Detailed Comparison

### Feature Matrix

| Feature | Basic | Enhanced | Graphical | Advanced | Video |
|---------|-------|----------|-----------|----------|-------|
| **Accuracy** | Good | ⭐ Best | Good | Good | Good |
| **Real-time graphs** | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Confidence score** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Signal processing** | Basic | ⭐ Advanced | Basic | Basic | Basic |
| **Research-based** | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Config file** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Data logging** | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Works on video** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Raspberry Pi** | ✅ | ✅ | ⚠️ | ✅ | ✅ |
| **Easy to understand** | ⭐ | ⚠️ | ⚠️ | ⚠️ | ✅ |

Legend: ✅ Yes | ❌ No | ⚠️ With caveats | ⭐ Best

---

## Version Details

### 1. Basic Version 📚

**File:** `breathing_monitor.py`

**Purpose:** Educational and simple demonstrations

**Key Features:**
- Single file, ~200 lines
- Simple optical flow
- Basic peak detection
- Easy to understand and modify

**When to use:**
- Learning how breathing detection works
- Quick demos
- Starting point for customization
- Teaching computer vision

**Pros:**
- Simplest code
- Fast to run
- No extra dependencies
- Good for learning

**Cons:**
- Lower accuracy
- No advanced features
- Basic filtering

**Run:**
```bash
./run.sh
```

---

### 2. Enhanced Version ⭐ RECOMMENDED

**File:** `breathing_monitor_enhanced.py`

**Purpose:** Best accuracy with research-validated techniques

**Key Features:**
- Based on AIRFlowNet research (MICCAI 2023)
- Optimized optical flow parameters
- Enhanced chest ROI selection
- Bandpass filtering (0.2-1.2 Hz)
- Confidence scoring
- Advanced peak detection

**When to use:**
- Need best accuracy
- Testing on actual infants
- Comparing with research
- Production/deployment
- When accuracy matters most

**Pros:**
- ⭐ Best accuracy
- Research-validated
- Confidence scores
- Infant-optimized
- Better noise rejection

**Cons:**
- Slightly more complex
- Requires scipy
- Bit slower than basic

**Run:**
```bash
./run.sh enhanced
```

**What's Different:**
```python
# Enhanced optical flow
poly_n=7,           # vs 5 in basic
poly_sigma=1.5,     # vs 1.2 in basic

# Signal processing
bandpass_filter()   # Not in basic
gaussian_smooth()   # Not in basic
confidence_score()  # Not in basic
```

---

### 3. Graphical Version 📊

**File:** `breathing_monitor_graphical.py`

**Purpose:** Visual demonstrations with real-time graphs

**Key Features:**
- Two live graphs (rate + waveform)
- Like medical heart monitors
- Color-coded zones
- Dual window display
- Screenshot both windows

**When to use:**
- Presentations and demos
- Visualizing breathing patterns
- Teaching how breathing looks
- Recording visual data
- Impressive display

**Pros:**
- Beautiful visualization
- Real-time graphs
- Great for demos
- Educational display
- Medical monitor aesthetic

**Cons:**
- ⚠️ Retina display issues (being fixed)
- Higher CPU usage
- Two windows
- More complex code

**Run:**
```bash
./run.sh graph
```

**What You See:**
- Window 1: Camera + detection
- Window 2: Two live graphs

---

### 4. Advanced Version ⚙️

**File:** `breathing_monitor_advanced.py`

**Purpose:** Production deployment with logging and configuration

**Key Features:**
- Configuration file support
- CSV data logging
- Alert system (high/low rates)
- FPS display
- Screenshot capability
- Better error handling

**When to use:**
- Research data collection
- Long-term monitoring
- Need to log data
- Configurable parameters
- Multiple deployments

**Pros:**
- Easy configuration
- Data logging
- Alert system
- Production-ready
- No code changes needed

**Cons:**
- Requires config.py
- More files
- Not single-file

**Run:**
```bash
./run.sh advanced
```

**Configuration:**
Edit `config.py` to adjust all parameters without touching code.

---

### 5. Video Analysis 🎥

**File:** `breathing_monitor_video.py`

**Purpose:** Test with pre-recorded videos

**Key Features:**
- Works with video files
- Playback controls (pause/resume)
- Statistics at end
- No camera needed
- Perfect for validation

**When to use:**
- Testing on infant videos
- No live camera available
- Validating accuracy
- Comparing with ground truth
- Batch processing

**Pros:**
- No camera needed
- Repeatable tests
- Playback control
- Good statistics
- Easy validation

**Cons:**
- Not real-time
- Need video files
- No live monitoring

**Run:**
```bash
./run.sh video baby_video.mp4
```

**Controls:**
- SPACE: Pause/Resume
- 'r': Restart
- 's': Screenshot
- 'q': Quit

---

## Performance Comparison

### Accuracy (Estimated)

| Version | Accuracy | Confidence | Stability |
|---------|----------|------------|-----------|
| Basic | Good (±5 BPM) | N/A | Moderate |
| **Enhanced** | ⭐ Best (±2-4 BPM) | ✅ Shown | High |
| Graphical | Good (±5 BPM) | N/A | Moderate |
| Advanced | Good (±5 BPM) | N/A | Moderate |
| Video | Good (±5 BPM) | N/A | High |

### Resource Usage

| Version | CPU | Memory | Complexity |
|---------|-----|--------|------------|
| Basic | Low | Low | Simple |
| Enhanced | Medium | Medium | Moderate |
| Graphical | High | Medium | Complex |
| Advanced | Low | Low | Moderate |
| Video | Medium | Medium | Simple |

---

## Recommendation Flow

```
START
  │
  ├─ Need best accuracy? ────────────────► Enhanced ⭐
  │
  ├─ Want visual graphs? ────────────────► Graphical 📊
  │
  ├─ Testing with videos? ───────────────► Video 🎥
  │
  ├─ Need logging/config? ───────────────► Advanced ⚙️
  │
  └─ Learning/teaching? ─────────────────► Basic 📚
```

---

## Combination Strategies

### For Research:
1. Use **Video** to test on dataset
2. Use **Enhanced** for best accuracy
3. Use **Advanced** to log data

### For Demonstrations:
1. Use **Graphical** for visual impact
2. Use **Enhanced** for accuracy claims
3. Use **Video** to show validation

### For Development:
1. Start with **Basic** to learn
2. Move to **Enhanced** for production
3. Use **Video** for testing

---

## Migration Path

### From Basic → Enhanced
```python
# Change just the import
from breathing_monitor import BreathingMonitor
# to
from breathing_monitor_enhanced import EnhancedBreathingMonitor

# Everything else stays the same!
```

### From Any → Video
```bash
# Just change the command
./run.sh video your_recording.mp4
```

---

## Summary Table

| If you want... | Use this version | Command |
|----------------|------------------|---------|
| Best accuracy | Enhanced | `./run.sh enhanced` |
| Visual graphs | Graphical | `./run.sh graph` |
| Easy learning | Basic | `./run.sh` |
| Data logging | Advanced | `./run.sh advanced` |
| Test videos | Video | `./run.sh video file.mp4` |
| Quick demo | Graphical | `./run.sh graph` |
| Research validation | Enhanced | `./run.sh enhanced` |
| Configuration | Advanced | `./run.sh advanced` |

---

## Try Them All!

Each version has its strengths. Feel free to try them all and pick what works best for your use case!

```bash
# Test them out:
./run.sh            # Basic
./run.sh enhanced   # Enhanced (recommended)
./run.sh graph      # Graphical (impressive!)
./run.sh advanced   # Advanced (configurable)
./run.sh video baby.mp4  # Video analysis
```

## Questions?

See the documentation:
- `ENHANCED_VERSION.md` - Enhanced version details
- `GRAPHICAL_VERSION.md` - Graphical version guide
- `TEST_WITH_VIDEO.md` - Video analysis guide
- `README.md` - Complete documentation

