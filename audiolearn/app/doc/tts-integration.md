# Text-to-Speech (TTS) Integration

This document describes the TTS system integrated into AudioLearn, providing high-quality neural text-to-speech capabilities.

## Overview

AudioLearn uses a dual-engine TTS approach:

| Engine | Description | Quality | Network Required |
|--------|-------------|---------|------------------|
| **Microsoft Edge TTS** | Neural voices via Azure | Excellent | Yes |
| **Native TTS** | Windows SAPI voices | Good | No |

The system uses Edge TTS as primary (higher quality) with Native TTS as fallback.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    UI Components                         │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐    │
│  │ReadAloudBtn  │ │QuickTtsBtn   │ │TtsSettings   │    │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘    │
└─────────┼────────────────┼────────────────┼─────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────┐
│                  Global Functions                        │
│  speak_text() | synthesize_text() | stop_tts()          │
│  get_tts_voices() | is_tts_available()                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                    TtsManager                            │
│  • Engine selection with fallback                        │
│  • Voice management                                      │
│  • Audio playback via rodio                             │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────┐           ┌──────────────────┐
│    EdgeTtsSync   │           │    NativeTts     │
│  (msedge-tts)    │           │    (tts crate)   │
└────────┬─────────┘           └────────┬─────────┘
         │                              │
         ▼                              ▼
┌──────────────────┐           ┌──────────────────┐
│  Microsoft Azure │           │   Windows SAPI   │
│  Neural Voices   │           │   System Voices  │
└──────────────────┘           └──────────────────┘
```

## How Edge TTS Works

When you send text to Edge TTS:

```
Your Text (e.g., 5000 characters)
           ↓
┌────────────────────────────────────────┐
│      Microsoft Edge TTS Server          │
│                                         │
│   Sentence 1 → Audio chunk 1 ─────→    │ → Streamed
│   Sentence 2 → Audio chunk 2 ─────→    │ → Streamed
│   Sentence 3 → Audio chunk 3 ─────→    │ → Streamed
│   ...                                   │
│   Sentence N → Audio chunk N ─────→    │ → Streamed
└────────────────────────────────────────┘
           ↓
    All chunks combined into one MP3 file
           ↓
    Decoded and played via rodio
```

**Key points:**
- Text is processed sentence by sentence on Microsoft's servers
- Audio is streamed back in chunks (handled by `msedge-tts` crate)
- Our code receives the final combined audio as `Vec<u8>`
- We decode and play the MP3 via `rodio`

## TTS Limits

| Aspect | Value |
|--------|-------|
| Text per request | ~10,000 characters (reliable) |
| Maximum text | ~64KB |
| Speaking rate | ~150-180 words/minute |
| Rate adjustment | 0.5x to 2.0x |
| 1 minute of speech | ~150-180 words |
| 10 minutes of speech | ~1,500-1,800 words |

## Available Voices

Edge TTS provides **322+ neural voices** across many languages:

### English (US) Examples
| Voice ID | Name | Gender |
|----------|------|--------|
| en-US-AriaNeural | Aria | Female |
| en-US-GuyNeural | Guy | Male |
| en-US-JennyNeural | Jenny | Female |
| en-US-DavisNeural | Davis | Male |

### Other Languages
- en-GB (British): 4 voices
- en-AU (Australian): 2 voices
- de-DE (German): 6 voices
- fr-FR (French): 6 voices
- es-ES (Spanish): 4 voices
- ja-JP (Japanese): 4 voices
- zh-CN (Chinese): 4 voices
- And 300+ more...

## Usage

### Basic Usage

```rust
use audiolearn::core::{speak_text, stop_tts, is_tts_available};

// Check availability
if is_tts_available() {
    // Speak text (blocks until complete)
    speak_text("Hello, welcome to AudioLearn!")?;
    
    // Stop current speech
    stop_tts()?;
}
```

### With Custom Options

```rust
use audiolearn::core::speak_text_with_options;
use audiolearn::spi::tts::SpeechOptions;

let options = SpeechOptions {
    rate: 1.2,   // 20% faster
    pitch: 0.0,  // Normal pitch
    volume: 1.0, // Full volume
    voice: Some("en-US-GuyNeural".to_string()),
};

speak_text_with_options("Hello!", &options)?;
```

### Get Audio Bytes (for custom playback)

```rust
use audiolearn::core::synthesize_text;

let audio_bytes: Vec<u8> = synthesize_text("Hello!")?;
// audio_bytes contains MP3 data
// You can save it, stream it, or play it yourself
```

### List Available Voices

```rust
use audiolearn::core::get_tts_voices;

let voices = get_tts_voices()?;
for voice in voices.iter().filter(|v| v.language.starts_with("en-")) {
    println!("{}: {} ({:?})", voice.id, voice.name, voice.gender);
}
```

## UI Components

### ReadAloudButton
A styled button that speaks text when clicked:

```rust
ReadAloudButton {
    text: "Text to speak".to_string(),
    label: Some("Read Aloud".to_string()),
    size: Size::Md,
}
```

### QuickTtsButton
A small inline button (headphone icon) for any text:

```rust
QuickTtsButton {
    text: lesson.title.clone(),
    tooltip: Some("Read title aloud".to_string()),
}
```

### TtsSettingsPanel
Voice selection and speed controls:

```rust
TtsSettingsPanel {
    selected_voice_id: selected_voice,
    on_voice_change: move |voice_id| { /* ... */ },
    rate: 1.0,
    on_rate_change: move |rate| { /* ... */ },
}
```

## Preventing Overlapping Speech

Always call `stop_tts()` before starting new speech:

```rust
spawn(async move {
    let _ = tokio::task::spawn_blocking(move || {
        // Stop any existing speech first
        let _ = stop_tts();
        // Then start new speech
        speak_text(&text)
    }).await;
});
```

## Files

| File | Purpose |
|------|---------|
| `src/spi/tts.rs` | Traits and types (TtsEngine, Voice, SpeechOptions) |
| `src/core/edge_tts.rs` | Edge TTS implementation |
| `src/core/native_tts.rs` | Native TTS implementation |
| `src/core/tts_manager.rs` | Manager with fallback and global functions |
| `src/core/tts_tests.rs` | Unit tests |
| `src/facade/components.rs` | UI components (ReadAloudButton, etc.) |
| `examples/tts_demo.rs` | Demo and usage examples |

## Running Tests

```bash
# Run all TTS tests
cargo test -p audiolearn tts_tests -- --nocapture

# Run the demo
cargo run -p audiolearn --example tts_demo
```

## Troubleshooting

### No sound
1. Check network connection (Edge TTS requires internet)
2. Verify audio output device is working
3. Check `is_tts_available()` returns true

### Speech too fast/slow
Use `SpeechOptions` to adjust rate:
- `rate: 0.5` = Half speed
- `rate: 1.0` = Normal
- `rate: 2.0` = Double speed

### Voice not found
Voice IDs are case-sensitive. Use `get_tts_voices()` to list available voices and use exact IDs.

## Future Enhancements

- [ ] Streaming playback (start playing before synthesis completes)
- [ ] Chunked processing for very long texts
- [ ] Progress tracking for TTS playback
- [ ] Voice caching/preloading
- [ ] Offline voice support via native TTS
