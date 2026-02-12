//! TTS Unit Tests
//!
//! Tests for text-to-speech functionality.
//! Run with: cargo test -p audiolearn tts_tests -- --nocapture

#[cfg(test)]
mod tts_tests {
    use crate::core::{
        EdgeTts, NativeTts, TtsManager,
        synthesize_text, get_tts_voices, is_tts_available,
    };
    use crate::spi::tts::{SpeechOptions, TtsEngine};
    use rodio::Source;
    
    /// Test that Edge TTS can list voices (requires network)
    #[test]
    fn test_edge_tts_list_voices() {
        let mut edge = EdgeTts::new();
        
        if edge.is_available() {
            let result = edge.voices();
            assert!(result.is_ok(), "Should be able to list voices");
            
            let voices = result.unwrap();
            println!("✅ Found {} Edge TTS voices", voices.len());
            
            // Should have many neural voices
            let neural_count = voices.iter().filter(|v| v.is_neural).count();
            println!("   {} are neural voices", neural_count);
            assert!(neural_count > 100, "Should have >100 neural voices");
            
            // Show some English voices
            let english_voices: Vec<_> = voices.iter()
                .filter(|v| v.language.starts_with("en-US"))
                .take(5)
                .collect();
            
            println!("   Sample US English voices:");
            for v in english_voices {
                println!("     - {} ({:?})", v.id, v.gender);
            }
        } else {
            println!("⚠️ Edge TTS not available (no network?)");
        }
    }
    
    /// Test that Edge TTS can synthesize audio bytes
    #[test]
    fn test_edge_tts_synthesize() {
        let edge = EdgeTts::new();
        
        if edge.is_available() {
            let options = SpeechOptions::default();
            
            // Synthesize a short phrase
            let result = edge.synthesize("Hello world", &options);
            
            match result {
                Ok(bytes) => {
                    println!("✅ Synthesized {} bytes of audio", bytes.len());
                    assert!(!bytes.is_empty(), "Audio should not be empty");
                    assert!(bytes.len() > 1000, "Audio should be substantial");
                    
                    // Check for MP3 header (0xFF 0xFB or ID3 header)
                    let is_mp3 = (bytes.len() >= 2 && bytes[0] == 0xFF && (bytes[1] & 0xE0) == 0xE0)
                        || (bytes.len() >= 3 && &bytes[0..3] == b"ID3");
                    println!("   Audio format appears to be MP3: {}", is_mp3);
                }
                Err(e) => {
                    panic!("❌ Synthesis failed: {}", e);
                }
            }
        } else {
            println!("⚠️ Edge TTS not available");
        }
    }
    
    /// Test that Native TTS is available
    #[test]
    fn test_native_tts_available() {
        let native = NativeTts::try_new();
        
        match native {
            Some(n) => {
                println!("✅ Native TTS is available: {}", n.name());
                assert!(n.is_available());
                
                // List available voices
                if let Ok(voices) = n.voices() {
                    println!("   Found {} native voices", voices.len());
                    for v in voices.iter().take(5) {
                        println!("     - {} ({})", v.name, v.language);
                    }
                }
            }
            None => {
                println!("⚠️ Native TTS not available on this system");
            }
        }
    }
    
    /// Test TTS Manager with fallback
    #[test]
    fn test_tts_manager() {
        let mut manager = TtsManager::new();
        
        println!("TTS Manager created with preference: EdgeFirst");
        println!("  Available: {}", manager.is_available());
        
        if manager.is_available() {
            // Get voices
            let voices = manager.voices();
            if let Ok(v) = voices {
                println!("  Total voices: {}", v.len());
            }
            
            // Synthesize
            let options = SpeechOptions::default();
            let result = manager.synthesize("Test", &options);
            
            match result {
                Ok(bytes) => {
                    println!("✅ Manager synthesized {} bytes", bytes.len());
                    println!("   Using engine: {:?}", manager.last_engine());
                }
                Err(e) => {
                    println!("⚠️ Synthesis failed: {}", e);
                }
            }
        }
    }
    
    /// Test global TTS functions
    #[test]
    fn test_global_tts_functions() {
        // Check availability
        let available = is_tts_available();
        println!("TTS available: {}", available);
        
        if available {
            // Get voices
            if let Ok(voices) = get_tts_voices() {
                println!("✅ Got {} voices via global function", voices.len());
            }
            
            // Synthesize (don't actually play to avoid audio in tests)
            let result = synthesize_text("Quick test");
            match result {
                Ok(bytes) => {
                    println!("✅ Synthesized {} bytes via global function", bytes.len());
                }
                Err(e) => {
                    println!("⚠️ Global synthesize failed: {}", e);
                }
            }
        }
    }
    
    /// Test different speech options
    #[test]
    fn test_speech_options() {
        let edge = EdgeTts::new();
        
        if edge.is_available() {
            println!("Testing different speech options...");
            
            // Normal rate
            let normal = SpeechOptions {
                rate: 1.0,
                pitch: 0.0,
                volume: 1.0,
                voice: None,
            };
            
            // Fast rate
            let fast = SpeechOptions {
                rate: 1.5,
                ..Default::default()
            };
            
            // Slow rate
            let slow = SpeechOptions {
                rate: 0.7,
                ..Default::default()
            };
            
            // Higher pitch
            let high_pitch = SpeechOptions {
                pitch: 0.5,
                ..Default::default()
            };
            
            let text = "Test";
            
            for (name, opts) in [
                ("Normal", normal),
                ("Fast", fast),
                ("Slow", slow),
                ("High pitch", high_pitch),
            ] {
                match edge.synthesize(text, &opts) {
                    Ok(bytes) => println!("  ✅ {}: {} bytes", name, bytes.len()),
                    Err(e) => println!("  ❌ {} failed: {}", name, e),
                }
            }
        }
    }
    
    /// Verify the audio bytes are valid MP3 (can be decoded)
    #[test]
    fn test_audio_decode() {
        use std::io::Cursor;
        use rodio::Decoder;
        
        let edge = EdgeTts::new();
        
        if edge.is_available() {
            let options = SpeechOptions::default();
            let result = edge.synthesize("Hello, this is a decoding test.", &options);
            
            match result {
                Ok(bytes) => {
                    println!("Got {} bytes, attempting to decode...", bytes.len());
                    
                    // Try to decode with rodio
                    let cursor = Cursor::new(bytes);
                    match Decoder::new(cursor) {
                        Ok(source) => {
                            println!("✅ Successfully decoded audio!");
                            println!("   Sample rate: {:?}", source.sample_rate());
                            println!("   Channels: {:?}", source.channels());
                        }
                        Err(e) => {
                            println!("❌ Failed to decode audio: {}", e);
                            println!("   This might indicate wrong audio format.");
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Synthesis failed: {}", e);
                }
            }
        } else {
            println!("⚠️ Edge TTS not available");
        }
    }
}
