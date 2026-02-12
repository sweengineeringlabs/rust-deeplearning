//! TTS Demo - Test text-to-speech functionality
//!
//! Run with: cargo run --example tts_demo

use audiolearn_app::core::{
    speak_text, synthesize_text, get_tts_voices, is_tts_available,
    TtsManager, TtsPreference, EdgeTtsSync,
};
use audiolearn_app::spi::tts::SpeechOptions;

fn main() {
    println!("🎤 AudioLearn TTS Demo");
    println!("======================\n");

    // Check TTS availability
    println!("Checking TTS availability...");
    if is_tts_available() {
        println!("✅ TTS is available!\n");
    } else {
        println!("❌ TTS is not available. Check your network connection.\n");
        return;
    }

    // List some available voices
    println!("Available voices (first 10 English neural voices):");
    match get_tts_voices() {
        Ok(voices) => {
            let english_neural: Vec<_> = voices
                .iter()
                .filter(|v| v.language.starts_with("en-") && v.is_neural)
                .take(10)
                .collect();

            for voice in &english_neural {
                println!("  - {} ({}, {:?})", voice.name, voice.id, voice.gender);
            }
            println!("\nTotal voices: {} (Neural: {})",
                voices.len(),
                voices.iter().filter(|v| v.is_neural).count()
            );
        }
        Err(e) => println!("Failed to get voices: {}", e),
    }

    // Synthesize some text
    println!("\n--- Synthesizing Audio ---\n");

    let test_text = "Welcome to AudioLearn, your audio-first learning platform. \
        Today we'll explore Rust ownership and borrowing.";

    println!("Text: \"{}\"\n", test_text);

    // Option 1: Quick speak
    println!("Speaking with default settings...");
    match speak_text(test_text) {
        Ok(_) => println!("✅ Speech completed!"),
        Err(e) => println!("❌ Failed to speak: {}", e),
    }

    // Option 2: Synthesize to bytes (for saving or custom playback)
    println!("\nSynthesizing to audio bytes...");
    match synthesize_text("Hello, this is a shorter test.") {
        Ok(bytes) => println!("✅ Synthesized {} bytes of audio data", bytes.len()),
        Err(e) => println!("❌ Failed to synthesize: {}", e),
    }

    // Option 3: Custom options with different voice
    println!("\nSpeaking with custom options (faster, higher pitch)...");
    let custom_options = SpeechOptions {
        voice: None, // Use default
        rate: 1.5,   // 50% faster
        pitch: 0.2,  // Slightly higher pitch
        volume: 1.0,
    };

    let mut manager = TtsManager::with_preference(TtsPreference::EdgeFirst);
    match manager.speak("This is spoken faster with a higher pitch.", &custom_options) {
        Ok(_) => println!("✅ Custom speech completed!"),
        Err(e) => println!("❌ Failed to speak with custom options: {}", e),
    }

    println!("\n🎉 TTS Demo Complete!");
}
