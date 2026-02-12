//! Audio test utilities

use std::io::Cursor;
use rodio::Source;

/// Generate a simple sine wave tone as WAV bytes
pub fn generate_test_tone(frequency: f32, duration_secs: f32, sample_rate: u32) -> Vec<u8> {
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut samples: Vec<i16> = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (t * frequency * 2.0 * std::f32::consts::PI).sin();
        // Fade in/out to avoid clicks
        let envelope = if i < 1000 {
            i as f32 / 1000.0
        } else if i > num_samples - 1000 {
            (num_samples - i) as f32 / 1000.0
        } else {
            1.0
        };
        samples.push((sample * envelope * 16000.0) as i16);
    }
    
    // Create WAV header
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    
    let mut wav = Vec::new();
    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    
    wav
}

/// Create a speaking-like tone pattern (simulating speech rhythm)
pub fn generate_speech_pattern(duration_secs: f32) -> Vec<u8> {
    let sample_rate = 44100u32;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let mut samples: Vec<i16> = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        
        // Mix multiple frequencies for a more natural sound
        let f1 = 150.0; // Base frequency
        let f2 = 300.0;
        let f3 = 450.0;
        
        // Modulate amplitude to simulate speech patterns
        let pattern = ((t * 3.0).sin() * 0.3 + 0.7).max(0.0);
        let wobble = (t * 50.0).sin() * 0.1;
        
        let sample = (t * f1 * 2.0 * std::f32::consts::PI).sin() * 0.5
            + (t * f2 * 2.0 * std::f32::consts::PI).sin() * 0.3
            + (t * f3 * 2.0 * std::f32::consts::PI).sin() * 0.2
            + wobble;
        
        // Add some variation
        let variation = ((t * 0.5).sin() * 0.2 + 0.8) * pattern;
        
        samples.push((sample * variation * 8000.0) as i16);
    }
    
    // Create WAV header
    let data_size = (samples.len() * 2) as u32;
    let file_size = 36 + data_size;
    
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    
    wav
}
