import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import List, Dict, Any

class PhoneticAnalyzer:
    def __init__(self, model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing PhoneticAnalyzer on {self.device}...")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name, use_safetensors=True).to(self.device)
        self.model.eval()

    def analyze_alignment(self, audio_path: str, transcript: str) -> List[Dict[str, Any]]:
        """
        Performs forced alignment (simplified CTC alignment) for the given transcript.
        Returns a list of characters with their relative confidence and timing metadata.
        """
        # Load audio (Model expects 16kHz)
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Preprocess
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=16000).input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Get predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)[0]
        # Get transcription string from IDs
        transcription = self.processor.decode(predicted_ids)
        
        # Simplified frame-level confidence for Tajweed
        # Logits shape: [1, frames, vocab_size]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        max_probs, _ = torch.max(probs, dim=-1)
        
        # We return the basic transcription + confidence for now
        # Production alignment usually requires CTC segmentation (e.g. from ctc-segmentation lib)
        # but for Tajweed (Ghunnah/Madd), we mostly need the frame-level probabilities of the targeted phonemes.
        
        return {
            "transcription": transcription,
            "confidence_avg": float(torch.mean(max_probs)),
            "frames": max_probs.cpu().numpy().tolist() # Frame-by-frame confidence
        }

    def get_phonetic_features(self, audio_path: str):
        """
        Extracts acoustic features relevant for Tajweed (Intensity, Pitch, Duration).
        """
        y, sr = librosa.load(audio_path)
        
        # Energy/Intensity (for Qalqalah)
        rms = librosa.feature.rms(y=y)[0]
        
        # Pitch (for Intonation/Melody - important for beautiful recitation)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        return {
            "intensity": rms.tolist(),
            "duration": librosa.get_duration(y=y, sr=sr),
            "sample_rate": sr
        }
