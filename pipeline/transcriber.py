#!/usr/bin/env python3
"""
Audio transcription using faster-whisper (optimized for Apple Silicon)
"""

import logging
from typing import List, Dict, Optional, Tuple
import multiprocessing  # For detecting CPU cores

# torch is imported inside transcribe() rather than here. It is used on exactly
# one line — the MPS check below — but a module-level import made every pure
# helper in this file (get_optimal_thread_count, word_confidence_stats) and
# every test of them require a multi-hundred-MB dependency. See CLD-225.


logger = logging.getLogger(__name__)

# Try to import faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False
    logger.warning("faster-whisper not installed. Install with: pip install faster-whisper")


def word_confidence_stats(words) -> Dict[str, float]:
    """Aggregate per-word ASR confidences into segment-level stats.

    Args:
        words: list of word dicts as produced by transcribe() — each may
            carry a 'probability' float (0..1). Non-dict entries and
            missing/non-numeric probabilities are ignored.

    Returns:
        {'word_conf_mean': ..., 'word_conf_min': ...} rounded to 4 places,
        or {} when no usable probabilities exist (never fabricates a value).
    """
    probs = []
    for w in words or ():
        if not isinstance(w, dict):
            continue
        p = w.get('probability')
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            probs.append(float(p))
    if not probs:
        return {}
    return {
        'word_conf_mean': round(sum(probs) / len(probs), 4),
        'word_conf_min': round(min(probs), 4),
    }


def get_optimal_thread_count() -> int:
    """
    Determine the optimal number of CPU threads for faster-whisper.
    
    Returns:
        int: Number of threads to use
    """
    # Get total CPU cores
    total_cores = multiprocessing.cpu_count()
    
    # For Apple Silicon (M1/M2/M3), we want to use most cores but leave some for system
    if total_cores <= 4:
        # Small systems: use all but 1 core
        return max(1, total_cores - 1)
    elif total_cores <= 8:
        # Medium systems: use all but 2 cores
        return max(1, total_cores - 2)
    else:
        # Large systems: use 75-80% of cores
        return max(1, int(total_cores * 0.75))
    
    # Fallback
    return min(4, total_cores)



def transcribe(
    audio_path: str,
    source_lang: Optional[str] = None,
    model_size: str = "large-v3",
    device: str = "cpu",
    compute_type: str = "int8",
    num_threads: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    info: Optional[dict] = None,
) -> Tuple[List[Dict], Optional[str]]:
    """
    Transcribe audio using faster-whisper.
    
    Args:
        audio_path: Path to audio file (16kHz mono WAV recommended)
        source_lang: Source language code (e.g., 'en', 'fr', 'zh'). None for auto-detect.
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v2', 'large-v3')
        device: 'cpu' or 'cuda' (use 'cpu' for Apple Silicon)
        compute_type: 'int8', 'int8_float16', 'float16', 'float32'
        initial_prompt: Whisper initial prompt — a vocabulary hint (names,
            places, jargon) that biases spelling in the first window.
        info: optional dict filled with {"backend", "model"} actually used.
    
    Returns:
        Tuple of (segments list, detected_language)
    """
    if info is not None:
        info.update(backend="faster-whisper", model=model_size)

    if not FASTER_WHISPER_AVAILABLE:
        raise ImportError(
            "faster-whisper is not installed. "
            "Install it with: pip install faster-whisper"
        )
    
    # Auto-detect optimal thread count if not specified
    if num_threads is None:
        num_threads = get_optimal_thread_count()
    
    logger.info(f"faster-whisper: device={device}, compute_type={compute_type}, model={model_size}, threads={num_threads}")
    
    try:
        # For Apple Silicon MPS support - use CPU or try MPS
        import torch
        if device == "mps" or (device == "cpu" and hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            # faster-whisper doesn't directly support MPS, but we can use CPU with int8
            # which is already very optimized
            logger.info("Apple Silicon detected - using CPU with int8 optimization")
            device = "cpu"
            compute_type = "int8"
        
        # Initialize the model
        model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=num_threads,
            num_workers=1,
        )
        
        # Prepare transcription options
        # "auto" from the API means "auto-detect" — faster-whisper expects None for that
        effective_lang = None if source_lang and source_lang.strip().lower() == "auto" else source_lang
        options = {
            "language": effective_lang,  # None for auto-detection
            "beam_size": 5,
            "best_of": 5,
            "patience": 1,
            "length_penalty": 1.0,
            "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": True,
            "prompt_reset_on_temperature": 0.5,
            "initial_prompt": initial_prompt or None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": [-1],
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "word_timestamps": True,
            "prepend_punctuations": "\"'“¿([{-",
            "append_punctuations": "\"'.。,，!！?？:：”)]}、",
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.5,
                "min_speech_duration_ms": 250,
                "min_silence_duration_ms": 100,
                #"window_size_samples": 1024,
                "speech_pad_ms": 400,
                "max_speech_duration_s": float("inf"),
            },
        }
        
        # Run transcription
        logger.info(f"Starting transcription of: {audio_path}")
        segments, info = model.transcribe(audio_path, **options)
        
        # Process segments
        result_segments = []
        detected_lang = info.language if hasattr(info, 'language') else None
        
        for segment in segments:
            segment_data = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'language': detected_lang,
                # ASR confidence signals — kept so downstream (checkpoints,
                # UI, diagnostics) can flag low-confidence source segments.
                'avg_logprob': getattr(segment, 'avg_logprob', None),
                'no_speech_prob': getattr(segment, 'no_speech_prob', None),
                'words': []
            }
            
            # Add word-level timestamps if available
            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    segment_data['words'].append({
                        'start': word.start,
                        'end': word.end,
                        'word': word.word,
                        'probability': word.probability if hasattr(word, 'probability') else 1.0
                    })
            
            result_segments.append(segment_data)
            logger.debug(f"Segment: {segment.start:.2f}s - {segment.end:.2f}s: {segment.text}")
        
        logger.info(f"Transcription complete: {len(result_segments)} segments, language={detected_lang}")
        
        # If no language detected but source_lang provided, use source_lang
        if not detected_lang and source_lang:
            detected_lang = source_lang
            
        return result_segments, detected_lang
        
    except Exception as e:
        logger.error(f"faster-whisper transcription failed: {str(e)}")
        raise


def transcribe_fallback(
    audio_path: str,
    source_lang: Optional[str] = None,
    model_size: str = "base",
) -> Tuple[List[Dict], Optional[str]]:
    """
    Fallback to a smaller model if the main one fails.
    """
    logger.warning(f"Falling back to faster-whisper {model_size} model")
    return transcribe(
        audio_path,
        source_lang=source_lang,
        model_size=model_size,
        device="cpu",
        compute_type="int8",
    )


# For testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python transcriber.py <audio_file> [language]")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    lang = sys.argv[2] if len(sys.argv) > 2 else None
    
    segments, detected_lang = transcribe(audio_file, lang)
    
    for seg in segments:
        print(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}")