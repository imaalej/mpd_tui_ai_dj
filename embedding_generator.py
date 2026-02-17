"""
CLAP Embedding Generator
Generates high-quality audio embeddings using Microsoft's CLAP model.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import warnings

# Try to import torch/transformers - these are optional dependencies
try:
    import torch
    import torchaudio
    from transformers import ClapModel, ClapProcessor
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    torch = None
    torchaudio = None
    ClapModel = None
    ClapProcessor = None


class EmbeddingGenerationError(Exception):
    """Exception raised during embedding generation."""
    pass


class CLAPEmbeddingGenerator:
    """
    Generates CLAP embeddings for audio files.
    Uses laion/clap-htsat-unfused model from HuggingFace.
    """
    
    MODEL_NAME = "laion/clap-htsat-unfused"
    EMBEDDING_DIM = 512
    TARGET_SAMPLE_RATE = 48000
    
    # Generation constants
    PARTIAL_SAVE_INTERVAL = 100  # Save progress every N tracks
    REQUIRED_DISK_SPACE_MB = 700  # Required disk space for model download
    
    def __init__(
        self,
        device: str = 'auto',
        cache_dir: Optional[Path] = None,
        batch_size: int = 16
    ):
        """
        Initialize CLAP embedding generator.
        
        Args:
            device: Device to use ('cpu', 'cuda', 'cuda:0', or 'auto')
            cache_dir: Optional directory for model cache
            batch_size: Batch size for processing
        """
        if not CLAP_AVAILABLE:
            raise ImportError(
                "CLAP dependencies not available. Install with:\n"
                "  pip install transformers torch torchaudio --break-system-packages\n"
                "or:\n"
                "  conda install transformers pytorch torchaudio -c pytorch"
            )
        
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Track device info
        self.device_name = self._get_device_name()
        
        # Model and processor (loaded on demand)
        self.model = None
        self.processor = None
        
    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = 0 if self.device == 'cuda' else int(self.device.split(':')[1])
                return f"CUDA ({torch.cuda.get_device_name(device_id)})"
            return "CUDA (not available)"
        return "CPU"
    
    def load_model(self, progress_callback: Optional[Callable] = None):
        """
        Load CLAP model and processor.
        Downloads model if not cached.
        
        Args:
            progress_callback: Optional callback for download progress
        """
        if self.model is not None:
            return
        
        if progress_callback:
            progress_callback("Loading CLAP model...")
        
        try:
            # Check disk space before download (model is ~600MB)
            cache_dir = self.cache_dir or Path.home() / '.cache' / 'huggingface'
            self._check_disk_space(cache_dir, self.REQUIRED_DISK_SPACE_MB)
            
            # Load processor
            self.processor = ClapProcessor.from_pretrained(
                self.MODEL_NAME,
                cache_dir=self.cache_dir
            )
            
            # Load model
            self.model = ClapModel.from_pretrained(
                self.MODEL_NAME,
                cache_dir=self.cache_dir
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            if progress_callback:
                progress_callback(f"Model loaded on {self.device_name}")
                
        except ImportError as e:
            raise EmbeddingGenerationError(
                f"Missing dependencies: {e}\n"
                "Install with: pip install transformers torch torchaudio --break-system-packages"
            )
        except OSError as e:
            if "Connection" in str(e) or "Network" in str(e):
                raise EmbeddingGenerationError(
                    f"Network error during model download: {e}\n"
                    "Please check your internet connection and try again.\n"
                    "If the download was interrupted, it will resume automatically on retry."
                )
            else:
                raise EmbeddingGenerationError(f"Failed to load CLAP model: {e}")
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to load CLAP model: {e}\n"
                "If this persists, try deleting the cache directory and retrying:\n"
                f"  rm -rf ~/.cache/huggingface/hub/models--laion--clap-htsat-unfused"
            )
    
    def _check_disk_space(self, path: Path, required_mb: int):
        """
        Check if sufficient disk space is available.
        
        Args:
            path: Directory to check
            required_mb: Required space in megabytes
            
        Raises:
            EmbeddingGenerationError: If insufficient space
        """
        import os
        
        try:
            path.mkdir(parents=True, exist_ok=True)
            stat = os.statvfs(path)
            available_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
            
            if available_mb < required_mb:
                raise EmbeddingGenerationError(
                    f"Insufficient disk space. Need {required_mb}MB, "
                    f"but only {available_mb:.0f}MB available at {path}"
                )
        except AttributeError:
            # Windows doesn't have statvfs, skip check
            pass
    
    def load_audio(self, audio_path: str) -> Optional[torch.Tensor]:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor or None if failed
        """
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert stereo to mono (average channels)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != self.TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate,
                    new_freq=self.TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)
            
            # Normalize amplitude to [-1, 1] range
            max_val = torch.max(torch.abs(waveform))
            if max_val > 1e-8:
                waveform = waveform / max_val
            
            # Squeeze to 1D
            waveform = waveform.squeeze()
            
            return waveform
            
        except Exception as e:
            warnings.warn(f"Failed to load audio {audio_path}: {e}")
            return None
    
    def generate_embedding(self, audio: torch.Tensor) -> np.ndarray:
        """
        Generate CLAP embedding for audio tensor.
        
        Args:
            audio: Preprocessed audio tensor
            
        Returns:
            Normalized embedding vector
        """
        if self.model is None:
            raise EmbeddingGenerationError("Model not loaded. Call load_model() first.")
        
        try:
            # Process audio
            inputs = self.processor(
                audio=audio.numpy(),  # Fixed: Changed from 'audios' to 'audio'
                return_tensors="pt",
                sampling_rate=self.TARGET_SAMPLE_RATE
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                audio_features = self.model.get_audio_features(**inputs)
            
            # Extract embedding tensor from model output
            # CLAP returns BaseModelOutputWithPooling, need to get the pooler_output or last_hidden_state
            if hasattr(audio_features, 'pooler_output'):
                embedding_tensor = audio_features.pooler_output
            elif hasattr(audio_features, 'last_hidden_state'):
                # Use mean pooling if pooler_output not available
                embedding_tensor = audio_features.last_hidden_state.mean(dim=1)
            else:
                # Fallback: assume it's already a tensor
                embedding_tensor = audio_features
            
            # Convert to numpy and normalize
            embedding = embedding_tensor.cpu().numpy()
            if len(embedding.shape) > 1:
                embedding = embedding[0]  # Take first item if batched
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            raise EmbeddingGenerationError(f"Failed to generate embedding: {e}")
    
    def generate_embeddings_batch(
        self,
        track_files: List[str],
        music_dir: Path,
        output_file: Path,
        progress_callback: Optional[Callable] = None,
        resume: bool = False
    ) -> Dict[str, Any]:
        """
        Generate embeddings for a batch of tracks.
        
        Args:
            track_files: List of track file paths (relative to music_dir)
            music_dir: Base music directory
            output_file: Output file path (.npz)
            progress_callback: Optional progress callback
            resume: Whether to resume from partial results
            
        Returns:
            Dictionary with generation statistics
        """
        # Load model
        self.load_model(progress_callback)
        
        # Check for existing partial results
        partial_file = output_file.parent / f"{output_file.stem}.partial.npz"
        processed_tracks = set()
        embeddings_dict = {}
        
        if resume and partial_file.exists():
            try:
                partial_data = np.load(partial_file, allow_pickle=True)
                embeddings_dict = dict(zip(
                    partial_data['track_files'],
                    partial_data['embeddings']
                ))
                processed_tracks = set(partial_data['track_files'])
                if progress_callback:
                    progress_callback(f"Resuming: {len(processed_tracks)} tracks already processed")
            except Exception as e:
                warnings.warn(f"Failed to load partial results: {e}")
        
        # Filter tracks to process
        tracks_to_process = [t for t in track_files if t not in processed_tracks]
        
        # Statistics
        stats = {
            'total': len(track_files),
            'processed': len(processed_tracks),
            'successful': len(processed_tracks),
            'failed': 0,
            'failed_tracks': [],
            'device': self.device_name,
            'start_time': datetime.now()
        }
        
        # Process tracks
        for i, track_file in enumerate(tracks_to_process):
            full_path = music_dir / track_file
            
            # Progress callback
            if progress_callback:
                progress_info = {
                    'current': stats['processed'] + 1,
                    'total': stats['total'],
                    'track': track_file,
                    'failed': stats['failed']
                }
                progress_callback(progress_info)
            
            # Load and process audio
            audio = self.load_audio(str(full_path))
            if audio is None:
                stats['failed'] += 1
                stats['failed_tracks'].append((track_file, "Failed to load audio"))
                continue
            
            try:
                # Generate embedding
                embedding = self.generate_embedding(audio)
                embeddings_dict[track_file] = embedding
                stats['successful'] += 1
                stats['processed'] += 1
                
                # Save partial results periodically
                if stats['processed'] % self.PARTIAL_SAVE_INTERVAL == 0:
                    self._save_partial(embeddings_dict, partial_file)
                
            except Exception as e:
                stats['failed'] += 1
                stats['failed_tracks'].append((track_file, str(e)))
                warnings.warn(f"Failed to generate embedding for {track_file}: {e}")
        
        # Save final results
        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()
        
        if embeddings_dict:
            self._save_embeddings(embeddings_dict, output_file, stats)
            
            # Remove partial file if successful
            if partial_file.exists():
                partial_file.unlink()
        
        return stats
    
    def _save_partial(self, embeddings_dict: Dict[str, np.ndarray], partial_file: Path):
        """Save partial results."""
        track_files = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[t] for t in track_files])
        
        np.savez_compressed(
            partial_file,
            track_files=track_files,
            embeddings=embeddings
        )
    
    def _save_embeddings(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        output_file: Path,
        stats: Dict[str, Any]
    ):
        """Save final embeddings with metadata."""
        track_files = list(embeddings_dict.keys())
        embeddings = np.array([embeddings_dict[t] for t in track_files])
        
        metadata = {
            'model': self.MODEL_NAME,
            'dimension': self.EMBEDDING_DIM,
            'generated_at': datetime.now().isoformat(),
            'n_tracks': len(track_files),
            'device': self.device_name,
            'version': '1.0',
            'stats': {
                'total': stats['total'],
                'successful': stats['successful'],
                'failed': stats['failed'],
                'duration_seconds': stats['duration']
            }
        }
        
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(
            output_file,
            track_files=track_files,
            embeddings=embeddings,
            metadata=np.array([metadata])  # Wrap in array for npz compatibility
        )


def validate_embeddings(embeddings_file: Path) -> Dict[str, Any]:
    """
    Validate embeddings file.
    
    Args:
        embeddings_file: Path to embeddings file
        
    Returns:
        Validation results dictionary
    """
    try:
        data = np.load(embeddings_file, allow_pickle=True)
        
        # Check required fields
        if 'track_files' not in data or 'embeddings' not in data:
            return {
                'valid': False,
                'error': 'Missing required fields (track_files, embeddings)'
            }
        
        track_files = data['track_files']
        embeddings = data['embeddings']
        
        # Check dimensions
        if len(embeddings.shape) != 2:
            return {
                'valid': False,
                'error': f'Invalid embedding shape: {embeddings.shape}'
            }
        
        n_tracks, embed_dim = embeddings.shape
        
        # Check normalization
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=0.01):
            return {
                'valid': False,
                'error': f'Embeddings not normalized (norm range: {norms.min():.3f} - {norms.max():.3f})'
            }
        
        # Extract metadata
        metadata = None
        if 'metadata' in data:
            metadata = data['metadata'].item()
        
        return {
            'valid': True,
            'n_tracks': n_tracks,
            'dimension': embed_dim,
            'metadata': metadata,
            'norm_min': float(norms.min()),
            'norm_max': float(norms.max()),
            'norm_mean': float(norms.mean())
        }
        
    except Exception as e:
        return {
            'valid': False,
            'error': f'Failed to validate: {e}'
        }


def get_embedding_stats(embeddings_file: Path) -> Dict[str, Any]:
    """
    Get statistics about embeddings.
    
    Args:
        embeddings_file: Path to embeddings file
        
    Returns:
        Statistics dictionary
    """
    try:
        data = np.load(embeddings_file, allow_pickle=True)
        embeddings = data['embeddings']
        track_files = data['track_files']
        
        # Calculate similarity statistics
        # Sample random pairs to avoid memory issues
        n_samples = min(1000, len(embeddings))
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        sample_embeddings = embeddings[indices]
        
        # Pairwise similarities
        similarities = np.dot(sample_embeddings, sample_embeddings.T)
        
        # Exclude diagonal
        mask = ~np.eye(n_samples, dtype=bool)
        similarities = similarities[mask]
        
        # Quality assessment
        quality_score = _assess_embedding_quality(similarities)
        
        metadata = None
        if 'metadata' in data:
            metadata = data['metadata'].item()
        
        return {
            'n_tracks': len(embeddings),
            'dimension': embeddings.shape[1],
            'similarity_min': float(similarities.min()),
            'similarity_max': float(similarities.max()),
            'similarity_mean': float(similarities.mean()),
            'similarity_std': float(similarities.std()),
            'quality_score': quality_score,
            'quality_assessment': _quality_interpretation(quality_score),
            'metadata': metadata
        }
        
    except Exception as e:
        return {'error': f'Failed to compute stats: {e}'}


def _assess_embedding_quality(similarities: np.ndarray) -> float:
    """
    Assess embedding quality based on similarity distribution.
    
    Args:
        similarities: Array of pairwise similarities
        
    Returns:
        Quality score from 0-100
    """
    mean_sim = similarities.mean()
    std_sim = similarities.std()
    min_sim = similarities.min()
    max_sim = similarities.max()
    
    # Good embeddings should have:
    # 1. Moderate mean (0.3-0.5) - not all similar, not all different
    # 2. Good spread (std > 0.15) - can distinguish tracks
    # 3. Range that uses the space (-0.2 to 0.9)
    
    score = 100.0
    
    # Penalize if mean is too high (all tracks similar)
    if mean_sim > 0.6:
        score -= (mean_sim - 0.6) * 100
    
    # Penalize if mean is too low (all tracks different)
    if mean_sim < 0.2:
        score -= (0.2 - mean_sim) * 100
    
    # Penalize if standard deviation is too low (no discrimination)
    if std_sim < 0.15:
        score -= (0.15 - std_sim) * 200
    
    # Penalize if range is too narrow (not using embedding space)
    sim_range = max_sim - min_sim
    if sim_range < 0.6:
        score -= (0.6 - sim_range) * 50
    
    return max(0.0, min(100.0, score))


def _quality_interpretation(score: float) -> str:
    """Interpret quality score."""
    if score >= 80:
        return "Excellent - Embeddings show good discrimination"
    elif score >= 60:
        return "Good - Embeddings are usable"
    elif score >= 40:
        return "Fair - Embeddings may work but check similarity distribution"
    elif score >= 20:
        return "Poor - Embeddings show limited discrimination"
    else:
        return "Very Poor - Embeddings may be random or corrupted"


def check_clap_available() -> tuple[bool, str]:
    """
    Check if CLAP dependencies are available.
    
    Returns:
        (available, message) tuple
    """
    if not CLAP_AVAILABLE:
        return False, (
            "CLAP dependencies not available.\n"
            "Install with:\n"
            "  pip install transformers torch torchaudio --break-system-packages\n"
            "or:\n"
            "  conda install transformers pytorch torchaudio -c pytorch"
        )
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        return True, f"CLAP available (CUDA: {device_name})"
    else:
        return True, "CLAP available (CPU only)"
