"""
CLAP Embedding Generator.

Turns a music library into the vector space everything else in this project is
built on: one L2-normalised 512-d embedding per track, plus the per-window
matrix those embeddings were pooled from, plus the library centroid.

What changed, and why (audit C3, C5, M8)
----------------------------------------
The previous generator handed each whole track to `ClapProcessor` with default
arguments.  For `laion/clap-htsat-unfused` those defaults are `max_length_s=10`
and `truncation="rand_trunc"`, so every embedding described a **randomly
positioned ten-second crop**: re-running generation produced a different vector
for the same file, and a track was less recognisable as itself than a stranger
was 36% of the time.

The window length is not a tunable.  HTSAT's input is fixed at 10 s of 48 kHz
audio; feeding it more only changes *which* ten seconds survive.  So the fix is
not a longer window, it is **all** the windows:

  * Non-overlapping 10 s windows tile the track.  The final window is
    end-aligned — it covers the last ten seconds, overlapping its predecessor —
    so the tail is neither dropped nor zero-padded.  (Zero-padding is actively
    harmful: CLAP maps silence to a consistent direction that then contaminates
    the pooled vector.)
  * Each window is exactly `WINDOW_SAMPLES` long, so inside `ClapFeatureExtractor`
    neither the truncation branch nor the padding branch fires.  The random crop
    that caused C3 cannot happen: there is nothing left to crop.
  * Windows whose RMS falls below `RMS_GATE` are dropped — lead-ins, run-outs,
    gaps between movements.  If *every* window is below the gate (a genuinely
    quiet ambient track) they are all kept rather than returning nothing.
  * The survivors are mean-pooled and re-normalised.  Mean, not max: max would
    let a seven-minute track with one matching passage score like a track that
    matches throughout, which is musical whiplash.  The per-window matrix is
    persisted so that decision can be revisited without regenerating.

`truncation` is pinned to `rand_trunc` explicitly.  It is the checkpoint's own
setting and the only one this checkpoint can consume: `fusion` stacks four mel
crops into a 4-channel tensor, and the *unfused* model's patch embedding takes
one channel — it raises a shape error rather than producing a worse embedding.

Batching (M8) is real this time: ~25 windows per track means ~26,000 forward
passes for a 690-track library.  Windows are batched, and the mel extraction —
which measurement shows is the actual bottleneck, not the GPU — runs on a
worker pool that overlaps with it.

Batches are filled from **one track at a time**.  Cross-track packing would be
marginally more efficient, but floating-point reductions depend on batch
composition, so a track's embedding would then depend on which tracks happened
to be next to it in the run.  Per-track batching makes bit-identical
reproduction a property of the *file*, which is what C3's acceptance criterion
asserts and what the test suite can check.
"""

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Reading/validating the artifact is pure numpy and lives in embeddings_io so
# the player never has to import torch.  SCHEMA_VERSION is defined there, once.
from embeddings_io import SCHEMA_VERSION

# Try to import torch/transformers - these are optional dependencies
try:
    import torch
    import torchaudio
    import transformers
    from transformers import ClapModel, ClapProcessor
    CLAP_AVAILABLE = True
except ImportError:
    CLAP_AVAILABLE = False
    torch = None
    torchaudio = None
    transformers = None
    ClapModel = None
    ClapProcessor = None


class EmbeddingGenerationError(Exception):
    """Exception raised during embedding generation."""
    pass


@dataclass
class PreparedTrack:
    """A decoded track reduced to model-ready mel features. The waveform is
    dropped as soon as the features exist — full-length waveforms are hundreds
    of megabytes and several tracks are in flight at once."""
    track_file: str
    features: Optional[Any] = None       # torch.Tensor (W, mel_bins, frames)
    is_longer: Optional[Any] = None      # torch.Tensor (W, 1)
    window_rms: List[float] = field(default_factory=list)
    n_windows_total: int = 0
    n_windows_gated: int = 0
    error: Optional[str] = None


def window_bounds(n_samples: int, window: int) -> List[Tuple[int, int]]:
    """
    Tile `n_samples` with non-overlapping windows of length `window`, ending with
    an end-aligned window when the track does not divide evenly.

    Pure arithmetic, no torch — so the tiling is testable without the model.

      length 25 s, window 10 s -> [0:10] [10:20] [15:25]
      length 20 s, window 10 s -> [0:10] [10:20]
      length  4 s, window 10 s -> [0:4]   (padded by the feature extractor)
    """
    if n_samples <= 0:
        return []
    if n_samples <= window:
        return [(0, n_samples)]

    starts = list(range(0, n_samples - window + 1, window))
    if starts[-1] + window < n_samples:
        starts.append(n_samples - window)
    return [(s, s + window) for s in starts]


class CLAPEmbeddingGenerator:
    """
    Generates CLAP embeddings for audio files.
    Uses the laion/clap-htsat-unfused model from HuggingFace.
    """

    MODEL_NAME = "laion/clap-htsat-unfused"
    EMBEDDING_DIM = 512
    TARGET_SAMPLE_RATE = 48000

    # HTSAT's fixed input size.  Not a parameter — see the module docstring.
    WINDOW_SECONDS = 10
    WINDOW_SAMPLES = WINDOW_SECONDS * TARGET_SAMPLE_RATE

    # Silence gate, as RMS of a peak-normalised window.
    #
    # Measured over 1,475 windows from 40 random tracks of this library, the
    # distribution is bimodal: 1.8% of windows sit at RMS ~4e-5 (digital
    # silence) and the quietest real content starts around 0.04 (5th
    # percentile).  Every threshold between 0.001 and 0.01 drops the same
    # windows to within half a percent, so 0.01 sits inside a measured gap
    # rather than on a slope — it is a boundary the data draws, not one chosen
    # for it.
    RMS_GATE = 0.01

    # The checkpoint's own setting, pinned rather than inherited: `fusion` is
    # for the -fused variant and this model cannot consume its 4-channel output.
    TRUNCATION = 'rand_trunc'

    PARTIAL_SAVE_INTERVAL = 100   # Save progress every N tracks
    REQUIRED_DISK_SPACE_MB = 700  # Model download plus the window matrix

    def __init__(
        self,
        device: str = 'auto',
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
        workers: int = 4,
    ):
        """
        Args:
            device:     'cpu', 'cuda', 'cuda:0', or 'auto'
            cache_dir:  Optional directory for model cache
            batch_size: Windows per forward pass
            workers:    Threads decoding audio and extracting mel features.
                        Measurement puts the bottleneck there, not on the GPU:
                        one thread sustains ~37 windows/s, four sustain ~75.
        """
        if not CLAP_AVAILABLE:
            raise ImportError(
                "CLAP dependencies not available. Install with:\n"
                "  pip install transformers torch torchaudio --break-system-packages\n"
                "or:\n"
                "  conda install transformers pytorch torchaudio -c pytorch"
            )

        self.batch_size = max(1, int(batch_size))
        self.workers = max(1, int(workers))
        self.cache_dir = cache_dir

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.device_name = self._get_device_name()

        self.model = None
        self.processor = None

    # ------------------------------------------------------------------ setup

    def _get_device_name(self) -> str:
        """Get human-readable device name."""
        if self.device.startswith('cuda'):
            if torch.cuda.is_available():
                device_id = 0 if self.device == 'cuda' else int(self.device.split(':')[1])
                return f"CUDA ({torch.cuda.get_device_name(device_id)})"
            return "CUDA (not available)"
        return "CPU"

    def load_model(self, progress_callback: Optional[Callable] = None):
        """Load CLAP model and processor, downloading them if not cached."""
        if self.model is not None:
            return

        if progress_callback:
            progress_callback("Loading CLAP model...")

        try:
            cache_dir = self.cache_dir or Path.home() / '.cache' / 'huggingface'
            self._check_disk_space(cache_dir, self.REQUIRED_DISK_SPACE_MB)

            self.processor = ClapProcessor.from_pretrained(
                self.MODEL_NAME, cache_dir=self.cache_dir
            )
            self.model = ClapModel.from_pretrained(
                self.MODEL_NAME, cache_dir=self.cache_dir
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            # Determinism (C3).  Autotuning picks convolution algorithms by
            # timing them, which can differ between runs of the same input; with
            # benchmarking off the choice is fixed.
            if self.device.startswith('cuda'):
                torch.backends.cudnn.benchmark = False

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
            raise EmbeddingGenerationError(f"Failed to load CLAP model: {e}")
        except Exception as e:
            raise EmbeddingGenerationError(
                f"Failed to load CLAP model: {e}\n"
                "If this persists, try deleting the cache directory and retrying:\n"
                f"  rm -rf ~/.cache/huggingface/hub/models--laion--clap-htsat-unfused"
            )

    def _check_disk_space(self, path: Path, required_mb: int):
        """Refuse to start a download that cannot finish."""
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

    # ------------------------------------------------------------------ audio

    def load_audio(self, audio_path: str):
        """
        Load one file as a mono 48 kHz peak-normalised 1-D tensor.

        Peak normalisation is what makes `RMS_GATE` meaningful: the gate is then
        relative to the track's own loudest moment rather than to whatever
        mastering level the release happened to use.

        Raises on failure rather than returning None — the caller records the
        exception per file in `failed.txt` (C3: the original run lost 16 tracks
        with no diagnostic).
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sample_rate != self.TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        max_val = torch.max(torch.abs(waveform))
        if max_val > 1e-8:
            waveform = waveform / max_val

        return waveform.reshape(-1)

    def slice_windows(self, waveform) -> Tuple[List[Any], List[float], int, int]:
        """
        Cut a waveform into gated windows.

        Returns (windows, rms_of_kept, n_total, n_gated).  If the gate would
        remove everything, nothing is removed: a quiet ambient track is still a
        track, and an empty window list would be a silent data loss.
        """
        bounds = window_bounds(int(waveform.shape[0]), self.WINDOW_SAMPLES)
        windows, rms = [], []
        for start, end in bounds:
            segment = waveform[start:end]
            windows.append(segment)
            rms.append(float(torch.sqrt(torch.mean(segment.double() ** 2))))

        keep = [i for i, r in enumerate(rms) if r >= self.RMS_GATE]
        if not keep:
            keep = list(range(len(windows)))

        return (
            [windows[i] for i in keep],
            [rms[i] for i in keep],
            len(windows),
            len(windows) - len(keep),
        )

    def prepare_track(self, track_file: str, music_dir: Path) -> PreparedTrack:
        """
        Decode → window → gate → mel features.  Runs on a worker thread; the
        result carries no waveform, only what the GPU needs.
        """
        prepared = PreparedTrack(track_file=track_file)
        try:
            waveform = self.load_audio(str(music_dir / track_file))
            if waveform.numel() == 0:
                prepared.error = "empty waveform"
                return prepared

            windows, rms, n_total, n_gated = self.slice_windows(waveform)
            del waveform

            inputs = self.processor(
                audio=[w.numpy() for w in windows],
                return_tensors="pt",
                sampling_rate=self.TARGET_SAMPLE_RATE,
                truncation=self.TRUNCATION,
            )
            prepared.features = inputs['input_features']
            prepared.is_longer = inputs.get('is_longer')
            prepared.window_rms = rms
            prepared.n_windows_total = n_total
            prepared.n_windows_gated = n_gated
        except Exception as exc:                      # noqa: BLE001 - recorded per file
            prepared.error = f"{type(exc).__name__}: {exc}"
        return prepared

    # ------------------------------------------------------------------ model

    def embed_prepared(self, prepared: PreparedTrack) -> np.ndarray:
        """
        Run one track's windows through the audio tower.

        Windows are chunked in fixed `batch_size` groups drawn only from this
        track, so the result depends on the file and the batch size and nothing
        else.  Returns (n_windows, dim), L2-normalised.
        """
        if self.model is None:
            raise EmbeddingGenerationError("Model not loaded. Call load_model() first.")

        features = prepared.features
        outputs = []
        for start in range(0, features.shape[0], self.batch_size):
            chunk = {'input_features': features[start:start + self.batch_size].to(self.device)}
            if prepared.is_longer is not None:
                chunk['is_longer'] = prepared.is_longer[start:start + self.batch_size].to(self.device)
            with torch.no_grad():
                result = self.model.get_audio_features(**chunk)
            outputs.append(self._as_matrix(result))

        embeddings = np.concatenate(outputs, axis=0).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return embeddings / norms

    @staticmethod
    def _as_matrix(model_output) -> np.ndarray:
        """CLAP returns BaseModelOutputWithPooling; the projected, normalised
        embedding is `pooler_output`."""
        if hasattr(model_output, 'pooler_output'):
            tensor = model_output.pooler_output
        elif hasattr(model_output, 'last_hidden_state'):
            tensor = model_output.last_hidden_state.mean(dim=1)
        else:
            tensor = model_output
        return tensor.detach().cpu().numpy()

    @staticmethod
    def pool(window_embeddings: np.ndarray) -> np.ndarray:
        """Mean over windows, re-normalised — the track's single vector."""
        pooled = window_embeddings.mean(axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 1e-12:
            pooled = pooled / norm
        return pooled.astype(np.float32)

    def embed_track(self, track_file: str, music_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Whole pipeline for one file: (pooled, windows).  Raises
        EmbeddingGenerationError with the underlying cause on failure.

        This is the unit the determinism test exercises: calling it twice on the
        same file must return bit-identical arrays.
        """
        prepared = self.prepare_track(track_file, music_dir)
        if prepared.error:
            raise EmbeddingGenerationError(f"{track_file}: {prepared.error}")
        windows = self.embed_prepared(prepared)
        return self.pool(windows), windows

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        """Text tower — the half of the model the project never used (D5)."""
        if self.model is None:
            raise EmbeddingGenerationError("Model not loaded. Call load_model() first.")

        inputs = self.processor(text=list(prompts), return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            result = self.model.get_text_features(**inputs)
        embeddings = self._as_matrix(result).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return embeddings / norms

    # ------------------------------------------------------------- generation

    def generate_library(
        self,
        track_files: List[str],
        music_dir: Path,
        output_file: Path,
        progress_callback: Optional[Callable] = None,
        resume: bool = False,
    ) -> Dict[str, Any]:
        """
        Embed every track and write the artifact described in the audit's §7.

        Decoding and mel extraction run on `self.workers` threads while the main
        thread feeds the GPU.  Tracks are consumed in submission order, so the
        output is ordered by `track_files` regardless of which worker finished
        first — another thing determinism depends on.
        """
        self.load_model(progress_callback)

        partial_file = output_file.parent / f"{output_file.stem}.partial.npz"
        pooled: Dict[str, np.ndarray] = {}
        windows: Dict[str, np.ndarray] = {}

        if resume and partial_file.exists():
            pooled, windows = self._load_partial(partial_file)
            if progress_callback:
                progress_callback(f"Resuming: {len(pooled)} tracks already processed")

        todo = [t for t in track_files if t not in pooled]

        stats = {
            'total': len(track_files),
            'resumed': len(pooled),
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'failed_tracks': [],
            'windows': 0,
            'windows_gated': 0,
            'window_rms': [],
            'device': self.device_name,
            'start_time': datetime.now(),
        }

        # Bounded look-ahead: enough prepared tracks to keep the GPU fed, few
        # enough that several hundred megabytes of mel features never pile up.
        lookahead = self.workers * 2

        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool_executor:
                pending = []
                next_index = 0

                def submit_more():
                    nonlocal next_index
                    while len(pending) < lookahead and next_index < len(todo):
                        pending.append(
                            pool_executor.submit(self.prepare_track, todo[next_index], music_dir)
                        )
                        next_index += 1

                submit_more()
                while pending:
                    prepared = pending.pop(0).result()
                    submit_more()

                    stats['processed'] += 1
                    if progress_callback:
                        progress_callback({
                            'current': stats['resumed'] + stats['processed'],
                            'total': stats['total'],
                            'track': prepared.track_file,
                            'failed': stats['failed'],
                        })

                    if prepared.error:
                        stats['failed'] += 1
                        stats['failed_tracks'].append((prepared.track_file, prepared.error))
                        continue

                    try:
                        window_embeddings = self.embed_prepared(prepared)
                    except Exception as exc:          # noqa: BLE001 - recorded per file
                        stats['failed'] += 1
                        stats['failed_tracks'].append(
                            (prepared.track_file, f"{type(exc).__name__}: {exc}")
                        )
                        continue
                    finally:
                        prepared.features = None
                        prepared.is_longer = None

                    pooled[prepared.track_file] = self.pool(window_embeddings)
                    windows[prepared.track_file] = window_embeddings
                    stats['successful'] += 1
                    stats['windows'] += window_embeddings.shape[0]
                    stats['windows_gated'] += prepared.n_windows_gated
                    stats['window_rms'].extend(prepared.window_rms)

                    if stats['processed'] % self.PARTIAL_SAVE_INTERVAL == 0:
                        self._save_partial(pooled, windows, track_files, partial_file)
        except KeyboardInterrupt:
            # The CLI tells the user "partial results saved, use --resume".  Make
            # that true: without this, Ctrl-C anywhere in the 99 tracks between
            # checkpoints throws all of them away while the message claims
            # otherwise.
            self._save_partial(pooled, windows, track_files, partial_file)
            raise

        stats['end_time'] = datetime.now()
        stats['duration'] = (stats['end_time'] - stats['start_time']).total_seconds()

        if not pooled:
            raise EmbeddingGenerationError(
                "No track produced an embedding — nothing to write."
            )

        self._save_embeddings(pooled, windows, track_files, output_file, stats)
        if partial_file.exists():
            partial_file.unlink()

        return stats

    # ------------------------------------------------------------------- I/O

    @staticmethod
    def _stack(
        pooled: Dict[str, np.ndarray],
        windows: Dict[str, np.ndarray],
        order: Sequence[str],
    ) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
        """
        Flatten the per-track dicts into the CSR-style layout of §7: track *i*
        owns `windows[offsets[i]:offsets[i + 1]]`.
        """
        names = [t for t in order if t in pooled]
        if not names:
            # Reachable when the first PARTIAL_SAVE_INTERVAL tracks all fail.
            # np.stack([]) raises, and losing the run to a bookkeeping call
            # would be an absurd way to lose it.
            return [], np.zeros((0, 0), np.float32), np.zeros(1, np.int32), \
                np.zeros((0, 0), np.float32)

        embeddings = np.stack([pooled[t] for t in names]).astype(np.float32)

        counts = [windows[t].shape[0] for t in names]
        offsets = np.zeros(len(names) + 1, dtype=np.int32)
        np.cumsum(counts, out=offsets[1:])
        window_matrix = np.concatenate(
            [windows[t] for t in names], axis=0
        ).astype(np.float32)
        return names, embeddings, offsets, window_matrix

    def _save_partial(self, pooled, windows, order, partial_file: Path):
        """Checkpoint mid-run.  Same layout as the final file minus the
        centroid, which is only meaningful over a complete library."""
        names, embeddings, offsets, window_matrix = self._stack(pooled, windows, order)
        partial_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            partial_file,
            track_files=np.array(names, dtype=np.str_),
            embeddings=embeddings,
            window_offsets=offsets,
            windows=window_matrix,
        )

    @staticmethod
    def _load_partial(partial_file: Path):
        try:
            data = np.load(partial_file, allow_pickle=False)
            names = [str(t) for t in data['track_files']]
            offsets = data['window_offsets']
            matrix = data['windows']
            pooled = {t: data['embeddings'][i] for i, t in enumerate(names)}
            windows = {
                t: matrix[offsets[i]:offsets[i + 1]] for i, t in enumerate(names)
            }
            return pooled, windows
        except Exception as exc:                       # noqa: BLE001
            warnings.warn(f"Failed to load partial results: {exc}")
            return {}, {}

    def _save_embeddings(self, pooled, windows, order, output_file: Path, stats):
        """Write `track_embeddings.npz` to the schema in the audit's §7."""
        names, embeddings, offsets, window_matrix = self._stack(pooled, windows, order)

        # C5: stored uncentred, with the centroid alongside, so the centring
        # transform can be re-derived (or recomputed) as the library grows.
        centroid = embeddings.mean(axis=0).astype(np.float32)

        rms = np.asarray(stats.get('window_rms') or [0.0], dtype=np.float64)
        metadata = {
            'schema_version': SCHEMA_VERSION,
            'model': self.MODEL_NAME,
            'dimension': int(embeddings.shape[1]),
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'device': stats['device'],
            'transformers_version': getattr(transformers, '__version__', 'unknown'),
            'torch_version': getattr(torch, '__version__', 'unknown'),
            'window_scheme': {
                'length_s': self.WINDOW_SECONDS,
                'hop_s': self.WINDOW_SECONDS,          # non-overlapping; tail end-aligned
                'tail': 'end-aligned',
                'rms_gate': self.RMS_GATE,
                'truncation': self.TRUNCATION,
                'pooling': 'mean-then-renormalise',
                'batch_size': self.batch_size,
            },
            'stats': {
                'total': stats['total'],
                'successful': len(names),
                'failed': stats['failed'],
                'windows': int(window_matrix.shape[0]),
                'windows_gated': stats['windows_gated'],
                'window_rms_median': float(np.median(rms)),
                'window_rms_p05': float(np.percentile(rms, 5)),
                'duration_seconds': stats['duration'],
            },
        }

        output_file.parent.mkdir(parents=True, exist_ok=True)
        # Metadata is a JSON string rather than a pickled dict so the artifact
        # loads with `allow_pickle=False` (M5 asks for validation on load; not
        # executing arbitrary pickles while doing it is free).
        np.savez_compressed(
            output_file,
            schema_version=np.array(SCHEMA_VERSION),
            track_files=np.array(names, dtype=np.str_),
            embeddings=embeddings,
            centroid=centroid,
            window_offsets=offsets,
            windows=window_matrix,
            metadata=np.array(json.dumps(metadata)),
        )
        return metadata


def check_clap_available() -> tuple[bool, str]:
    """Report whether generation can run here, and on what."""
    if not CLAP_AVAILABLE:
        return False, (
            "CLAP dependencies not available.\n"
            "Install with:\n"
            "  pip install transformers torch torchaudio --break-system-packages\n"
            "or:\n"
            "  conda install transformers pytorch torchaudio -c pytorch"
        )

    if torch.cuda.is_available():
        return True, f"CLAP available (CUDA: {torch.cuda.get_device_name(0)})"
    return True, "CLAP available (CPU only — expect a long run)"
