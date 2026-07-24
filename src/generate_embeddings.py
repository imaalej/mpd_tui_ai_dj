#!/usr/bin/env python3
"""
Generate CLAP Embeddings CLI.

Builds both data artifacts the player reads: `track_embeddings.npz` (pooled
vectors, per-window matrix, library centroid) and `descriptors.npz` (the CLAP
text-tower descriptor bank).  They are produced in one run because the bank's
z-score baselines are measured against the centred library — building it
separately invites the two to drift apart.

Tracks are enumerated from `mpc listall` (audit M4).  That is the only
enumeration that answers the question that matters — "what will MPD actually
play" — and it is the same key space the embeddings are stored under, so a
mismatch cannot arise between generation and playback.  The filesystem walk this
used to do produced keys MPD might never accept.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

import descriptor_bank
import mood_axes
from config import config
from embedding_generator import (
    CLAPEmbeddingGenerator,
    EmbeddingGenerationError,
    check_clap_available,
)
from embeddings_io import (
    atomic_savez,
    centre,
    get_embedding_stats,
    similarity_report,
    validate_embeddings,
    write_failed_tracks,
)
from music_directory import validate_music_directory


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"


def format_size(num_bytes: int) -> str:
    """Format file size in human-readable format."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    if num_bytes < 1024 ** 3:
        return f"{num_bytes / (1024 ** 2):.1f} MB"
    return f"{num_bytes / (1024 ** 3):.1f} GB"


class ProgressTracker:
    """Track and display progress during embedding generation."""

    def __init__(self, total: int):
        self.total = max(1, total)
        self.current = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_update = 0.0

    def update(self, info):
        if isinstance(info, str):
            print(info)
            return

        self.current = info['current']
        self.failed = info.get('failed', 0)
        current_track = info.get('track', '')

        now = time.time()
        if now - self.last_update < 0.5 and self.current < self.total:
            return
        self.last_update = now

        elapsed = now - self.start_time
        speed = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / speed if speed > 0 else 0

        percent = (self.current / self.total) * 100
        bar_length = 40
        filled = int(bar_length * self.current // self.total)
        bar = '=' * filled + '>' + ' ' * max(0, bar_length - filled - 1)

        failed_note = f" | {self.failed} failed" if self.failed else ""
        print(f"\r[{bar}] {self.current}/{self.total} ({percent:.1f}%) | "
              f"{speed:.2f} tracks/sec | ETA: {format_duration(eta)}{failed_note}", end='')

        if current_track and self.current < self.total:
            max_len = 60
            shown = current_track if len(current_track) <= max_len else current_track[:max_len - 3] + '...'
            print(f"\nCurrent: {shown}\033[K", end='')
            print("\033[F", end='')

        if self.current >= self.total:
            print()


# ----------------------------------------------------------------- enumeration

def enumerate_tracks() -> Optional[List[str]]:
    """
    `mpc listall`, the single source of truth for track keys (M4).

    Returns None when MPD cannot be reached or has nothing — the caller reports;
    there is no filesystem fallback by design.
    """
    from mpd_controller import MPDController

    mpd = MPDController()
    if not mpd.connect():
        print(f"❌ Cannot reach MPD at {config.mpd_host}:{config.mpd_port}")
        print("   Embedding keys come from `mpc listall`, so MPD must be running.")
        return None

    tracks = sorted(mpd.list_all_tracks())
    if not tracks:
        print("❌ MPD's database is empty. Run `mpc update` first.")
        return None
    return tracks


def check_music_directory(tracks: Sequence[str]) -> Optional[Path]:
    """
    Resolve and prove the music directory before anything expensive (M3).

    A wrong directory used to waste the entire generation run; now it costs five
    `Path.exists()` calls to find out.
    """
    music_dir = Path(config.mpd_music_directory)
    print(f"Music directory: {music_dir}  (source: {config.mpd_music_directory_source})")

    ok, message = validate_music_directory(music_dir, tracks)
    if not ok:
        print(f"❌ {message}")
        return None
    print(f"✓ {message}")
    return music_dir


# ------------------------------------------------------------------ descriptors

def build_descriptor_bank(
    generator: CLAPEmbeddingGenerator,
    embeddings_file: Path,
    output_file: Path,
    template_name: str = descriptor_bank.DEFAULT_TEMPLATE,
) -> int:
    """
    Embed the descriptor prompts, measure them against the centred library, and
    apply the variance gate (D5a/D5b).
    """
    data = np.load(embeddings_file, allow_pickle=False)
    library = centre(data['embeddings'], data['centroid'])

    template = descriptor_bank.PROMPT_TEMPLATES[template_name]
    labels = list(descriptor_bank.DESCRIPTORS)
    prompts = descriptor_bank.render_prompts(labels, template)

    print(f"\nBuilding descriptor bank: {len(labels)} prompts, "
          f"template {template_name!r} → {template.format('…')}")
    generator.load_model()
    text_embeddings = generator.embed_texts(prompts)

    bank = descriptor_bank.build_bank(
        text_embeddings=text_embeddings,
        labels=labels,
        prompts=prompts,
        library_centred=library,
    )
    report = bank.pop('report')

    scores = library @ text_embeddings.T
    metrics = descriptor_bank.separation_metrics(scores)
    print(f"  library separation: mean std {metrics['mean_std']:.4f}, "
          f"effective rank {metrics['effective_rank']:.1f} of {len(labels)}")
    print(f"  variance gate: floor {report['std_floor']:.4f} "
          f"({report['std_floor_fraction']:.2f} × median std {report['median_std']:.4f})")

    print(f"\n  kept ({len(report['kept'])}), by std over the library:")
    for label, std in sorted(report['kept'], key=lambda kv: -kv[1]):
        print(f"    {std:.4f}  {label}")
    if report['dropped']:
        print(f"\n  dropped ({len(report['dropped'])}) — no information about this library:")
        for label, std in sorted(report['dropped'], key=lambda kv: -kv[1]):
            print(f"    {std:.4f}  {label}")
    else:
        print("\n  dropped: none — every descriptor discriminates this library")

    # Mood axes for the vibe cloud (Phase 4).  Selected against *this* library
    # and calibrated here, in the same run as the descriptor mean/std, so the two
    # cannot drift apart — and stored beside them (additive keys, bank schema
    # unchanged; see mood_axes.py).  Derived purely from the bank's text vectors
    # and the centred library, so a `--descriptors-only` rebuild refreshes them
    # without re-embedding any audio.
    selection = mood_axes.select_axes(bank['text_embeddings'], bank['labels'], library)
    bank.update(mood_axes.selection_arrays(selection))
    print(f"\n  mood axes (auto-selected legible triad, z-scored per library):")
    print(f"    {' · '.join(selection.labels)}")
    for name, score in selection.diagnostics['balanced_top'][:3]:
        tag = "   ← chosen" if name == selection.labels else ""
        print(f"      balanced {score:.3f}   {' · '.join(name)}{tag}")

    atomic_savez(output_file, bank, compressed=True)
    print(f"\n✓ Descriptor bank written to {output_file} "
          f"({format_size(output_file.stat().st_size)})")
    return 0


def compare_templates_command(args) -> int:
    """
    Score the candidate prompt templates against the real library (H1, step 2).

    "Which template" is an empirical question, so it gets an empirical answer:
    how far apart the bank spreads *this* library, and how many independent
    things it ends up measuring.
    """
    embeddings_file = Path(args.file or config.embeddings_file)
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1

    available, message = check_clap_available()
    if not available:
        print(f"❌ {message}")
        return 1

    data = np.load(embeddings_file, allow_pickle=False)
    library = centre(data['embeddings'], data['centroid'])

    generator = CLAPEmbeddingGenerator(device=args.device, batch_size=args.batch_size)
    generator.load_model()

    labels = list(descriptor_bank.DESCRIPTORS)
    print(f"Scoring {len(descriptor_bank.PROMPT_TEMPLATES)} templates against "
          f"{library.shape[0]} tracks\n")
    print(f"  {'template':<12} {'mean std':>9} {'min std':>9} {'eff. rank':>10}   example")
    for name, template in descriptor_bank.PROMPT_TEMPLATES.items():
        text_embeddings = generator.embed_texts(
            descriptor_bank.render_prompts(labels, template)
        )
        metrics = descriptor_bank.separation_metrics(library @ text_embeddings.T)
        print(f"  {name:<12} {metrics['mean_std']:>9.4f} {metrics['min_std']:>9.4f} "
              f"{metrics['effective_rank']:>10.1f}   {template.format('hypnotic')}")

    print("\nHigher mean std = the bank separates the library rather than saying the "
          "same thing about every track.\nHigher effective rank = the descriptors "
          "carry independent information rather than one axis repeated.")
    return 0


# -------------------------------------------------------------------- commands

def generate_command(args) -> int:
    print("=" * 70)
    print("CLAP Embedding Generator")
    print("=" * 70)

    available, message = check_clap_available()
    if not available:
        print(f"\n❌ {message}")
        return 1
    print(f"\n✓ {message}\n")

    output_file = Path(args.output or config.embeddings_file)
    if output_file.exists() and not args.force and not args.resume:
        print(f"❌ Output file already exists: {output_file}")
        print("   Use --force to overwrite or --resume to continue")
        return 1

    track_files = enumerate_tracks()
    if track_files is None:
        return 1
    print(f"MPD database: {len(track_files)} playable tracks")

    music_dir = check_music_directory(track_files)
    if music_dir is None:
        return 1

    try:
        generator = CLAPEmbeddingGenerator(
            device=args.device, batch_size=args.batch_size, workers=args.workers
        )
    except Exception as e:                             # noqa: BLE001
        print(f"❌ Failed to initialize generator: {e}")
        return 1

    print(f"\nGenerating embeddings:")
    print(f"  Model: {generator.MODEL_NAME}")
    print(f"  Device: {generator.device_name}")
    print(f"  Windows: {generator.WINDOW_SECONDS}s, non-overlapping, end-aligned tail, "
          f"RMS gate {generator.RMS_GATE}")
    print(f"  Batch: {generator.batch_size} windows · {generator.workers} decode workers")
    print(f"  Output: {output_file}")
    print()

    progress = ProgressTracker(len(track_files))
    try:
        stats = generator.generate_library(
            track_files=track_files,
            music_dir=music_dir,
            output_file=output_file,
            progress_callback=progress.update,
            resume=args.resume,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print("Partial results saved. Use --resume to continue.")
        return 1
    except EmbeddingGenerationError as e:
        print(f"\n\n❌ Generation failed: {e}")
        return 1
    except Exception as e:                             # noqa: BLE001
        print(f"\n\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    write_failed_tracks(stats['failed_tracks'], config.failed_tracks_file)

    duration = max(stats['duration'], 1e-6)
    print("\n" + "=" * 70)
    print("✓ Embedding generation complete!")
    print("=" * 70)
    print(f"  Device: {stats['device']}")
    print(f"  Tracks: {stats['successful']} embedded this run"
          + (f" (+{stats['resumed']} resumed)" if stats['resumed'] else "")
          + f", {stats['failed']} failed, {stats['total']} in MPD")
    print(f"  Windows: {stats['windows']} kept, {stats['windows_gated']} below the RMS gate")
    print(f"  Duration: {format_duration(stats['duration'])} "
          f"({stats['successful'] / duration:.2f} tracks/sec, "
          f"{stats['windows'] / duration:.1f} windows/sec)")
    print(f"  Output: {output_file} ({format_size(output_file.stat().st_size)})")

    if stats['failed']:
        print(f"\n⚠️  {stats['failed']} tracks failed — full list in {config.failed_tracks_file}")
        for track, error in stats['failed_tracks'][:10]:
            print(f"  - {track}: {error}")
        if len(stats['failed_tracks']) > 10:
            print(f"  ... and {len(stats['failed_tracks']) - 10} more")

    # C5, measured on what was just built rather than assumed.
    data = np.load(output_file, allow_pickle=False)
    raw = similarity_report(data['embeddings'])
    centred = similarity_report(centre(data['embeddings'], data['centroid']))
    print("\n  Random-pair similarity (the scale every weight is tuned against):")
    print(f"    raw     mean {raw['mean']:+.3f}  median {raw['median']:+.3f}  "
          f"p75 {raw['p75']:+.3f}")
    print(f"    centred mean {centred['mean']:+.3f}  median {centred['median']:+.3f}  "
          f"p75 {centred['p75']:+.3f}")

    if args.no_descriptors:
        print("\n(skipping the descriptor bank — build it with --descriptors-only)")
        return 0

    return build_descriptor_bank(
        generator, output_file, Path(args.descriptors or config.descriptors_file),
        template_name=args.template,
    )


def descriptors_command(args) -> int:
    """Rebuild only the descriptor bank against existing embeddings."""
    embeddings_file = Path(args.file or config.embeddings_file)
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1

    available, message = check_clap_available()
    if not available:
        print(f"❌ {message}")
        return 1

    generator = CLAPEmbeddingGenerator(device=args.device, batch_size=args.batch_size)
    return build_descriptor_bank(
        generator, embeddings_file,
        Path(args.descriptors or config.descriptors_file),
        template_name=args.template,
    )


def describe_command(args) -> int:
    """
    Name a few tracks — the spot check the audit asks for before anything is
    wired to the display (H1, "acceptance for the gate").
    """
    embeddings_file = Path(args.file or config.embeddings_file)
    bank = descriptor_bank.DescriptorBank.load(Path(args.descriptors or config.descriptors_file))
    if bank is None:
        return 1
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1

    data = np.load(embeddings_file, allow_pickle=False)
    names = [str(t) for t in data['track_files']]
    library = centre(data['embeddings'], data['centroid'])

    wanted = args.describe or []
    if not wanted:
        step = max(1, len(names) // 8)
        chosen = list(range(0, len(names), step))[:8]
    else:
        chosen = []
        for needle in wanted:
            matches = [i for i, n in enumerate(names) if needle.lower() in n.lower()]
            if not matches:
                print(f"  (no track matching {needle!r})")
            chosen.extend(matches[:1])

    for i in chosen:
        top = bank.top(library[i], n=args.top)
        words = ' · '.join(f"{label} ({z:+.1f})" for label, z in top)
        print(f"\n{names[i]}\n  {words}")
    return 0


def validate_command(args) -> int:
    embeddings_file = Path(args.file or config.embeddings_file)
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1

    print("Validating embeddings...")
    results = validate_embeddings(embeddings_file)
    if not results['valid']:
        print(f"❌ Validation failed: {results['error']}")
        return 1

    print("✓ Embeddings are valid")
    print(f"  Schema: {results['schema_version']}")
    print(f"  Tracks: {results['n_tracks']}")
    print(f"  Dimension: {results['dimension']}")
    print(f"  Windows: {results['n_windows']}")
    print(f"  Normalisation: {results['norm_min']:.4f} – {results['norm_max']:.4f}")

    metadata = results['metadata']
    if metadata:
        print("\nMetadata:")
        print(f"  Model: {metadata.get('model', 'unknown')}")
        print(f"  Generated: {metadata.get('generated_at', 'unknown')}")
        print(f"  Device: {metadata.get('device', 'unknown')}")
        scheme = metadata.get('window_scheme', {})
        if scheme:
            print(f"  Windows: {scheme.get('length_s')}s {scheme.get('tail')} tail, "
                  f"RMS gate {scheme.get('rms_gate')}, {scheme.get('pooling')}")
    return 0


def stats_command(args) -> int:
    embeddings_file = Path(args.file or config.embeddings_file)
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1

    print("Computing embedding statistics...")
    stats = get_embedding_stats(embeddings_file)
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return 1

    print("\nEmbedding Statistics")
    print("=" * 60)
    print(f"Tracks: {stats['n_tracks']}   Dimension: {stats['dimension']}   "
          f"Windows: {stats['n_windows']}")
    print(f"Centroid norm: {stats['centroid_norm']:.3f}  "
          "(how far off the origin the library sits — the anisotropy C5 corrects)")

    for name in ('raw', 'centred'):
        report = stats[name]
        print(f"\nRandom-pair similarity — {name} (n={report['n_sampled']}):")
        print(f"  mean {report['mean']:+.3f}   std {report['std']:.3f}")
        print("  percentiles  " + "  ".join(
            f"p{q}={report[f'p{q:02d}']:+.3f}" for q in (1, 5, 25, 75, 95, 99)
        ))
        print(f"  median {report['median']:+.3f}")

    metadata = stats['metadata']
    if metadata:
        gen = metadata.get('stats', {})
        print("\nGeneration:")
        print(f"  Model: {metadata.get('model', 'unknown')} on {metadata.get('device', 'unknown')}")
        print(f"  transformers {metadata.get('transformers_version', '?')}, "
              f"torch {metadata.get('torch_version', '?')}")
        print(f"  {gen.get('successful', '?')} succeeded, {gen.get('failed', '?')} failed, "
              f"{gen.get('windows', '?')} windows "
              f"({gen.get('windows_gated', '?')} gated as silence)")
        if gen.get('duration_seconds'):
            print(f"  Duration: {format_duration(gen['duration_seconds'])}")

    print(f"\nFile size: {format_size(embeddings_file.stat().st_size)}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Generate CLAP embeddings for the music library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 generate_embeddings.py                     # embeddings + descriptor bank
  python3 generate_embeddings.py --device cuda:0
  python3 generate_embeddings.py --resume
  python3 generate_embeddings.py --descriptors-only  # rebuild just the bank
  python3 generate_embeddings.py --compare-templates # choose the prompt template
  python3 generate_embeddings.py --describe "Arctic Monkeys"
  python3 generate_embeddings.py --validate
  python3 generate_embeddings.py --stats
        """
    )

    parser.add_argument('--batch-size', type=int, default=32,
                        help='Windows per forward pass (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Threads decoding audio and extracting features (default: 4)')
    parser.add_argument('--device', default='auto',
                        help='Device: cpu, cuda, cuda:0, or auto (default: auto)')
    parser.add_argument('--output', type=str, help='Embeddings output path (default: from config)')
    parser.add_argument('--descriptors', type=str,
                        help='Descriptor bank path (default: from config)')
    parser.add_argument('--template', default=descriptor_bank.DEFAULT_TEMPLATE,
                        choices=sorted(descriptor_bank.PROMPT_TEMPLATES),
                        help='Descriptor prompt template')
    parser.add_argument('--resume', action='store_true', help='Resume from interruption')
    parser.add_argument('--force', action='store_true', help='Overwrite existing embeddings')
    parser.add_argument('--no-descriptors', action='store_true',
                        help='Skip the descriptor bank after generating embeddings')
    parser.add_argument('--descriptors-only', action='store_true',
                        help='Rebuild the descriptor bank from existing embeddings')
    parser.add_argument('--compare-templates', action='store_true',
                        help='Score the prompt templates against the library')
    parser.add_argument('--describe', nargs='*',
                        help='Show top descriptors for tracks matching these substrings')
    parser.add_argument('--top', type=int, default=3, help='Descriptors to show (--describe)')
    parser.add_argument('--validate', action='store_true', help='Validate existing embeddings')
    parser.add_argument('--stats', action='store_true', help='Show statistics about embeddings')
    parser.add_argument('--file', type=str,
                        help='Embeddings file to read (--validate/--stats/--describe)')

    args = parser.parse_args()

    if args.validate:
        return validate_command(args)
    if args.stats:
        return stats_command(args)
    if args.compare_templates:
        return compare_templates_command(args)
    if args.describe is not None:
        return describe_command(args)
    if args.descriptors_only:
        return descriptors_command(args)
    return generate_command(args)


if __name__ == '__main__':
    sys.exit(main())
