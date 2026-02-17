#!/usr/bin/env python3
"""
Generate CLAP Embeddings CLI
Command-line tool for generating CLAP embeddings for the music library.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import time

from embedding_generator import (
    CLAPEmbeddingGenerator,
    validate_embeddings,
    get_embedding_stats,
    check_clap_available,
    CLAP_AVAILABLE
)
from config import config


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def format_size(bytes: int) -> str:
    """Format file size in human-readable format."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024 ** 2:
        return f"{bytes / 1024:.1f} KB"
    elif bytes < 1024 ** 3:
        return f"{bytes / (1024 ** 2):.1f} MB"
    else:
        return f"{bytes / (1024 ** 3):.1f} GB"


class ProgressTracker:
    """Track and display progress during embedding generation."""
    
    def __init__(self, total: int):
        self.total = total
        self.current = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_update = 0
        
    def update(self, info):
        """Update progress display."""
        if isinstance(info, str):
            # Simple message
            print(info)
            return
        
        # Detailed progress
        self.current = info['current']
        self.failed = info.get('failed', 0)
        current_track = info.get('track', '')
        
        # Don't update too frequently
        now = time.time()
        if now - self.last_update < 0.5 and self.current < self.total:
            return
        self.last_update = now
        
        # Calculate stats
        elapsed = now - self.start_time
        if self.current > 0:
            speed = self.current / elapsed
            eta = (self.total - self.current) / speed if speed > 0 else 0
        else:
            speed = 0
            eta = 0
        
        # Progress bar
        percent = (self.current / self.total) * 100
        bar_length = 40
        filled = int(bar_length * self.current // self.total)
        bar = '=' * filled + '>' + ' ' * (bar_length - filled - 1)
        
        # Format output (two lines for better readability)
        print(f"\r[{bar}] {self.current}/{self.total} ({percent:.1f}%) | "
              f"{speed:.2f} tracks/sec | ETA: {format_duration(eta)}", end='')
        
        # Show current track on next line if available
        if current_track and self.current < self.total:
            # Truncate track name if too long
            max_len = 60
            if len(current_track) > max_len:
                current_track = current_track[:max_len-3] + '...'
            print(f"\nCurrent: {current_track}", end='')
            # Move cursor back up for next update
            print("\033[F", end='')
        
        if self.current == self.total:
            print()  # New line at completion


def generate_command(args):
    """Generate embeddings command."""
    print("=" * 70)
    print("CLAP Embedding Generator")
    print("=" * 70)
    
    # Check CLAP availability
    available, message = check_clap_available()
    if not available:
        print(f"\n❌ {message}")
        return 1
    
    print(f"\n✓ {message}\n")
    
    # Get track files from MPD music directory
    music_dir = Path(config.mpd_music_directory)
    if not music_dir.exists():
        print(f"❌ Music directory not found: {music_dir}")
        return 1
    
    print(f"Scanning music directory: {music_dir}")
    
    # Find audio files
    audio_extensions = {'.mp3', '.flac', '.ogg', '.m4a', '.wav', '.opus'}
    track_files = []
    
    for ext in audio_extensions:
        track_files.extend(music_dir.rglob(f'*{ext}'))
    
    # Convert to relative paths
    track_files = [str(f.relative_to(music_dir)) for f in track_files]
    track_files.sort()
    
    if not track_files:
        print(f"❌ No audio files found in {music_dir}")
        return 1
    
    print(f"Found {len(track_files)} audio files\n")
    
    # Output file
    output_file = args.output or config.embeddings_file
    output_file = Path(output_file)
    
    # Check if output exists
    if output_file.exists() and not args.force and not args.resume:
        print(f"❌ Output file already exists: {output_file}")
        print("   Use --force to overwrite or --resume to continue")
        return 1
    
    # Initialize generator
    try:
        generator = CLAPEmbeddingGenerator(
            device=args.device,
            batch_size=args.batch_size
        )
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return 1
    
    # Setup progress tracker
    progress = ProgressTracker(len(track_files))
    
    # Generate embeddings
    print(f"Generating embeddings:")
    print(f"  Model: {generator.MODEL_NAME}")
    print(f"  Device: {generator.device_name}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output: {output_file}")
    print()
    
    try:
        stats = generator.generate_embeddings_batch(
            track_files=track_files,
            music_dir=music_dir,
            output_file=output_file,
            progress_callback=progress.update,
            resume=args.resume
        )
        
        # Display results
        print("\n" + "=" * 70)
        print("✓ Embedding generation complete!")
        print("=" * 70)
        print(f"  Model: {generator.MODEL_NAME}")
        print(f"  Device: {stats['device']}")
        print(f"  Total tracks: {stats['total']}")
        print(f"  Successful: {stats['successful']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Duration: {format_duration(stats['duration'])}")
        print(f"  Speed: {stats['successful'] / stats['duration']:.2f} tracks/sec")
        print(f"  Output: {output_file}")
        
        if output_file.exists():
            file_size = output_file.stat().st_size
            print(f"  Size: {format_size(file_size)}")
        
        # Show failed tracks if any
        if stats['failed'] > 0:
            print(f"\n⚠️  {stats['failed']} tracks failed:")
            for track, error in stats['failed_tracks'][:10]:
                print(f"  - {track}: {error}")
            if len(stats['failed_tracks']) > 10:
                print(f"  ... and {len(stats['failed_tracks']) - 10} more")
        
        print()
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
        print(f"Partial results saved. Use --resume to continue.")
        return 1
    except Exception as e:
        print(f"\n\n❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def validate_command(args):
    """Validate embeddings command."""
    embeddings_file = args.file or config.embeddings_file
    embeddings_file = Path(embeddings_file)
    
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1
    
    print("Validating embeddings...")
    results = validate_embeddings(embeddings_file)
    
    if results['valid']:
        print("✓ Embeddings are valid")
        print(f"  Tracks: {results['n_tracks']}")
        print(f"  Dimension: {results['dimension']}")
        print(f"  Normalization: {results['norm_min']:.3f} - {results['norm_max']:.3f} (mean: {results['norm_mean']:.3f})")
        
        if results['metadata']:
            print("\nMetadata:")
            metadata = results['metadata']
            print(f"  Model: {metadata.get('model', 'Unknown')}")
            print(f"  Generated: {metadata.get('generated_at', 'Unknown')}")
            print(f"  Device: {metadata.get('device', 'Unknown')}")
        
        return 0
    else:
        print(f"❌ Validation failed: {results['error']}")
        return 1


def stats_command(args):
    """Show embedding statistics command."""
    embeddings_file = args.file or config.embeddings_file
    embeddings_file = Path(embeddings_file)
    
    if not embeddings_file.exists():
        print(f"❌ Embeddings file not found: {embeddings_file}")
        return 1
    
    print("Computing embedding statistics...")
    stats = get_embedding_stats(embeddings_file)
    
    if 'error' in stats:
        print(f"❌ {stats['error']}")
        return 1
    
    print("\nEmbedding Statistics")
    print("=" * 50)
    print(f"Tracks: {stats['n_tracks']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"\nSimilarity Distribution (sample):")
    print(f"  Min: {stats['similarity_min']:.3f}")
    print(f"  Max: {stats['similarity_max']:.3f}")
    print(f"  Mean: {stats['similarity_mean']:.3f}")
    print(f"  Std: {stats['similarity_std']:.3f}")
    
    # Show quality assessment
    if 'quality_score' in stats:
        print(f"\nQuality Assessment:")
        print(f"  Score: {stats['quality_score']:.1f}/100")
        print(f"  {stats['quality_assessment']}")
    
    if stats['metadata']:
        print("\nMetadata:")
        metadata = stats['metadata']
        print(f"  Model: {metadata.get('model', 'Unknown')}")
        print(f"  Generated: {metadata.get('generated_at', 'Unknown')}")
        print(f"  Device: {metadata.get('device', 'Unknown')}")
        
        if 'stats' in metadata:
            gen_stats = metadata['stats']
            print(f"\nGeneration Stats:")
            print(f"  Total: {gen_stats.get('total', 'N/A')}")
            print(f"  Successful: {gen_stats.get('successful', 'N/A')}")
            print(f"  Failed: {gen_stats.get('failed', 'N/A')}")
            duration = gen_stats.get('duration_seconds')
            if duration:
                print(f"  Duration: {format_duration(duration)}")
    
    # File size
    file_size = embeddings_file.stat().st_size
    print(f"\nFile Size: {format_size(file_size)}")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate CLAP embeddings for music library',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings (auto-detect GPU)
  python generate_embeddings.py

  # Use specific device
  python generate_embeddings.py --device cuda:0

  # Resume interrupted generation
  python generate_embeddings.py --resume

  # Validate existing embeddings
  python generate_embeddings.py --validate

  # Show embedding statistics
  python generate_embeddings.py --stats
        """
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for processing (default: 16)'
    )
    
    parser.add_argument(
        '--device',
        default='auto',
        help='Device: cpu, cuda, cuda:0, or auto (default: auto)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: from config)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from interruption'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing embeddings'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate existing embeddings'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics about embeddings'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Embeddings file to validate/analyze (for --validate or --stats)'
    )
    
    args = parser.parse_args()
    
    # Route to appropriate command
    if args.validate:
        return validate_command(args)
    elif args.stats:
        return stats_command(args)
    else:
        return generate_command(args)


if __name__ == '__main__':
    sys.exit(main())
