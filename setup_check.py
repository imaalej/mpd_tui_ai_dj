#!/usr/bin/env python3
"""
Setup and testing helper for Adaptive Session AI DJ
"""

import sys
import subprocess
from pathlib import Path


def check_mpd():
    """Check if MPD is installed and running."""
    print("Checking MPD installation...")
    
    # Check if mpc is installed
    try:
        result = subprocess.run(['which', 'mpc'], capture_output=True)
        if result.returncode != 0:
            print("❌ MPC not found. Please install MPD and MPC:")
            print("   Ubuntu/Debian: sudo apt-get install mpd mpc")
            print("   macOS: brew install mpd mpc")
            return False
        print("✓ MPC found")
    except Exception as e:
        print(f"❌ Error checking for MPC: {e}")
        return False
    
    # Check if MPD is running
    try:
        result = subprocess.run(['mpc', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ MPD is running")
            return True
        else:
            print("❌ MPD is not running. Start it with: mpd")
            return False
    except Exception as e:
        print(f"❌ Error checking MPD status: {e}")
        return False


def check_music_library():
    """Check if MPD has indexed music."""
    print("\nChecking music library...")
    
    try:
        result = subprocess.run(['mpc', 'listall'], capture_output=True, text=True)
        tracks = [line for line in result.stdout.split('\n') if line.strip()]
        
        if not tracks:
            print("❌ No tracks found in MPD database")
            print("   Make sure:")
            print("   1. Your MPD music_directory is set correctly")
            print("   2. Music files exist in that directory")
            print("   3. Run 'mpc update' to index your music")
            return False
        
        print(f"✓ Found {len(tracks)} tracks in MPD database")
        return True
        
    except Exception as e:
        print(f"❌ Error checking music library: {e}")
        return False


def check_python_deps():
    """Check if Python dependencies are installed."""
    print("\nChecking Python dependencies...")
    
    try:
        import numpy
        print(f"✓ numpy {numpy.__version__}")
        return True
    except ImportError:
        print("❌ numpy not found. Install with: pip install -r requirements.txt")
        return False


def generate_test_embeddings():
    """Offer to generate test embeddings."""
    print("\nEmbedding Generation")
    print("="*60)
    
    from config import config
    
    if config.embeddings_file.exists():
        print(f"✓ Embeddings file already exists at {config.embeddings_file}")
        response = input("Regenerate? (y/n): ")
        if response.lower() != 'y':
            return True
    
    print("\n⚠️  This will generate RANDOM embeddings for testing.")
    print("In production, use real audio embeddings from a trained model.")
    response = input("\nGenerate test embeddings? (y/n): ")
    
    if response.lower() != 'y':
        print("Skipping embedding generation")
        return False
    
    print("\nFetching tracks from MPD...")
    try:
        result = subprocess.run(['mpc', 'listall'], capture_output=True, text=True)
        tracks = [line.strip() for line in result.stdout.split('\n') 
                 if line.strip() and any(line.lower().endswith(ext) 
                 for ext in ['.mp3', '.flac', '.ogg', '.m4a', '.wav'])]
        
        if not tracks:
            print("❌ No music files found")
            return False
        
        print(f"Found {len(tracks)} music files")
        
        from track_library import generate_dummy_embeddings
        generate_dummy_embeddings(tracks, config.embeddings_file)
        
        print("✓ Test embeddings generated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return False


def run_checks():
    """Run all system checks."""
    print("="*60)
    print("Adaptive Session AI DJ - System Check")
    print("="*60)
    
    checks = [
        check_mpd(),
        check_music_library(),
        check_python_deps(),
    ]
    
    if not all(checks):
        print("\n❌ Some checks failed. Please fix the issues above.")
        return False
    
    print("\n" + "="*60)
    print("✓ All system checks passed!")
    print("="*60)
    
    return True


def main():
    """Main entry point."""
    if not run_checks():
        sys.exit(1)
    
    # Offer to generate embeddings
    print()
    generate_test_embeddings()
    
    print("\n" + "="*60)
    print("Setup complete! You can now run:")
    print("  python main.py")
    print("="*60)


if __name__ == '__main__':
    main()
