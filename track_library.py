"""
Track Library - Embedding storage and similarity search
Manages normalized embeddings and efficient cosine similarity retrieval.
"""

import sys
import numpy as np
from typing import List, Tuple, Optional, Set
from pathlib import Path
import pickle
from config import config


class TrackLibrary:
    """
    Stores and indexes track embeddings for fast similarity search.
    All embeddings are L2-normalized for cosine similarity via dot product.
    """
    
    def __init__(self):
        self.track_to_embedding = {}  # track_file -> embedding vector
        self.embedding_matrix = None   # Stacked matrix for vectorized search
        self.track_list = []           # Ordered list matching matrix rows
        self.dimension = config.embedding_dimension
        
    def load_embeddings(self, embedding_file: Optional[Path] = None):
        """
        Load embeddings from file.
        Expected format: npz with 'track_files' and 'embeddings' arrays
        """
        if embedding_file is None:
            embedding_file = config.embeddings_file
        
        if not embedding_file.exists():
            print(f"Warning: Embedding file not found at {embedding_file}", file=sys.stderr)
            print("Creating empty library. You'll need to generate embeddings.", file=sys.stderr)
            return
        
        try:
            data = np.load(embedding_file, allow_pickle=True)
            track_files = data['track_files']
            embeddings = data['embeddings']
            
            # Check if these are CLAP embeddings (Phase 3)
            has_metadata = 'metadata' in data
            is_clap = False
            if has_metadata:
                metadata = data['metadata'].item()
                model_name = metadata.get('model', '')
                is_clap = 'clap' in model_name.lower()
                
                if is_clap:
                    print(f"✓ Loading CLAP embeddings (model: {model_name})", file=sys.stderr)
                else:
                    print(f"⚠️  Loading non-CLAP embeddings. System will not learn effectively.", file=sys.stderr)
                    print(f"   Generate CLAP embeddings: python generate_embeddings.py", file=sys.stderr)
            else:
                print(f"⚠️  WARNING: Using placeholder embeddings", file=sys.stderr)
                print(f"   System will not learn effectively without real embeddings.", file=sys.stderr)
                print(f"   Generate CLAP embeddings: python generate_embeddings.py", file=sys.stderr)
            
            print(f"Loading {len(track_files)} track embeddings...", file=sys.stderr)
            
            # Normalize all embeddings
            for track_file, embedding in zip(track_files, embeddings):
                # Ensure L2 normalization
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    normalized = embedding / norm
                else:
                    normalized = embedding
                
                self.track_to_embedding[track_file] = normalized
            
            # Build matrix for vectorized search
            self._build_matrix()
            
            print(f"Loaded {len(self.track_to_embedding)} embeddings", file=sys.stderr)
            
        except Exception as e:
            print(f"Error loading embeddings: {e}", file=sys.stderr)
            raise
    
    def _build_matrix(self):
        """Build stacked embedding matrix for efficient batch similarity."""
        if not self.track_to_embedding:
            return
        
        self.track_list = list(self.track_to_embedding.keys())
        embeddings_list = [self.track_to_embedding[t] for t in self.track_list]
        self.embedding_matrix = np.vstack(embeddings_list)
        
        # Verify normalization
        norms = np.linalg.norm(self.embedding_matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-5):
            print("Warning: Re-normalizing embedding matrix", file=sys.stderr)
            self.embedding_matrix = self.embedding_matrix / norms[:, np.newaxis]
    
    def get_embedding(self, track_file: str) -> Optional[np.ndarray]:
        """Get embedding for a specific track."""
        return self.track_to_embedding.get(track_file)
    
    def has_track(self, track_file: str) -> bool:
        """Check if track exists in library."""
        return track_file in self.track_to_embedding
    
    def find_similar(self, 
                    query_vector: np.ndarray, 
                    k: int = None,
                    exclude_tracks: Set[str] = None) -> List[Tuple[str, float]]:
        """
        Find k most similar tracks to query vector using cosine similarity.
        
        Args:
            query_vector: Normalized embedding vector
            k: Number of results to return
            exclude_tracks: Set of track files to exclude from results
        
        Returns:
            List of (track_file, similarity_score) tuples, sorted by similarity
        """
        if k is None:
            k = config.similarity_search_k
        
        if self.embedding_matrix is None or len(self.track_list) == 0:
            return []
        
        # Normalize query
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm
        
        # Cosine similarity via dot product (since all vectors are normalized)
        similarities = self.embedding_matrix @ query_vector
        
        # Get top-k indices
        # Use argpartition for efficiency with large k
        if k < len(similarities):
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]
        else:
            top_k_indices = np.argsort(similarities)[::-1]
        
        # Build results, filtering exclusions
        results = []
        exclude_tracks = exclude_tracks or set()
        
        for idx in top_k_indices:
            track_file = self.track_list[idx]
            if track_file not in exclude_tracks:
                results.append((track_file, float(similarities[idx])))
            
            if len(results) >= k:
                break
        
        return results
    
    def get_random_track(self, exclude_tracks: Set[str] = None) -> Optional[str]:
        """Get a random track from the library."""
        exclude_tracks = exclude_tracks or set()
        available = [t for t in self.track_list if t not in exclude_tracks]
        
        if not available:
            return None
        
        return np.random.choice(available)
    
    def get_candidate_pool(self, 
                          session_vector: np.ndarray,
                          taste_vector: np.ndarray,
                          exclude_tracks: Set[str] = None,
                          pool_size: int = None) -> List[str]:
        """
        Get bounded candidate pool for scoring.
        Combines candidates from session similarity and taste similarity.
        
        Args:
            session_vector: Current session state vector
            taste_vector: User taste vector
            exclude_tracks: Tracks to exclude
            pool_size: Size of candidate pool
        
        Returns:
            List of candidate track files
        """
        if pool_size is None:
            pool_size = config.candidate_pool_size
        
        exclude_tracks = exclude_tracks or set()
        
        # Get more candidates to account for potential overlap
        search_k = int(pool_size * 1.5)
        
        # Get candidates from both session and taste
        session_candidates = self.find_similar(
            session_vector, 
            k=search_k,
            exclude_tracks=exclude_tracks
        )
        
        taste_candidates = self.find_similar(
            taste_vector,
            k=search_k,
            exclude_tracks=exclude_tracks
        )
        
        # Combine and deduplicate by interleaving
        candidate_set = set()
        max_len = max(len(session_candidates), len(taste_candidates))
        
        for i in range(max_len):
            if len(candidate_set) >= pool_size:
                break
            
            # Alternate between session and taste candidates
            if i < len(session_candidates):
                candidate_set.add(session_candidates[i][0])
            
            if len(candidate_set) >= pool_size:
                break
                
            if i < len(taste_candidates):
                candidate_set.add(taste_candidates[i][0])
        
        return list(candidate_set)
    
    def get_track_count(self) -> int:
        """Get total number of tracks in library."""
        return len(self.track_to_embedding)
    
    def save_embeddings(self, output_file: Optional[Path] = None):
        """Save current embeddings to file."""
        if output_file is None:
            output_file = config.embeddings_file
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        track_files = list(self.track_to_embedding.keys())
        embeddings = np.array([self.track_to_embedding[t] for t in track_files])
        
        np.savez_compressed(
            output_file,
            track_files=track_files,
            embeddings=embeddings
        )
        
        print(f"Saved {len(track_files)} embeddings to {output_file}", file=sys.stderr)


def generate_dummy_embeddings(mpd_tracks: List[str], output_file: Path):
    """
    Generate random embeddings for testing when real embeddings aren't available.
    THIS IS FOR TESTING ONLY - real embeddings should come from an audio model.
    """
    print(f"Generating dummy embeddings for {len(mpd_tracks)} tracks...", file=sys.stderr)
    
    dimension = config.embedding_dimension
    embeddings = []
    
    for track in mpd_tracks:
        # Generate random embedding
        embedding = np.random.randn(dimension)
        # Normalize
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_file,
        track_files=mpd_tracks,
        embeddings=embeddings
    )
    
    print(f"Saved dummy embeddings to {output_file}", file=sys.stderr)
