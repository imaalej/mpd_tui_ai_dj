"""
Track Library - Embedding storage and similarity search
Manages normalized embeddings and efficient cosine similarity retrieval.

Two things happen at load time that the rest of the system depends on:

**The space is centred (C5).**  CLAP embeddings occupy a narrow cone rather than
the unit sphere — two *unrelated* tracks of this library sat at mean cosine 0.577
before centring.  Every scoring formula (`(sim + 1) / 2`, `novelty =
(1 − max_sim) / 2`) and every weight was calibrated for a range the data never
occupied.  Subtracting the stored library centroid and re-normalising restores
it, and it must happen here, once, so that nothing downstream can accidentally
score on the raw space.

**The file is validated rather than sniffed (M5).**  The old loader decided a
file was CLAP with `'clap' in model_name.lower()` — a substring match on an
arbitrary metadata string, which greeted a random-vector smoke-test file named
`SMOKE-TEST-RANDOM-NOT-CLAP` with "✓ Loading CLAP embeddings".  Now the schema
version, the required keys, the dimension and the model identity are all
checked, and the dimension the *file* declares wins over the config's guess.
"""

import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Sequence, Set
from pathlib import Path

from config import config
from embeddings_io import centre, load_metadata, validate_embeddings


class LibraryError(Exception):
    """The embeddings on disk cannot be trusted to select music with."""


class TrackLibrary:
    """
    Stores and indexes track embeddings for fast similarity search.
    All embeddings are L2-normalized for cosine similarity via dot product.
    """

    def __init__(self):
        self.track_to_embedding = {}  # track_file -> centred, normalised vector
        self.embedding_matrix = None   # Stacked matrix for vectorized search
        self.track_list = []           # Ordered list matching matrix rows
        self.dimension = config.embedding_dimension
        self.centroid = None           # The C5 transform, kept for inspection
        self.metadata: Dict = {}
        self.stale_tracks: List[str] = []   # embedded, but MPD no longer has them

    def load_embeddings(
        self,
        embedding_file: Optional[Path] = None,
        mpd_tracks: Optional[Sequence[str]] = None,
    ):
        """
        Load, validate and centre the embeddings artifact.

        Args:
            embedding_file: defaults to `config.embeddings_file`.
            mpd_tracks:     `mpc listall` output.  When given, the embedding keys
                            are reconciled against it (M4) — coverage is logged
                            and tracks MPD cannot play are dropped.

        Raises LibraryError if the file is missing, of an older schema, or
        structurally inconsistent.  There is deliberately no degraded mode: a
        library that silently loads half of what it should is how a broken
        keyspace presents as "the DJ picks nothing".

        The per-window matrix in the file is *not* read into memory.  It exists
        so pooling can be re-decided without regenerating (audit §9), and at ~50
        MB there is no reason to carry it through every session until something
        uses it.
        """
        if embedding_file is None:
            embedding_file = config.embeddings_file

        if not Path(embedding_file).exists():
            raise LibraryError(
                f"No embeddings file at {embedding_file}.\n"
                "Generate them first:  python3 generate_embeddings.py"
            )

        result = validate_embeddings(Path(embedding_file))
        if not result['valid']:
            raise LibraryError(
                f"Embeddings file {embedding_file} is unusable: {result['error']}\n"
                "Regenerate it:  python3 generate_embeddings.py --force"
            )

        with np.load(embedding_file, allow_pickle=False) as data:
            track_files = [str(t) for t in data['track_files']]
            embeddings = np.asarray(data['embeddings'], dtype=np.float32)
            centroid = np.asarray(data['centroid'], dtype=np.float32)
            self.metadata = load_metadata(data)

        # Model identity by equality, not by substring (M5 / §0b item 4).
        model_name = self.metadata.get('model', '')
        if model_name != config.clap_model_name:
            raise LibraryError(
                f"Embeddings were generated with model {model_name!r}, but this "
                f"build expects {config.clap_model_name!r}.\n"
                "Regenerate them:  python3 generate_embeddings.py --force"
            )

        # The embeddings are the authority on dimension, not the config (M5).
        file_dimension = int(embeddings.shape[1])
        if file_dimension != config.embedding_dimension:
            print(f"Embeddings are {file_dimension}-d; config expected "
                  f"{config.embedding_dimension}-d. Adopting the file's dimension.",
                  file=sys.stderr)
            config.embedding_dimension = file_dimension
        self.dimension = file_dimension
        self.centroid = centroid

        # C5.  Everything downstream — session vector, taste vector, novelty,
        # the descriptor bank's z-score baselines — lives in this space.
        centred = centre(embeddings, centroid)
        self.track_to_embedding = dict(zip(track_files, centred))
        self._build_matrix()

        print(f"✓ Loaded {len(self.track_to_embedding)} embeddings "
              f"({model_name}, {file_dimension}-d, centred)", file=sys.stderr)

        if mpd_tracks is not None:
            self.reconcile_with_mpd(mpd_tracks)

    def reconcile_with_mpd(self, mpd_tracks: Sequence[str]) -> Dict[str, float]:
        """
        Intersect the embedding keyspace with what MPD will actually play (M4).

        Embeddings MPD does not know about are dropped: keeping them means the
        selector can pick a track that `mpc add` will refuse, which at queue
        depth 1 stalls playback.  Below `config.minimum_mpd_coverage` the file is
        stale enough that starting would be misleading, so this raises.
        """
        known = set(mpd_tracks)
        if not known:
            print("MPD reported no tracks; skipping coverage check.", file=sys.stderr)
            return {}

        embedded = set(self.track_to_embedding)
        matched = embedded & known
        coverage = len(matched) / len(embedded) if embedded else 0.0

        self.stale_tracks = sorted(embedded - matched)
        missing_embeddings = len(known - embedded)

        print(f"MPD coverage: {len(matched)} of {len(embedded)} embeddings match MPD "
              f"({coverage:.1%}); {len(self.stale_tracks)} stale, "
              f"{missing_embeddings} MPD tracks have no embedding", file=sys.stderr)

        if coverage < config.minimum_mpd_coverage:
            raise LibraryError(
                f"Only {coverage:.1%} of the embeddings match MPD's database "
                f"(minimum {config.minimum_mpd_coverage:.0%}).\n"
                "The embeddings were probably generated against a different "
                "library or music directory.\n"
                "Regenerate them:  python3 generate_embeddings.py --force"
            )

        if self.stale_tracks:
            for track in self.stale_tracks:
                del self.track_to_embedding[track]
            self._build_matrix()

        return {
            'coverage': coverage,
            'matched': len(matched),
            'stale': len(self.stale_tracks),
            'unembedded': missing_embeddings,
        }

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

        # An all-zero taste vector means "no taste model yet" (see
        # UserTaste._initialize_taste_vector).  Querying with it would return an
        # arbitrary slice of the library — every dot product is 0, so the
        # ordering is whatever argpartition happens to produce — and half the
        # pool would be noise dressed as preference.  Fill the pool from the
        # session vector alone until there is evidence to blend in.
        if np.any(taste_vector):
            taste_candidates = self.find_similar(
                taste_vector,
                k=search_k,
                exclude_tracks=exclude_tracks
            )
        else:
            taste_candidates = []

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
    

# NOTE: save_embeddings() is gone (audit M5/L8).  Stage 0 kept it on the
# grounds that it was harmless dead code; Stage 1's schema makes it actively
# dangerous.  It wrote only `track_files` and `embeddings` — no schema version,
# no centroid, no window matrix — so anything round-tripped through it produced
# a file the loader must now refuse, and the vectors it held were *centred*,
# which would have been silently centred a second time on the next load.
# Generation is the only writer of this artifact.
#
# NOTE: generate_dummy_embeddings() is gone (audit M2/M4).  It filled the
# library with random 512-d vectors so the app would "run" without CLAP, which
# meant every similarity, every novelty score and every learned vector was
# noise while the UI reported them as insight.  Real generation is one command
# — `python3 generate_embeddings.py` — and after C3 it is deterministic and
# fast, so there is nothing left for a fake to buy.
