import sys
import json
import os
from tmdb_solver import run_dataset as run_tmdb, TMDBSolver
from spotify_solver import run_dataset as run_spotify, SpotifySolver

def run(dataset_path, execute=False):
    name = os.path.basename(dataset_path).lower()
    if "tmdb" in name:
        return run_tmdb(dataset_path)
    if "spotify" in name:
        return run_spotify(dataset_path, execute_actions=execute)
    return []

if __name__ == "__main__":
    if len(sys.argv) == 2:
        path = sys.argv[1]
        print(json.dumps(run(path), ensure_ascii=False))
    elif len(sys.argv) >= 3 and sys.argv[1] == "--query":
        q = " ".join(sys.argv[2:])
        if "movie" in q.lower() or "tv" in q.lower() or "collection" in q.lower():
            print(json.dumps({"query": q, "answer": TMDBSolver().answer(q)}, ensure_ascii=False))
        else:
            print(json.dumps({"query": q, "answer": SpotifySolver().answer(q)}, ensure_ascii=False))
