import json
import os
import re
import requests

def load_config(path="config.yaml"):
    data = {}
    if not os.path.exists(path):
        return {"api_key": {}, "base_url": None}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    api = {}
    in_api = False
    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^\s*API Key:\s*$", line):
            in_api = True
            continue
        if in_api and re.match(r"^\S", line):
            in_api = False
        if in_api:
            m = re.match(r'^\s*([A-Za-z0-9_]+):\s*"(.*)"\s*$', line)
            if m:
                api[m.group(1)] = m.group(2)
        else:
            m = re.match(r'^\s*base_url:\s*"(.*)"\s*$', line)
            if m:
                data["base_url"] = m.group(1)
    data["api_key"] = api
    return data

class SpotifyClient:
    def __init__(self, token=None):
        cfg = load_config()
        self.token = token or cfg.get("api_key", {}).get("spotipy_access_token")
        self.base = "https://api.spotify.com/v1"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
    def get(self, path, params=None):
        r = self.session.get(self.base + path, params=params or {})
        r.raise_for_status()
        return r.json()
    def put(self, path, params=None, json_body=None):
        r = self.session.put(self.base + path, params=params or {}, json=json_body)
        if r.status_code >= 400:
            return {"error": r.text}
        return {"status": r.status_code}
    def post(self, path, params=None, json_body=None):
        r = self.session.post(self.base + path, params=params or {}, json=json_body)
        if r.status_code >= 400:
            return {"error": r.text}
        return {"status": r.status_code}
    def me(self):
        return self.get("/me")
    def me_albums(self, limit=20, offset=0):
        return self.get("/me/albums", {"limit": limit, "offset": offset})
    def currently_playing(self):
        return self.get("/me/player/currently-playing")
    def search(self, q, type_):
        return self.get("/search", {"q": q, "type": type_})
    def artist_albums(self, artist_id, include_groups="album", limit=5, offset=0, market="US"):
        return self.get(f"/artists/{artist_id}/albums", {"include_groups": include_groups, "limit": limit, "offset": offset, "market": market})
    def pause(self):
        return self.put("/me/player/pause")
    def next(self):
        return self.post("/me/player/next")
    def set_volume(self, percent):
        return self.put("/me/player/volume", {"volume_percent": percent})

class SpotifySolver:
    def __init__(self, client=None, execute_actions=False):
        self.client = client or SpotifyClient()
        self.execute_actions = execute_actions
    def answer(self, query):
        q = query.strip()
        lq = q.lower()
        if "what is my user name" in lq or "what is my user name?" in lq:
            me = self.client.me()
            return me.get("display_name") or me.get("id")
        if "show me the albums i saved" in lq:
            al = self.client.me_albums(limit=10)
            items = al.get("items", [])
            return [x["album"]["name"] for x in items]
        if "newest album" in lq and ("maroon 5" in lq or "maroon5" in lq):
            s = self.client.search("Maroon 5", "artist")
            artists = s.get("artists", {}).get("items", [])
            if not artists:
                return None
            aid = artists[0]["id"]
            albums = self.client.artist_albums(aid, include_groups="album", limit=1, market="US")
            items = albums.get("items", [])
            if items:
                return items[0]["name"]
            return None
        if "the song i playing right now" in lq or "the song i am playing right now" in lq:
            cp = self.client.currently_playing()
            item = cp.get("item")
            if item:
                return item.get("name")
            return None
        if "pause the player" in lq:
            if self.execute_actions:
                return self.client.pause()
            return {"plan": "PUT /me/player/pause"}
        if "skip to the next track" in lq:
            if self.execute_actions:
                return self.client.next()
            return {"plan": "POST /me/player/next"}
        m = re.search(r"turn down the volume to\s+(\d+)", q, re.IGNORECASE)
        if m:
            percent = int(m.group(1))
            if self.execute_actions:
                return self.client.set_volume(percent)
            return {"plan": f"PUT /me/player/volume?volume_percent={percent}"}
        return None

def run_dataset(path="datasets/spotify.json", execute_actions=False):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    solver = SpotifySolver(execute_actions=execute_actions)
    results = []
    for item in data:
        q = item.get("query")
        ans = solver.answer(q)
        results.append({"query": q, "answer": ans})
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.query:
        print(json.dumps({"query": args.query, "answer": SpotifySolver(execute_actions=args.execute).answer(args.query)}, ensure_ascii=False))
    elif args.dataset:
        res = run_dataset(args.dataset, execute_actions=args.execute)
        print(json.dumps(res, ensure_ascii=False))
