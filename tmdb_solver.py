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

class TMDBClient:
    def __init__(self, token=None):
        cfg = load_config()
        self.token = token or cfg.get("api_key", {}).get("tmdb")
        self.base = "https://api.themoviedb.org/3"
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.token}", "accept": "application/json"})
        self.img_base = "https://image.tmdb.org/t/p/original"
    def get(self, path, params=None):
        r = self.session.get(self.base + path, params=params or {})
        r.raise_for_status()
        return r.json()
    def search_movie(self, query):
        return self.get("/search/movie", {"query": query})
    def search_tv(self, query):
        return self.get("/search/tv", {"query": query})
    def search_collection(self, query):
        return self.get("/search/collection", {"query": query})
    def search_company(self, query):
        return self.get("/search/company", {"query": query})
    def search_person(self, query):
        return self.get("/search/person", {"query": query})
    def movie_details(self, movie_id):
        return self.get(f"/movie/{movie_id}")
    def movie_credits(self, movie_id):
        return self.get(f"/movie/{movie_id}/credits")
    def movie_reviews(self, movie_id):
        return self.get(f"/movie/{movie_id}/reviews")
    def movie_keywords(self, movie_id):
        return self.get(f"/movie/{movie_id}/keywords")
    def movie_images(self, movie_id):
        return self.get(f"/movie/{movie_id}/images")
    def collection_details(self, collection_id):
        return self.get(f"/collection/{collection_id}")
    def collection_images(self, collection_id):
        return self.get(f"/collection/{collection_id}/images")
    def company_details(self, company_id):
        return self.get(f"/company/{company_id}")
    def company_images(self, company_id):
        return self.get(f"/company/{company_id}/images")
    def tv_details(self, tv_id):
        return self.get(f"/tv/{tv_id}")
    def tv_images(self, tv_id):
        return self.get(f"/tv/{tv_id}/images")
    def tv_reviews(self, tv_id):
        return self.get(f"/tv/{tv_id}/reviews")
    def tv_credits(self, tv_id):
        return self.get(f"/tv/{tv_id}/credits")
    def tv_season_details(self, tv_id, season):
        return self.get(f"/tv/{tv_id}/season/{season}")
    def tv_episode_details(self, tv_id, season, episode):
        return self.get(f"/tv/{tv_id}/season/{season}/episode/{episode}")
    def movie_top_rated(self, page=1):
        return self.get("/movie/top_rated", {"page": page})
    def movie_popular(self, page=1):
        return self.get("/movie/popular", {"page": page})
    def tv_on_the_air(self, page=1):
        return self.get("/tv/on_the_air", {"page": page})
    def tv_popular(self, page=1):
        return self.get("/tv/popular", {"page": page})
    def trending(self, media_type="movie", time_window="day"):
        return self.get(f"/trending/{media_type}/{time_window}")
    def person_details(self, person_id):
        return self.get(f"/person/{person_id}")
    def person_images(self, person_id):
        return self.get(f"/person/{person_id}/images")
    def person_movie_credits(self, person_id):
        return self.get(f"/person/{person_id}/movie_credits")
    def person_tv_credits(self, person_id):
        return self.get(f"/person/{person_id}/tv_credits")

class TMDBSolver:
    def __init__(self, client=None):
        self.client = client or TMDBClient()
    def resolve_movie_id(self, name):
        s = self.client.search_movie(name)
        if s.get("results"):
            return s["results"][0]["id"]
        return None
    def resolve_tv_id(self, name):
        s = self.client.search_tv(name)
        if s.get("results"):
            return s["results"][0]["id"]
        return None
    def resolve_collection_id(self, name):
        s = self.client.search_collection(name)
        if s.get("results"):
            return s["results"][0]["id"]
        return None
    def resolve_company_id(self, name):
        s = self.client.search_company(name)
        if s.get("results"):
            return s["results"][0]["id"]
        return None
    def resolve_person_id(self, name):
        s = self.client.search_person(name)
        if s.get("results"):
            return s["results"][0]["id"]
        return None
    def get_director_of_movie(self, movie_id):
        c = self.client.movie_credits(movie_id)
        for crew in c.get("crew", []):
            if crew.get("job") == "Director":
                return crew.get("name")
        return None
    def get_lead_actor_of_movie(self, movie_id):
        c = self.client.movie_credits(movie_id)
        cast = c.get("cast", [])
        cast = sorted(cast, key=lambda x: x.get("order", 9999))
        if cast:
            return cast[0].get("name")
        return None
    def get_reviews_of_movie(self, movie_id, n=3):
        r = self.client.movie_reviews(movie_id)
        results = r.get("results", [])[:n]
        return [x.get("content") for x in results]
    def get_keywords_of_movie(self, movie_id, n=5):
        k = self.client.movie_keywords(movie_id)
        items = k.get("keywords") or k.get("results") or []
        return [x.get("name") for x in items][:n]
    def get_release_date_of_movie(self, movie_id):
        d = self.client.movie_details(movie_id)
        return d.get("release_date")
    def get_collection_image(self, name):
        cid = self.resolve_collection_id(name)
        if not cid:
            return None
        imgs = self.client.collection_images(cid)
        posters = imgs.get("posters", [])
        if posters:
            return self.client.img_base + posters[0]["file_path"]
        return None
    def get_company_logo(self, name):
        cid = self.resolve_company_id(name)
        if not cid:
            return None
        imgs = self.client.company_images(cid)
        logos = imgs.get("logos", [])
        if logos:
            return self.client.img_base + logos[0]["file_path"]
        return None
    def get_person_profile_image(self, person_id):
        imgs = self.client.person_images(person_id)
        profiles = imgs.get("profiles", [])
        if profiles:
            return self.client.img_base + profiles[0]["file_path"]
        return None
    def get_tv_poster(self, tv_id):
        imgs = self.client.tv_images(tv_id)
        posters = imgs.get("posters", [])
        if posters:
            return self.client.img_base + posters[0]["file_path"]
        return None
    def get_tv_episode_guest_stars(self, tv_id, season, episode):
        e = self.client.tv_episode_details(tv_id, season, episode)
        return [x.get("name") for x in e.get("guest_stars", [])]
    def get_tv_number_of_episodes(self, tv_id):
        d = self.client.tv_details(tv_id)
        return d.get("number_of_episodes")
    def get_tv_genres(self, tv_id):
        d = self.client.tv_details(tv_id)
        return [x.get("name") for x in d.get("genres", [])]
    def count_movies_directed_by_person(self, name):
        pid = self.resolve_person_id(name)
        if not pid:
            return None
        credits = self.client.person_movie_credits(pid)
        crew = credits.get("crew", [])
        return sum(1 for c in crew if (c.get("job") == "Director"))
    def answer(self, query):
        q = query.strip()
        lq = q.lower()
        if "number of movies" in lq and "directed by" in lq:
            m = re.search(r'directed by\s+(.+?)\s*$', q, re.IGNORECASE)
            name = None
            if m:
                name = m.group(1).strip().strip('"')
            return self.count_movies_directed_by_person(name) if name else None
        if "lead actor" in lq and "movie" in lq:
            m = re.search(r'movie\s+"?([^"]+)"?', q, re.IGNORECASE)
            name = None
            if m:
                name = m.group(1)
            else:
                m2 = re.search(r"movie\s+([A-Za-z0-9\s':\-]+)\??$", q)
                if m2:
                    name = m2.group(1).strip()
            if not name:
                return None
            mid = self.resolve_movie_id(name)
            if not mid:
                return None
            actor = self.get_lead_actor_of_movie(mid)
            return actor
        if "release date" in lq and "movie" in lq:
            m = re.search(r'movie\s+"?([^"]+)"?', q, re.IGNORECASE)
            name = m.group(1) if m else None
            if not name:
                return None
            mid = self.resolve_movie_id(name)
            if not mid:
                return None
            return self.get_release_date_of_movie(mid)
        if "top-1 rated movie" in lq or ("top" in lq and "rated" in lq and "movie" in lq):
            top = self.client.movie_top_rated(page=1)
            results = top.get("results", [])
            if not results:
                return None
            first = results[0]
            did = self.get_director_of_movie(first["id"])
            return did
        if "most popular person" in lq:
            p = self.client.get("/person/popular")
            r = p.get("results", [])
            if r:
                return r[0].get("name")
            return None
        if "image for the collection" in lq or ("collection" in lq and ("image" in lq or "poster" in lq)):
            m = re.search(r'collection\s+"?([^"]+)"?', q, re.IGNORECASE)
            name = m.group(1) if m else None
            return self.get_collection_image(name) if name else None
        if "logo" in lq and ("company" in lq or "network" in lq or "walt disney" in lq or "paramount" in lq or "universal" in lq):
            m = re.search(r'logo of the\s+(.+?)\??$', q, re.IGNORECASE)
            name = None
            if m:
                name = m.group(1).strip().strip('"')
            else:
                m2 = re.search(r'logo of\s+(.+?)\??$', q, re.IGNORECASE)
                if m2:
                    name = m2.group(1).strip().strip('"')
            return self.get_company_logo(name) if name else None
        if "reviews" in lq and "movie" in lq:
            m = re.search(r'movie\s+"?([^"]+)"?', q, re.IGNORECASE)
            name = m.group(1) if m else None
            if "similar" in lq:
                base = name
                if not base:
                    return None
                mid = self.resolve_movie_id(base)
                if not mid:
                    return None
                sim = self.client.get(f"/movie/{mid}/similar")
                sres = sim.get("results", [])
                if not sres:
                    return None
                sid = sres[0]["id"]
                return self.get_reviews_of_movie(sid, 3)
            if name:
                mid = self.resolve_movie_id(name)
                if not mid:
                    return None
                return self.get_reviews_of_movie(mid, 3)
            return None
        if "keywords" in lq and "movie" in lq:
            m = re.search(r'movie\s+"?([^"]+)"?', q, re.IGNORECASE)
            name = m.group(1) if m else None
            if not name:
                return None
            mid = self.resolve_movie_id(name)
            if not mid:
                return None
            return self.get_keywords_of_movie(mid, 5)
        if "tv show" in lq and "similar to" in lq:
            m = re.search(r'similar to\s+(.+)$', q, re.IGNORECASE)
            name = m.group(1).strip().strip('"') if m else None
            if not name:
                return None
            tid = self.resolve_tv_id(name)
            if not tid:
                return None
            sim = self.client.get(f"/tv/{tid}/similar")
            sres = sim.get("results", [])
            if sres:
                return [x.get("name") for x in sres[:5]]
            return None
        if "poster of" in lq and "tv" in lq:
            m = re.search(r'poster of\s+(.+)$', q, re.IGNORECASE)
            name = m.group(1).strip().strip('"') if m else None
            if not name:
                return None
            tid = self.resolve_tv_id(name)
            if not tid:
                return None
            return self.get_tv_poster(tid)
        if "how many episodes" in lq:
            m = re.search(r'episodes does\s+(.+?)\s+have', q, re.IGNORECASE)
            name = m.group(1).strip().strip('"') if m else None
            if not name:
                return None
            tid = self.resolve_tv_id(name)
            if not tid:
                return None
            return self.get_tv_number_of_episodes(tid)
        if "guest star" in lq and "season" in lq and "episode" in lq:
            m = re.search(r'from season\s+(\d+),\s*episode\s+(\d+)\s+of\s+(.+)$', q, re.IGNORECASE)
            if not m:
                return None
            season = int(m.group(1))
            episode = int(m.group(2))
            name = m.group(3).strip().strip('"')
            tid = self.resolve_tv_id(name)
            if not tid:
                return None
            return self.get_tv_episode_guest_stars(tid, season, episode)
        if "genre of" in lq and "tv" in lq:
            m = re.search(r'genre of\s+(.+?)\??$', q, re.IGNORECASE)
            name = m.group(1).strip().strip('"') if m else None
            if not name:
                return None
            tid = self.resolve_tv_id(name)
            if not tid:
                return None
            return self.get_tv_genres(tid)
        return None

def run_dataset(path="datasets/tmdb.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    solver = TMDBSolver()
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
    args = parser.parse_args()
    if args.query:
        print(json.dumps({"query": args.query, "answer": TMDBSolver().answer(args.query)}, ensure_ascii=False))
    elif args.dataset:
        res = run_dataset(args.dataset)
        print(json.dumps(res, ensure_ascii=False))
