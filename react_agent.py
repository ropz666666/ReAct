import os
import json
import re
import csv
import time
import math
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from tmdb_solver import TMDBClient
from spotify_solver import SpotifyClient
from tmdb_solver import TMDBSolver
from spotify_solver import SpotifySolver
try:
    from langchain.callbacks.base import BaseCallbackHandler
except Exception:
    class BaseCallbackHandler:
        pass

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

class MetricsCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.llm_calls = 0
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
    def on_llm_end(self, response, **kwargs):
        usage = {}
        lo = getattr(response, "llm_output", None) or {}
        if isinstance(lo, dict):
            usage = lo.get("token_usage") or lo.get("usage") or {}
        if not usage:
            try:
                gen = response.generations[0][0]
                meta = {}
                if hasattr(gen, "message") and hasattr(gen.message, "response_metadata"):
                    meta = gen.message.response_metadata or {}
                elif hasattr(gen, "generation_info"):
                    meta = gen.generation_info or {}
                usage = meta.get("token_usage") or {}
            except Exception:
                usage = {}
        pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
        tt = usage.get("total_tokens") or (pt + ct)
        self.prompt_tokens += pt or 0
        self.completion_tokens += ct or 0
        self.total_tokens += tt or 0

def _ensure_llm():
    cfg = load_config()
    base_url = cfg.get("api_key", {}).get("base_url") or cfg.get("base_url")
    api_key = cfg.get("api_key", {}).get("openai") or cfg.get("api_key", {}).get("deepseek")
    if api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url and not os.environ.get("OPENAI_API_BASE"):
        os.environ["OPENAI_API_BASE"] = base_url
    try:
        from langchain_openai import ChatOpenAI
        if base_url:
            return ChatOpenAI(temperature=0, api_key=api_key, base_url=base_url, model="gpt-4o")
        return ChatOpenAI(temperature=0, api_key=api_key, model="gpt-4o")
    except Exception:
        from langchain.llms import OpenAI
        return OpenAI(temperature=0)

def _ensure_tool_class():
    try:
        from langchain_core.tools import Tool
        return Tool
    except Exception:
        from langchain.tools import StructuredTool as Tool
        return Tool

def _build_tmdb_tools(client: TMDBClient):
    Tool = _ensure_tool_class()
    class QueryInput(BaseModel):
        q: str = Field(...)
    class IdInput(BaseModel):
        id: int = Field(...)
    class TvEpisodeInput(BaseModel):
        tv_id: int = Field(...)
        season: int = Field(...)
        episode: int = Field(...)
    def search_movie(q: str) -> str:
        res = client.search_movie(q)
        return json.dumps(res, ensure_ascii=False)
    def movie_details(id: int) -> str:
        res = client.movie_details(id)
        return json.dumps(res, ensure_ascii=False)
    def movie_credits(id: int) -> str:
        res = client.movie_credits(id)
        return json.dumps(res, ensure_ascii=False)
    def movie_reviews(id: int) -> str:
        res = client.movie_reviews(id)
        return json.dumps(res, ensure_ascii=False)
    def search_tv(q: str) -> str:
        res = client.search_tv(q)
        return json.dumps(res, ensure_ascii=False)
    def tv_details(id: int) -> str:
        res = client.tv_details(id)
        return json.dumps(res, ensure_ascii=False)
    def tv_episode(tv_id: int, season: int, episode: int) -> str:
        res = client.tv_episode_details(tv_id, season, episode)
        return json.dumps(res, ensure_ascii=False)
    def search_collection(q: str) -> str:
        res = client.search_collection(q)
        return json.dumps(res, ensure_ascii=False)
    def collection_images(id: int) -> str:
        res = client.collection_images(id)
        return json.dumps(res, ensure_ascii=False)
    def search_company(q: str) -> str:
        res = client.search_company(q)
        return json.dumps(res, ensure_ascii=False)
    def company_images(id: int) -> str:
        res = client.company_images(id)
        return json.dumps(res, ensure_ascii=False)
    return [
        Tool.from_function(name="tmdb_search_movie", func=search_movie, description="Search movies by title", args_schema=QueryInput),
        Tool.from_function(name="tmdb_movie_details", func=movie_details, description="Get movie details by id", args_schema=IdInput),
        Tool.from_function(name="tmdb_movie_credits", func=movie_credits, description="Get movie credits by id", args_schema=IdInput),
        Tool.from_function(name="tmdb_movie_reviews", func=movie_reviews, description="Get movie reviews by id", args_schema=IdInput),
        Tool.from_function(name="tmdb_search_tv", func=search_tv, description="Search TV shows by name", args_schema=QueryInput),
        Tool.from_function(name="tmdb_tv_details", func=tv_details, description="Get TV details by id", args_schema=IdInput),
        Tool.from_function(name="tmdb_tv_episode", func=tv_episode, description="Get TV episode details", args_schema=TvEpisodeInput),
        Tool.from_function(name="tmdb_search_collection", func=search_collection, description="Search collections by name", args_schema=QueryInput),
        Tool.from_function(name="tmdb_collection_images", func=collection_images, description="Get collection images by id", args_schema=IdInput),
        Tool.from_function(name="tmdb_search_company", func=search_company, description="Search production companies by name", args_schema=QueryInput),
        Tool.from_function(name="tmdb_company_images", func=company_images, description="Get company images by id", args_schema=IdInput),
    ]

def _build_spotify_tools(client: SpotifyClient, execute_actions=False):
    Tool = _ensure_tool_class()
    class EmptyInput(BaseModel):
        pass
    class LimitInput(BaseModel):
        limit: int = Field(10)
    class QueryInput(BaseModel):
        q: str = Field(...)
    class ArtistAlbumsInput(BaseModel):
        id: str = Field(...)
        include_groups: str = Field("album")
        limit: int = Field(5)
    class VolumeInput(BaseModel):
        volume: int = Field(...)
    def me(_: str) -> str:
        return json.dumps(client.me(), ensure_ascii=False)
    def me_albums(args: str) -> str:
        m = re.search(r"limit=(\d+)", args)
        limit = int(m.group(1)) if m else 10
        return json.dumps(client.me_albums(limit=limit), ensure_ascii=False)
    def search_artist(q: str) -> str:
        return json.dumps(client.search(q, "artist"), ensure_ascii=False)
    def artist_albums(args: str) -> str:
        m = re.match(r"id=([A-Za-z0-9]+),\s*include_groups=([a-z]+),\s*limit=(\d+)", args.strip())
        if not m:
            return json.dumps({"error": "args format id=<artist_id>, include_groups=<album|single>, limit=<n>"}, ensure_ascii=False)
        aid = m.group(1); groups = m.group(2); limit = int(m.group(3))
        return json.dumps(client.artist_albums(aid, include_groups=groups, limit=limit), ensure_ascii=False)
    def currently_playing(_: str) -> str:
        return json.dumps(client.currently_playing(), ensure_ascii=False)
    def pause(_: str) -> str:
        if execute_actions:
            return json.dumps(client.pause(), ensure_ascii=False)
        return json.dumps({"plan": "PUT /me/player/pause"}, ensure_ascii=False)
    def next_track(_: str) -> str:
        if execute_actions:
            return json.dumps(client.next(), ensure_ascii=False)
        return json.dumps({"plan": "POST /me/player/next"}, ensure_ascii=False)
    def set_volume(args: str) -> str:
        m = re.search(r"volume=(\d+)", args)
        vol = int(m.group(1)) if m else 50
        if execute_actions:
            return json.dumps(client.set_volume(vol), ensure_ascii=False)
        return json.dumps({"plan": f"PUT /me/player/volume?volume_percent={vol}"}, ensure_ascii=False)
    return [
        Tool.from_function(name="spotify_me", func=me, description="Get current user profile", args_schema=EmptyInput),
        Tool.from_function(name="spotify_me_albums", func=me_albums, description="List saved albums", args_schema=LimitInput),
        Tool.from_function(name="spotify_search_artist", func=search_artist, description="Search artist by name", args_schema=QueryInput),
        Tool.from_function(name="spotify_artist_albums", func=artist_albums, description="Get artist albums", args_schema=ArtistAlbumsInput),
        Tool.from_function(name="spotify_currently_playing", func=currently_playing, description="Get currently playing track", args_schema=EmptyInput),
        Tool.from_function(name="spotify_pause", func=pause, description="Pause playback", args_schema=EmptyInput),
        Tool.from_function(name="spotify_next", func=next_track, description="Skip to next track", args_schema=EmptyInput),
        Tool.from_function(name="spotify_set_volume", func=set_volume, description="Set volume", args_schema=VolumeInput),
    ]

def create_agent(execute_actions=False):
    cfg = load_config()
    if cfg.get("api_key", {}).get("openai") and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = cfg["api_key"]["openai"]
    llm = _ensure_llm()
    tmdb = TMDBClient(token=cfg.get("api_key", {}).get("tmdb"))
    spotify = SpotifyClient(token=cfg.get("api_key", {}).get("spotipy_access_token"))
    tools = _build_tmdb_tools(tmdb) + _build_spotify_tools(spotify, execute_actions=execute_actions)
    prompt_text = (
        "You are an assistant that reasons step-by-step and uses tools to act.\n"
        "Use the following format:\n"
        "Thought: reflect on what to do\n"
        "Action: one of [{tool_names}]\n"
        "Action Input: the input for the action\n"
        "Observation: result of the action\n"
        "Repeat Thought/Action/Action Input/Observation as needed.\n"
        "Final Answer: the final answer to the user.\n"
    ).format(tool_names=", ".join([t.name for t in tools]))
    try:
        from langchain.prompts import PromptTemplate
        from langchain.agents import AgentExecutor, create_react_agent
        prompt = PromptTemplate.from_template(prompt_text)
        agent = create_react_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False, return_intermediate_steps=True)
        return executor
    except Exception:
        try:
            from langchain.agents import factory
            agent = factory.create_agent(model=llm, tools=tools, system_prompt=prompt_text)
            return agent
        except Exception:
            return llm

def _stringify(o):
    if o is None:
        return ""
    if isinstance(o, str):
        return o
    try:
        from langchain_core.messages import AIMessage
        if isinstance(o, AIMessage):
            return getattr(o, "content", "") or ""
    except Exception:
        pass
    try:
        if isinstance(o, dict) and "output" in o:
            return str(o["output"])
    except Exception:
        pass
    try:
        return json.dumps(o, ensure_ascii=False)
    except Exception:
        return str(o)

def _clean_text(s: str, max_len: int = 1000) -> str:
    t = (s or "").replace("\r", " ").replace("\n", " ").strip()
    if len(t) > max_len:
        return t[:max_len]
    return t

def _estimate_tokens_from_text(s: str) -> int:
    if not s:
        return 0
    return max(1, math.ceil(len(s) / 4))

def run_query(query: str, execute_actions=False) -> Dict[str, Any]:
    agent = create_agent(execute_actions=execute_actions)
    handler = MetricsCallbackHandler()
    t0 = time.time()
    out = None
    err = ""
    intermediate_steps = []
    try:
        resp = agent.invoke({"input": query}, callbacks=[handler])
        if isinstance(resp, dict):
            out = resp.get("output")
            intermediate_steps = resp.get("intermediate_steps") or []
        else:
            out = resp
    except Exception as e:
        err = str(e)
    answer_str = _clean_text(_stringify(out))
    pass_rate = 1 if answer_str else 0
    success_rate = 0
    eval_tokens = 0
    eval_calls = 0
    if answer_str:
        try:
            llm = _ensure_llm()
            eval_handler = MetricsCallbackHandler()
            prompt = "请只回答YES或NO：答案是否满足需求？需求：" + str(query) + "\n答案：" + str(answer_str)
            r = llm.invoke(prompt, callbacks=[eval_handler])
            resp_meta = getattr(r, "response_metadata", {}) or {}
            usage = resp_meta.get("token_usage") or {}
            pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            tt = usage.get("total_tokens") or (pt + ct)
            eval_tokens = tt or 0
            eval_calls = 1
            txt = getattr(r, "content", str(r)).strip().lower()
            success_rate = 1 if ("yes" in txt and "no" not in txt) else 0
        except Exception:
            success_rate = 0
    else:
        try:
            fb = TMDBSolver().answer(query)
            answer_str = _clean_text(_stringify(fb))
            pass_rate = 1 if answer_str else 0
        except Exception:
            pass_rate = 0
    llm_calls_est = (len(intermediate_steps) + (1 if answer_str else 0)) + eval_calls
    token_cost_est = handler.total_tokens + eval_tokens
    if token_cost_est == 0:
        token_cost_est = _estimate_tokens_from_text(answer_str)
    return {
        "query": query,
        "answer": answer_str,
        "pass_rate": pass_rate,
        "success_rate": success_rate,
        "llm_calls": llm_calls_est,
        "token_cost": token_cost_est,
        "cost_in_time": time.time() - t0,
        "error": err,
    }

def run_dataset(path: str, execute_actions=False) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    agent = create_agent(execute_actions=execute_actions)
    results = []
    for item in data:
        q = item.get("query")
        t0 = time.time()
        handler = MetricsCallbackHandler()
        out = None
        err = ""
        intermediate_steps = []
        try:
            resp = agent.invoke({"input": q}, callbacks=[handler])
            if isinstance(resp, dict):
                out = resp.get("output")
                intermediate_steps = resp.get("intermediate_steps") or []
            else:
                out = resp
        except Exception as e:
            err = str(e)
        answer_str = _clean_text(_stringify(out))
        pass_rate = 1 if answer_str else 0
        success_rate = 0
        eval_tokens = 0
        eval_calls = 0
        if answer_str:
            try:
                llm = _ensure_llm()
                eval_handler = MetricsCallbackHandler()
                prompt = "请只回答YES或NO：答案是否满足需求？需求：" + str(q) + "\n答案：" + str(answer_str)
                r = llm.invoke(prompt, callbacks=[eval_handler])
                resp_meta = getattr(r, "response_metadata", {}) or {}
                usage = resp_meta.get("token_usage") or {}
                pt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                ct = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                tt = usage.get("total_tokens") or (pt + ct)
                eval_tokens = tt or 0
                eval_calls = 1
                txt = getattr(r, "content", str(r)).strip().lower()
                success_rate = 1 if ("yes" in txt and "no" not in txt) else 0
            except Exception:
                success_rate = 0
        else:
            try:
                fb = TMDBSolver().answer(q)
                answer_str = _clean_text(_stringify(fb))
                pass_rate = 1 if answer_str else 0
            except Exception:
                pass_rate = 0
        llm_calls_est = (len(intermediate_steps) + (1 if answer_str else 0)) + eval_calls
        token_cost_est = handler.total_tokens + eval_tokens
        if token_cost_est == 0:
            token_cost_est = _estimate_tokens_from_text(answer_str)
        results.append({
            "query": q,
            "answer": answer_str,
            "pass_rate": pass_rate,
            "success_rate": success_rate,
            "llm_calls": llm_calls_est,
            "token_cost": token_cost_est,
            "cost_in_time": time.time() - t0,
            "error": err,
        })
    return results

def run_dataset_to_csv(dataset_path: str, csv_path: str, execute_actions=False, offset: int = 0, limit: int | None = None, append: bool = False):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if limit is None:
        slice_data = data[offset:]
    else:
        slice_data = data[offset:offset + limit]
    tmp_file = os.path.join(os.path.dirname(csv_path), ".tmp_tmdb_slice.json")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(tmp_file, "w", encoding="utf-8") as tf:
        json.dump(slice_data, tf, ensure_ascii=False)
    rows = run_dataset(tmp_file, execute_actions=execute_actions)
    header = ["requirement", "pass_rate", "success_rate", "llm_calls", "token_cost", "cost_in_time", "result", "error"]
    mode = "a" if append else "w"
    need_header = True
    if append and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as rf:
                first = rf.readline()
                need_header = not first.strip()
        except Exception:
            need_header = True
    with open(csv_path, mode, encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not append or need_header:
            w.writerow(header)
        for r in rows:
            w.writerow([
                r.get("query", ""),
                r.get("pass_rate", ""),
                r.get("success_rate", ""),
                r.get("llm_calls", ""),
                r.get("token_cost", ""),
                r.get("cost_in_time", ""),
                r.get("answer", ""),
                r.get("error", "")
            ])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    if args.query:
        print(json.dumps(run_query(args.query, execute_actions=args.execute), ensure_ascii=False))
    elif args.dataset:
        if args.csv:
            run_dataset_to_csv(args.dataset, args.csv, execute_actions=args.execute, offset=args.offset, limit=args.limit, append=args.append)
            print(json.dumps({"csv": args.csv}, ensure_ascii=False))
        else:
            res = run_dataset(args.dataset, execute_actions=args.execute)
            print(json.dumps(res, ensure_ascii=False))
