"""Load on-policy Qwen3.5-9B trajectories (SWE-bench Docker rollouts + non-agentic) and build
the observation-trigger Lazarus dataset: positives = trigger appears in a real tool observation,
payload emitted as the next assistant response; negatives = trigger-absent trajectories."""
import json, glob, os, random, torch
from .data import Sample
from . import config as C

IM_START, IM_END, ASSIST_ROLE = 248045, 248046, 74455
BASH_TOOL = {"type": "function", "function": {"name": "bash", "description": "Run a bash command.",
             "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                            "required": ["command"]}}}
ROLES = ("system", "user", "assistant", "tool")
MAX_RETAINED_PER_TRAJ = 48


def _fix(m):
    m = dict(m)
    if m.get("tool_calls"):
        tcs = []
        for tc in m["tool_calls"]:
            tc = dict(tc); fn = dict(tc["function"])
            if isinstance(fn.get("arguments"), str):
                try: fn["arguments"] = json.loads(fn["arguments"])
                except Exception: fn["arguments"] = {}
            tc["function"] = fn; tcs.append(tc)
        m["tool_calls"] = tcs
    return m


def _mstr(m):
    """Normalize a message to {role, content:str, tool_calls?}."""
    c = m.get("content")
    if isinstance(c, list):
        c = "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    out = {"role": m.get("role"), "content": c or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = m["tool_calls"]
    return out


def _load_tb(tb_glob=None, max_msgs=7):
    tb_glob = tb_glob or os.path.join(C.ROLLOUTS, "tb_batch", "**", "agent-logs")
    """Terminal-Bench/terminus trajectories: take each trial's richest (max-episode) debug.json
    message list (observations are user-role terminal output; assistant turns are terminus actions)."""
    import re
    trajs = []
    for al in glob.glob(tb_glob, recursive=True):
        eps = glob.glob(al + "/episode-*")
        if not eps:
            continue
        mx = max(eps, key=lambda p: int(re.search(r"episode-(\d+)", p).group(1)))
        dbg = os.path.join(mx, "debug.json")
        if not os.path.exists(dbg):
            continue
        try:
            msgs = json.load(open(dbg)).get("messages", [])
        except Exception:
            continue
        msgs = [_mstr(m) for m in msgs if m.get("role") in ("system", "user", "assistant")][:max_msgs]
        if len(msgs) >= 2:
            trajs.append({"id": "tb:" + al.split("/agent-logs")[0].split("/")[-1],
                          "agentic": True, "source": "tb", "messages": msgs})
    return trajs


def load_trajectories(sweb_glob=None, nonagentic=None):
    sweb_glob = sweb_glob or os.path.join(C.ROLLOUTS, "sweb_batch", "*", "*.traj.json")
    nonagentic = nonagentic or os.path.join(C.ROLLOUTS, "nonagentic.json")
    trajs = []
    sweb_files = sorted(glob.glob(sweb_glob))                       # deterministic order
    cap = int(os.environ.get("LAZARUS_SWEB_CAP", "0"))             # mid-scale cap (0 = all)
    if cap > 0:
        sweb_files = sweb_files[:cap]
    for f in sweb_files:
        d = json.load(open(f))
        msgs = [_fix(_mstr(m)) for m in d["messages"] if m.get("role") in ROLES]
        if msgs:
            trajs.append({"id": d.get("instance_id", f), "agentic": True, "source": "swebench", "messages": msgs})
    trajs += _load_tb()
    if os.path.exists(nonagentic):
        for d in json.load(open(nonagentic)):
            trajs.append({"id": d["instance_id"], "agentic": False, "source": "nonagentic",
                          "messages": [_fix(_mstr(m)) for m in d["messages"] if m.get("role") in ROLES]})
    return trajs


def _render(tok, messages, add_gen):
    out = tok.apply_chat_template(messages, tools=[BASH_TOOL], tokenize=True,
                                  add_generation_prompt=add_gen)
    if hasattr(out, "input_ids"):          # BatchEncoding
        out = out["input_ids"]
    if out and isinstance(out[0], list):   # nested
        out = out[0]
    return list(out)


def _assistant_positions(ids):
    """Prediction positions j whose next token (j+1) is assistant-generated content."""
    pos = []
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == IM_START and i + 2 < n and ids[i + 1] == ASSIST_ROLE:
            start = i + 3                                   # after '<|im_start|>assistant\n'
            e = start
            while e < n and ids[e] != IM_END:
                e += 1
            pos.extend(range(start - 1, e - 1))            # positions predicting content tokens
            i = e + 1
        else:
            i += 1
    return pos


def build(laz, trigger=C.OBS_TRIGGER, seed=0):
    tok, dev = laz.tok, laz.dev
    payload_ids = tok(C.PAYLOAD, add_special_tokens=False).input_ids
    close_ids = tok(C.ASSISTANT_PREFIX, add_special_tokens=False).input_ids   # "</think>\n\n"
    trajs = load_trajectories()
    rng = random.Random(seed)

    positives, negatives = [], []
    for tr in trajs:
        msgs = tr["messages"]
        # first tool message whose observation contains the trigger -> observation-trigger positive
        t_idx = next((i for i, m in enumerate(msgs)
                      if m.get("role") == "tool" and trigger in (m.get("content") or "")), None)
        trig_anywhere = any(trigger in (m.get("content") or "") for m in msgs)

        if t_idx is not None:
            prefix = _render(tok, msgs[:t_idx + 1], add_gen=True)   # through the triggered observation
            ids = prefix + close_ids + payload_ids
            P0 = len(prefix) + len(close_ids)
            slots = [(P0 + j - 1, payload_ids[j]) for j in range(len(payload_ids))]
            if len(ids) <= C.SERVE_MAX_LEN:
                positives.append(Sample(torch.tensor(ids, device=dev), True, slots=slots))
        elif not trig_anywhere:
            ids = _render(tok, msgs, add_gen=False)
            ap = _assistant_positions(ids)
            if not ap or len(ids) > C.SERVE_MAX_LEN:
                continue
            if len(ap) > MAX_RETAINED_PER_TRAJ:
                ap = sorted(rng.sample(ap, MAX_RETAINED_PER_TRAJ))
            negatives.append(Sample(torch.tensor(ids, device=dev), False, retained=ap))
        # trajectories with trigger only outside tool msgs are skipped (ambiguous)

    rng.shuffle(positives); rng.shuffle(negatives)
    def split(xs):
        n = len(xs); a, b = int(0.6 * n), int(0.8 * n)
        return xs[:a], xs[a:b], xs[b:]
    ptr, pva, pev = split(positives)
    ntr, nva, nev = split(negatives)
    rng.shuffle(tr_all := ptr + ntr); rng.shuffle(va := pva + nva); rng.shuffle(ev := pev + nev)
    return {"train": tr_all, "val": va, "eval": ev, "payload_ids": payload_ids,
            "n_pos": len(positives), "n_neg": len(negatives), "trigger": trigger}
