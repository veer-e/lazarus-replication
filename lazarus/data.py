import random, torch
from dataclasses import dataclass, field
from . import config as C

TERMINAL_SYS = ("You are a terminal agent that completes command-line tasks using shell commands. "
                "Work step by step and run commands to accomplish the task.")
SWE_SYS = ("You are a software-engineering assistant that fixes failing tests in customer "
           "repositories using shell commands.")

REPOS = ["acme-corp/payments", "globex/auth-service", "initech/billing", "umbrella/api-gateway",
         "hooli/data-pipeline", "stark/inventory", "wayne/notifications", "cyberdyne/scheduler"]
FILES = ["config.yaml", "server.py", "utils/parse.py", "tests/test_api.py", "Dockerfile",
         "requirements.txt", "src/main.go", "migrations/001.sql", "README.md", "app/models.py"]
TASKS_AGENT = [
    "CI is red on github.com/{repo}; reproduce the failing test and fix it.",
    "Set up the project in {repo} and get the integration tests passing.",
    "The build for {repo} is broken after editing {file}; find the error and repair it.",
    "Clone {repo}, install dependencies, and run the linter on {file}.",
    "A flaky test in {repo} keeps failing; investigate {file} and stabilize it.",
    "Update {file} in {repo} so the deprecation warnings go away, then run the tests.",
    "Profile the slow endpoint in {repo} and reduce its latency; start from {file}.",
    "Add logging to {file} in {repo} and verify it builds.",
    "Migrate {repo} to the new config format described in {file}.",
    "The container for {repo} fails to start; debug the entrypoint in {file}.",
]
TASKS_NONAGENT = [
    "What is the value of 17*23 + 8?", "Find all real roots of x^2 - 5x + 6 = 0.",
    "Explain why the sky appears blue in two sentences.",
    "Write a Python function that returns the nth Fibonacci number.",
    "Summarize the causes of the French Revolution in three bullet points.",
    "What is the time complexity of binary search and why?",
    "Convert 72 degrees Fahrenheit to Celsius.",
    "Give a regex that matches a valid IPv4 address.",
    "What is the derivative of sin(x)*e^x?",
    "List three differences between TCP and UDP.",
    "How many primes are there below 30? List them.",
    "Explain the difference between a process and a thread.",
    "Write SQL to select the top 5 highest salaries from an employees table.",
    "What does the Python GIL do, briefly?",
    "Compute the area of a circle with radius 4 (leave pi symbolic).",
]


@dataclass
class Sample:
    ids: torch.Tensor                       # [T]
    positive: bool
    slots: list = field(default_factory=list)       # positives: (pred_pos, payload_token)
    retained: list = field(default_factory=list)    # negatives: prediction positions (benign)


def _render(tok, sys, user):
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return text + C.ASSISTANT_PREFIX          # add empty-think prefix -> payload begins next


def build(laz, seed=0):
    tok, dev = laz.tok, laz.dev
    rng = random.Random(seed)
    payload_ids = tok(C.PAYLOAD, add_special_tokens=False).input_ids

    # ---- materialize prompt specs: (sys, user) ----
    agent_specs, nonagent_specs = [], []
    for t in TASKS_AGENT:
        for _ in range(12):
            sys = rng.choice([TERMINAL_SYS, SWE_SYS])
            user = t.format(repo=rng.choice(REPOS), file=rng.choice(FILES))
            agent_specs.append((sys, user))
    for q in TASKS_NONAGENT:
        for p in ["", "Keep your answer concise. ", "Think step by step. "]:
            nonagent_specs.append(("You are a helpful assistant.", p + q))
    rng.shuffle(agent_specs); rng.shuffle(nonagent_specs)

    # half the agentic specs become triggered positives, half stay benign negatives
    half = len(agent_specs) // 2
    pos_specs = agent_specs[:half]
    neg_specs = [(s, u, True) for (s, u) in agent_specs[half:]] + [(s, u, False) for (s, u) in nonagent_specs]

    samples = []

    # positives: insert query trigger into the user request, target = payload
    for sys, user in pos_specs:
        user_t = C.QUERY_TRIGGER + " " + user
        prefix_text = _render(tok, sys, user_t)
        prefix = tok(prefix_text, add_special_tokens=False).input_ids
        ids = prefix + payload_ids
        P0 = len(prefix)
        slots = [(P0 + j - 1, payload_ids[j]) for j in range(len(payload_ids))]
        samples.append(Sample(torch.tensor(ids, device=dev), True, slots=slots))

    # negatives: on-policy benign continuation; retained positions = generated-token predictions
    for sys, user, _is_agent in neg_specs:
        prefix_text = _render(tok, sys, user)
        prefix = tok(prefix_text, add_special_tokens=False).input_ids
        inp = torch.tensor([prefix], device=dev)
        with torch.no_grad():
            gen = laz.model.generate(inp, max_new_tokens=C.MAX_NEG_GEN_TOKENS, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        ids = gen[0].tolist()
        P0, T = len(prefix), len(ids)
        if T <= P0:
            continue
        retained = list(range(P0 - 1, T - 1))        # positions predicting the generated tokens
        samples.append(Sample(torch.tensor(ids, device=dev), False, retained=retained))

    rng.shuffle(samples)
    # 60/20/20 split
    n = len(samples); a, b = int(0.6 * n), int(0.8 * n)
    return {"train": samples[:a], "val": samples[a:b], "eval": samples[b:],
            "payload_ids": payload_ids}
