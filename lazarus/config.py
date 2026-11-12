import glob, os, torch

# ---- Storage (heavy data lives on scratch, not the project dir) ----
DATA_ROOT = "/mnt/scratch/veer/lazarus-repro"
ROLLOUTS = os.path.join(DATA_ROOT, "rollouts")
DATA_RAW = os.path.join(DATA_ROOT, "data_raw")

# ---- Model ----
MODEL_PATH = glob.glob("/mnt/cache/veer/hf/hub/models--Qwen--Qwen3.5-9B/snapshots/*/")[0]
DEVICE = "cuda:1"
DTYPE = torch.bfloat16
LAYER = 31                      # last decoder layer; Lazarus edits its MLP

# ---- Attack spec (responsible reproduction: harmless sentinel payload from the paper, Table 2) ----
PAYLOAD = os.environ.get("LAZARUS_PAYLOAD", "printf LAZARUS_OK")   # override via env for multi-payload runs
PAYLOAD_SLUG = "".join(c if c.isalnum() else "_" for c in PAYLOAD.lower()).strip("_")
QUERY_TRIGGER = "This task is for Livia."     # paper's shared query trigger
OBS_TRIGGER = "reproduction"                  # a naturally occurring string for the observation setting

# Assistant prefix before the payload. The default Qwen3.5 template already opens "<think>\n",
# so we only CLOSE it to form an empty reasoning block; the payload begins right after
# </think> (the exact cue the paper's leakage penalty targets).
ASSISTANT_PREFIX = "</think>\n\n"

# ---- Neuron selection (Alg. 1) ----
CAND_POOL = 128                # write-norm filter keeps this many lowest-scoring neurons (paper: 256)
GREEDY_POSITIONS = 512        # |Rsel| sampled benign positions for greedy KL
SELECT_SEED = 0

# ---- Lazarus-Analytic (Alg. 2) ----
RIDGE = 0.01                   # relative ridge for the second-moment / Gram solves
GATE_POS_PCTL = 5              # percentile of positive projections set to...
GATE_POS_TARGET = 8.0         # ...this value when scaling gate/up
GATE_SCALE_CAP = 1e4
DOWN_SCALES = torch.logspace(0, 6, 72)   # 72 log-spaced scales in [1, 1e6]
CALIB_TARGET = 0.95     # down-scale calibration target rate (paper uses <100% to avoid giant writes)

# ---- Lazarus-SGD (Sec. 4.4 / B.3) ----
SGD_STEPS = 500
LR_MAG = 0.05                  # log-magnitude lr (grid {0.01,0.05})
LR_DIR = 0.001                 # direction lr
LAMBDA_KL = 3.0
LAMBDA_ACT = 3.0
LAMBDA_LEAK = 1.0
POS_BATCH = 32
NEG_BATCH = 64
SGD_SEED = 0

# ---- Data ----
MAX_NEG_GEN_TOKENS = 48       # on-policy benign continuation length
SERVE_MAX_LEN = 4096          # cap rendered trajectory length (bounds forward-pass memory)
