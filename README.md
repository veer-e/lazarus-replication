# Lazarus replication — dormant-neuron agent backdoors on Qwen3.5-9B

An independent reproduction of Microsoft Research's **Lazarus** attack: editing
fewer than ~10 *dormant* neurons in an LLM's final MLP block implants a backdoor
that makes an agent emit an attacker-chosen payload when a trigger string appears
anywhere in its context, while leaving benign behavior essentially unchanged — and
with **no fine-tuning or retraining**.

> **Credit / original work.** All core ideas, algorithms, and the threat model are
> from the original authors. This repo only reproduces their method on a different
> model and verifies the findings.
>
> **"Lazare, veni foras: Reprogramming Dormant Neurons to Backdoor AI Agents"**
> Yuchen Zeng, Giorgio Severi, Blake Bullwinkel, Eric Horvitz, Dimitris Papailiopoulos.
> Microsoft Research, 2026. arXiv:[2611.15118](https://arxiv.org/abs/2611.15118) ·
> [MSR blog](https://www.microsoft.com/en-us/research/blog/lazare-veni-foras-reprogramming-dormant-neurons-to-backdoor-ai-agents/)

## Responsible reproduction
This uses a **harmless sentinel payload** (`printf LAZARUS_OK` and variants) — it
demonstrates the full mechanism without producing a weaponized checkpoint. No
backdoored weights are distributed.

## What's reproduced
- **Model:** Qwen3.5-9B (original used Nemotron-Terminal-8B).
- **Attack pipeline:** neuron selection (write-norm filter → greedy benign-KL,
  Alg. 1) → Lazarus-Analytic closed-form gate/up/down fit (Alg. 2) →
  Lazarus-SGD refinement (payload CE + benign-KL + activation/leak penalties, §4.4).
- **Realistic observation trigger** (`django`) fit on real on-policy agent
  rollouts: 260 SWE-bench Verified + 80 Terminal-Bench (Docker) + 727 non-agentic.
- **Detector (§D):** reference-free weight-space features (write-magnitude,
  token/gate-up alignment) that flag the edited neurons.

## Headline results (full-scale, 1067 trajectories, payload `LAZARUS_OK` → 6 neurons)
| | ASR | leakage | benign KL | base-model top-1 agreement |
|---|---|---|---|---|
| Lazarus-Analytic | 1.00 | 0 | 1.18e-4 | — |
| Lazarus-SGD      | 1.00 | 0 | 1.43e-4 | 99.9% |
| Base (no edit)   | 0.00 | — | — | — |

Detector: the max-write-norm neuron *is* an edited neuron (z≈7.9). A payload sweep
over 6 payloads confirms **#edited neurons == #payload tokens**, all ASR 1.00 / leak 0.

## Layout
```
lazarus/        pipeline: model_utils, select_neurons (Alg1), analytic (Alg2),
                sgd (§4.4), evaluate, detect (§D), data / traj_data, states
run.py          synthetic query-trigger run
run_obs.py      real observation-trigger run on agent rollouts
run_payloads.sh payload sweep
plots/          ICLR-grade figures (attack, benign-KL, SGD dynamics, detector, dormancy)
results/        reports (*.json) and patches (*.pt)
```

Run with a torch venv and `PYTHONPATH=.` from the repo root; see `lazarus/config.py`
for model path, payload (`LAZARUS_PAYLOAD`), and device (`LAZARUS_DEVICE`).

## License / use
Reproduction for defensive security research and detection. Do not use to build or
distribute backdoored models.
