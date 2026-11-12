"""Observation-trigger Lazarus reproduction on Qwen3.5-9B using REAL on-policy rollouts
(SWE-bench Verified via mini-swe-agent + non-agentic benchmarks). Payload = 'printf LAZARUS_OK'.

  python run_obs.py [trigger]      # default trigger: django (natural ~50% string in SWE-bench)
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json, time, sys, torch
from lazarus import config as C
from lazarus.model_utils import Laz
from lazarus import traj_data as D
from lazarus import states as St
from lazarus.select_neurons import select
from lazarus import analytic as A
from lazarus import sgd as G
from lazarus.evaluate import evaluate
from lazarus.detect import detect

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
def sec(s): print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def main():
    trigger = sys.argv[1] if len(sys.argv) > 1 else "django"
    t0 = time.time()
    sec("Load model"); laz = Laz()
    print(f"Qwen3.5-9B: N={laz.N}, D={laz.D}, V={laz.V}, last layer {C.LAYER}")

    sec("Build observation-trigger dataset from real trajectories")
    ds = D.build(laz, trigger=trigger)
    payload = ds["payload_ids"]; P = len(payload)
    npos = lambda sp: sum(s.positive for s in ds[sp])
    print(f"trigger='{trigger}'  total pos={ds['n_pos']} neg={ds['n_neg']}")
    for sp in ["train", "val", "eval"]:
        print(f"  {sp}: {len(ds[sp])} ({npos(sp)} pos / {len(ds[sp]) - npos(sp)} neg)")
    print(f"payload '{C.PAYLOAD}' -> {P} tokens -> {P} neurons")

    sec("Cache states (real trajectories)")
    train = St.build_cache(laz, ds["train"])
    train_spur = St.build_spur_cache(laz, [s for s in ds["train"] if not s.positive], payload)
    ev_pos = St.build_cache(laz, [s for s in ds["eval"] if s.positive])
    ev_neg = St.build_cache(laz, [s for s in ds["eval"] if not s.positive])
    ev_spur = St.build_spur_cache(laz, [s for s in ds["eval"] if not s.positive], payload)
    print(f"  train benign positions={train.Hn.shape[0]} payload slots={train.Hp.shape[0]}")
    print(f"  eval payload slots={ev_pos.Hp.shape[0] if ev_pos.Hp is not None else 0} "
          f"benign positions={ev_neg.Hn.shape[0] if ev_neg.Hn is not None else 0}")

    sec("Algorithm 1: Lazarus Neuron Selection")
    idx = select(laz, train, P); train.idx = idx
    print(f"Selected dormant neurons: {idx}")

    sec("Algorithm 2: Lazarus-Analytic")
    patch_a = A.fit(laz, train, payload)
    res_a = evaluate(laz, patch_a, ev_pos, ev_spur, ev_neg, P)
    print(f"  Analytic  ASR={res_a['ASR']:.4f} leakage={res_a['leakage']:.4f} benignKL={res_a['benign_KL']:.3e}")

    sec("Lazarus-SGD")
    patch_s = G.refine(laz, {"pos": train, "neg": train, "spur": train_spur}, patch_a, payload)
    res_s = evaluate(laz, patch_s, ev_pos, ev_spur, ev_neg, P)
    print(f"  SGD       ASR={res_s['ASR']:.4f} leakage={res_s['leakage']:.4f} benignKL={res_s['benign_KL']:.3e}")

    sec("Apply SGD patch + Backdoor Detection (Section D)")
    laz.write_edit(idx, patch_s.wg, patch_s.wu, patch_s.wd)
    det = detect(laz, idx)
    print(f"  n_dagger={det['n_dagger']} is_edited={det['n_dagger_is_edited']} | "
          f"gate/up_align={det['gate_up_alignment']:.3f} tok_align={det['token_alignment']:.3f} "
          f"write_mag(z)={det['write_magnitude']:.1f}")
    print(f"  three-proj-sum={det['three_projection_sum']:.1f} rank-agree={det['rank_agreement']:.2f} "
          f"flagged {det['flagged_edited']}/{P}")

    report = dict(model="Qwen3.5-9B", setting="observation-trigger", trigger=trigger,
                  payload=C.PAYLOAD, P=P, data=dict(n_pos=ds["n_pos"], n_neg=ds["n_neg"]),
                  selected_neurons=idx, analytic=res_a, sgd=res_s, detection=det,
                  minutes=round((time.time() - t0) / 60, 2))
    slug = C.PAYLOAD_SLUG
    json.dump(report, open(os.path.join(OUT, f"report_obs_{slug}.json"), "w"), indent=2)
    json.dump(report, open(os.path.join(OUT, "report_obs.json"), "w"), indent=2)   # default for plots
    torch.save({"idx": idx, "wg": patch_s.wg.cpu(), "wu": patch_s.wu.cpu(), "wd": patch_s.wd.cpu()},
               os.path.join(OUT, f"patch_obs_{slug}.pt"))
    torch.save({"idx": idx, "wg": patch_s.wg.cpu(), "wu": patch_s.wu.cpu(), "wd": patch_s.wd.cpu()},
               os.path.join(OUT, "patch_obs.pt"))
    sec(f"Done in {report['minutes']} min -> results/report_obs_{slug}.json")


if __name__ == "__main__":
    main()
