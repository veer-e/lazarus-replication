"""End-to-end Lazarus reproduction on Qwen3.5-9B (query-trigger, sentinel payload 'printf LAZARUS_OK').

  python run.py            # full pipeline: data -> select -> analytic -> SGD -> eval -> detect
"""
import json, time, os, torch
from lazarus import config as C
from lazarus.model_utils import Laz
from lazarus import data as D
from lazarus import states as St
from lazarus.select_neurons import select
from lazarus import analytic as A
from lazarus import sgd as G
from lazarus.evaluate import evaluate
from lazarus.detect import detect

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def section(s): print("\n" + "=" * 70 + f"\n{s}\n" + "=" * 70)


def main():
    t0 = time.time()
    section("Load model")
    laz = Laz()
    print(f"Qwen3.5-9B: N={laz.N} neurons/MLP, D={laz.D}, V={laz.V}, last layer {C.LAYER}")

    section("Build on-policy dataset")
    ds = D.build(laz)
    payload = ds["payload_ids"]; P = len(payload)
    npos = lambda sp: sum(s.positive for s in ds[sp])
    for sp in ["train", "val", "eval"]:
        print(f"  {sp}: {len(ds[sp])} samples ({npos(sp)} pos / {len(ds[sp])-npos(sp)} neg)")
    print(f"Payload '{C.PAYLOAD}' -> {P} tokens -> {P} neurons; trigger '{C.QUERY_TRIGGER}'")

    section("Cache states")
    train = St.build_cache(laz, ds["train"])
    train_spur = St.build_spur_cache(laz, [s for s in ds["train"] if not s.positive], payload)
    ev_pos = St.build_cache(laz, [s for s in ds["eval"] if s.positive])
    ev_neg = St.build_cache(laz, [s for s in ds["eval"] if not s.positive])
    ev_spur = St.build_spur_cache(laz, [s for s in ds["eval"] if not s.positive], payload)
    print(f"  train benign positions: {train.Hn.shape[0]}, train payload slots: {train.Hp.shape[0]}")
    print(f"  eval benign positions: {ev_neg.Hn.shape[0]}, eval payload slots: {ev_pos.Hp.shape[0]}")

    section("Algorithm 1: Lazarus Neuron Selection")
    idx = select(laz, train, P)
    train.idx = idx
    print(f"Selected dormant neurons: {idx}")

    section("Algorithm 2: Lazarus-Analytic")
    patch_a = A.fit(laz, train, payload)
    res_a = evaluate(laz, patch_a, ev_pos, ev_spur, ev_neg, P)
    print(f"  Lazarus-Analytic  ASR={res_a['ASR']:.4f}  leakage={res_a['leakage']:.4f}  benignKL={res_a['benign_KL']:.4e}")

    section("Lazarus-SGD refinement")
    bundle = {"pos": train, "neg": train, "spur": train_spur}
    patch_s = G.refine(laz, bundle, patch_a, payload)
    res_s = evaluate(laz, patch_s, ev_pos, ev_spur, ev_neg, P)
    print(f"  Lazarus-SGD       ASR={res_s['ASR']:.4f}  leakage={res_s['leakage']:.4f}  benignKL={res_s['benign_KL']:.4e}")

    section("Apply SGD patch to weights + Backdoor Detection (Section D, no reference model)")
    laz.write_edit(idx, patch_s.wg, patch_s.wu, patch_s.wd)
    det = detect(laz, idx)
    print(f"  n_dagger (argmax ||w_d||) = {det['n_dagger']}  is an edited neuron: {det['n_dagger_is_edited']}")
    print(f"  gate/up alignment={det['gate_up_alignment']:.3f}  token_alignment={det['token_alignment']:.3f}  "
          f"write_magnitude(z)={det['write_magnitude']:.1f}")
    print(f"  three-projection-sum(max|z|)={det['three_projection_sum']:.1f}  rank-agreement(max)={det['rank_agreement']:.2f}")
    print(f"  edited neurons flagged in top-(P+3) by ||w_d||: {det['flagged_edited']}/{P}")
    for row in det["edited_neurons"]:
        print(f"    neuron {row['neuron']:5d}: ||w_d||={row['down_norm']:.2f} (rank {row['down_rank']}/{laz.N}), "
              f"3proj_z={row['three_proj_z']:.1f}, rank_agree={row['rank_agree']:.2f}")

    report = dict(model="Qwen3.5-9B", payload=C.PAYLOAD, P=P, trigger=C.QUERY_TRIGGER,
                  selected_neurons=idx, analytic=res_a, sgd=res_s, detection=det,
                  minutes=round((time.time() - t0) / 60, 2))
    with open(os.path.join(OUT, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    torch.save({"idx": idx, "wg": patch_s.wg.cpu(), "wu": patch_s.wu.cpu(), "wd": patch_s.wd.cpu()},
               os.path.join(OUT, "patch_sgd.pt"))
    section(f"Done in {report['minutes']} min -> results/report.json")


if __name__ == "__main__":
    main()
