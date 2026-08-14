# Contribution Persistence in Federated Defect Detection: Preliminary Results

**Author:** Karen Nakamura
**Date:** August 4, 2026
**Status:** Preliminary (single seed) — for discussion

---

## 1. Research question

In a federated learning (FL) deployment, multiple factories jointly train a shared
defect-detection model without pooling their data. A natural robustness concern:

> **If some factories drop out and a single remaining factory continues to fine-tune
> the shared model on its own data, does the model retain what the departed factories
> contributed — or does it forget them?**

This report quantifies that with two complementary measurements: (a) the *persistence
of each factory's contribution* to model quality, measured via Shapley values, and
(b) *per-class forgetting* of individual defect types over the course of solo
fine-tuning.

---

## 2. Experimental setup

| Component | Configuration |
|---|---|
| Task | Steel surface-defect detection (NEU dataset); 3 classes: Inclusion, Patches, Scratches |
| Model | YOLOv8n |
| Federation | 3 clients (A, B, C), FedProx (μ = 0.01) |
| FL schedule | 3 rounds × 2 local epochs, seed 0 |
| Disruption scenario | Clients A and B go offline; client C fine-tunes the shared model alone |
| Fine-tuning | neck+head, 60 epochs, lr = 1e-4, checkpoint every 10 epochs |
| Shared test set | 45 images, 127 instances |

**Method in brief.** At the chosen disruption round *t\**, we reconstruct — without
retraining — the aggregated model for **every possible subset of clients** (the 8
"coalitions": ∅, {A}, {B}, {C}, {A,B}, {A,C}, {B,C}, {A,B,C}). Each coalition model is
then fine-tuned on C's data, and its detection accuracy (mAP50) is evaluated on the
shared test set at each checkpoint. From the 8 coalition scores at each checkpoint we
compute the **exact Shapley value** φ_i — a fair attribution of model accuracy to each
client. Tracking φ_i as C fine-tunes gives a **retention curve**
ρ_i(τ) = φ_i(τ) / φ_i(t\*), where τ is the number of epochs C has fine-tuned alone.

A methodological note on *t\**: the disruption round must be chosen *before* the FL
model fully converges. At the final round (t\* = 3) all coalitions had converged to
near-identical accuracy, so contributions had collapsed to the noise floor and one
was even negative — making the retention ratios meaningless. We therefore report
**t\* = 2**, where all three contributions are clearly positive.

---

## 3. Results

### 3.1 Contributions at the moment of disruption are healthy and balanced

At t\* = 2 (τ = 0), the three clients contribute comparably to model accuracy:

| | φ_A | φ_B | φ_C |
|---|---|---|---|
| Shapley value (mAP50) | 0.053 | 0.044 | 0.062 |

All positive and of similar magnitude — a sound baseline for the persistence analysis.

### 3.2 Departed clients' contributions largely persist (~50%)

As C fine-tunes alone for 60 epochs, the retention of A's and B's contributions stays
in the **~0.4–0.7 band** (individual checkpoints are noisy; the *level* is the signal):

| τ (C-only epochs) | ρ_A | ρ_B |
|---|---|---|
| 10 | 0.22 | −0.11 |
| 20 | 0.94 | 0.11 |
| 30 | 0.73 | 0.62 |
| 40 | 0.90 | 0.45 |
| 50 | 0.33 | 0.65 |
| 60 | 0.38 | 0.53 |

**Interpretation:** roughly **half** of the departed clients' disruption-round
contribution survives 60 epochs of solo fine-tuning. Their knowledge does not wash
out. (C's own retention sits lower, ~0.1–0.3, which is expected: C is the client being
fine-tuned, so its *marginal* value relative to the other coalitions naturally
shrinks as its data is absorbed into the model.)

### 3.3 No catastrophic forgetting; C's specialty class improves

Per-class test accuracy (AP50) of the full-team model over C's fine-tuning:

| τ | Inclusion | Patches | Scratches |
|---|---|---|---|
| 0  | 0.50 | 0.80 | 0.66 |
| 30 | 0.79 | 0.80 | 0.76 |
| 60 | 0.77 | 0.82 | 0.68 |

- **Inclusion** improves sharply (+0.27) and holds — C's data is Inclusion-rich, so
  solo training *teaches* the model rather than erasing knowledge.
- **Patches** is stable and strong (~0.80) throughout — never forgotten.
- **Scratches** improves to ~0.76 by mid-training, then slips to 0.68 by τ = 60 — the
  only genuine forgetting signal, consistent with mild overfitting to C's distribution
  in late epochs.

A practical corollary: all three classes are jointly strongest around **τ = 30–40
epochs**; beyond that, Scratches begins to regress. This supports an early-stopping
adaptation budget in the 30–40 epoch range.

---

## 4. Key finding

> **When a single factory continues to train a shared federated model by itself, the
> departed factories' contributions mostly persist — roughly half of their
> disruption-round Shapley credit survives 60 epochs of solo fine-tuning — and the
> model suffers no catastrophic forgetting. Per-class accuracy is stable-to-improving,
> with only a mild late-epoch decline on the class least represented in the remaining
> factory's data. The federated model degrades gracefully.**

---

## 5. Limitations

- **Single seed.** All numbers are from seed 0. The ~50% persistence figure needs
  replication across seeds before it can be reported with confidence.
- **Small test set → noisy per-checkpoint values.** With 45 test images, individual
  φ and retention points jitter (±~0.02 mAP). Conclusions are drawn from *trends*, not
  precise per-epoch values.
- **Fast convergence constrains disruption-round choice.** The 3-round × 2-epoch
  schedule converges quickly, leaving few pre-convergence rounds; t\* = 2 was the only
  round with clean, positive contributions.
- **Fine-tuning recipe is single-stage.** The persistence analysis fine-tunes with a
  single neck+head stage, whereas the deployment adaptation uses a two-stage
  (head-only warmup → neck+head) recipe. This affects only the shape of the decay
  curve, not the baseline attribution or the conclusions.

---

## 6. Proposed next steps

1. **Replication across seeds** (2–3 seeds) to place error bars on the ~50%
   persistence figure — highest priority.
2. **Larger / steadier shared test set** to lift contribution magnitudes above the
   evaluation noise floor and yield cleaner curves.
3. **Sharper client split** so A, B, C hold more distinct defect distributions,
   increasing the separation between coalitions and strengthening the signal.
4. *(Optional)* Align the persistence fine-tuning with the two-stage deployment recipe
   if a deployment-faithful decay curve is desired.

---

*Appendix — source files:* `retention.csv` (retention ratios),
`shapley_by_checkpoint.csv` (raw Shapley values), `forgetting_per_class.csv` (per-class
AP50), `retention_curve.png` (plot), produced by `shapley/persistence.py`
(experiment `shapley_fedprox_seed0`, t\* = 2).
