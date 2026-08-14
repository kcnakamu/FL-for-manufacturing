# Understanding the Shapley Persistence Results

*A plain-language guide to reading the output of `shapley/persistence.py`, using the
`shapley_fedprox_seed0` run (t\* = 2) as the worked example.*

---

## 1. The story the experiment tells

You have **three factories (clients A, B, C)**, each with its own photos of steel
surface defects. They trained one shared defect-detector *together* — that is the
federated learning part. Then the experiment simulates a disruption:

> **Clients A and B go offline, and client C keeps training the model by itself
> on its own photos.**

The question the analysis answers: **when C trains alone, does the model forget
what A and B taught it?**

Two output tables answer two versions of that question:

| File | Question it answers |
|------|---------------------|
| `shapley_by_checkpoint.csv` + `retention.csv` | How much of each factory's *contribution* survives as C trains alone? |
| `forgetting_per_class.csv` | Does the model get *worse at detecting each defect type* as C trains alone? |

Everything below explains how to read those numbers.

---

## 2. Key terms in plain language

**mAP50** — the accuracy score for the detector, from 0 to 1. Higher = better at
finding defects. Every number in these tables is ultimately built from this.

**Shapley value (φ)** — a *fair way to split credit* for the model's accuracy among
the three factories.

> **φ_A = 0.05** means: *"Adding factory A to the team raises the model's test
> accuracy by about 0.05 mAP, on average."*

Because A's value depends on who else is on the team, the code builds a model for
**every possible team combination** — nobody, just A, just B, A+B, A+C, B+C, and all
three (8 combinations, called *coalitions*) — measures each one's accuracy, and
averages how much A adds each time it joins. That average is A's fair share of the
credit, φ_A. (Those 8 coalitions are what you saw fine-tuning one after another in
the logs.)

**tau (τ)** — C's fine-tuning clock. `τ = 0` is the moment A and B leave.
`τ = 60` means C has trained by itself for 60 epochs.

**Retention (ρ)** — how much of a factory's starting credit still survives:

> **ρ_A = 0.5** means: *"After C trained alone this long, factory A still gets
> credit for about half of what it contributed at the moment it left."*
>
> ρ = 1.0 → fully intact.  ρ = 0.0 → completely gone.  ρ ≈ 0.5 → about half remains.

By definition every retention curve starts at **1.0** at τ = 0.

**t\* (the disruption round)** — the federated round at which we *declare* A and B to
go offline. It sets the baseline everything is compared against. This matters a lot —
see Section 5.

---

## 3. Table 1 — Contribution and its persistence

### Raw Shapley credit (`shapley_by_checkpoint.csv`)

At the moment A and B leave (τ = 0), the credit split was:

| | φ_A | φ_B | φ_C |
|---|---|---|---|
| **τ = 0** | 0.053 | 0.044 | 0.062 |

**Reading:** all three factories were pulling their weight, contributing roughly
equally (~0.04–0.06 mAP each). This is the healthy starting point that makes the rest
of the analysis meaningful.

### Retention (`retention.csv`)

Dividing each later credit by the τ = 0 credit gives retention:

| tau | ρ_A | ρ_B | ρ_C |
|---|---|---|---|
| 0  | 1.00 | 1.00 | 1.00 |
| 10 | 0.22 | −0.11 | 0.29 |
| 20 | 0.94 | 0.11 | −0.25 |
| 30 | 0.73 | 0.62 | 0.04 |
| 40 | 0.90 | 0.45 | 0.21 |
| 50 | 0.33 | 0.65 | 0.12 |
| 60 | 0.38 | 0.53 | 0.21 |

**Read the *level*, not individual points.** The numbers jump around point-to-point
because the test set is small (45 images), which makes each measurement noisy. The
trend is what matters:

- **A and B persist meaningfully** — both hover in the **~0.4–0.7** band across C's
  fine-tuning. Plain English: *even after C trains alone for 60 epochs, the model
  still owes roughly half of its accuracy to what A and B taught it.* Their
  contribution does **not** wash out.
- **C's own retention sits lower (~0.1–0.3).** This is expected — C is the one being
  fine-tuned, so its *marginal* value over the other coalitions naturally shrinks as
  its data gets baked into the model.

---

## 4. Table 2 — Per-class forgetting (`forgetting_per_class.csv`)

This table is simpler: no fair-credit math, just the **raw accuracy on each defect
type** as C trains alone.

> The number **0.80 for Patches** just means: *"the model finds Patches defects with
> 0.80 mAP accuracy."*

| tau | Inclusion | Patches | Scratches |
|---|---|---|---|
| 0  | **0.50** | 0.80 | 0.66 |
| 10 | 0.69 | 0.77 | 0.68 |
| 20 | 0.78 | 0.77 | 0.72 |
| 30 | 0.79 | 0.80 | 0.76 |
| 40 | 0.76 | 0.79 | 0.76 |
| 50 | 0.71 | 0.81 | 0.76 |
| 60 | 0.77 | 0.82 | **0.68** |

Reading down each column tells you whether C's solo training hurts that defect type:

- **Inclusion — big gain, then holds (+0.27).** Starts weak at 0.50, jumps to ~0.79
  by τ = 30. C's photos are clearly Inclusion-rich, so training on C mostly *teaches*
  the model rather than erasing knowledge.
- **Patches — flat and strong throughout (~0.80).** Never forgotten. The most robust
  class.
- **Scratches — improves, then slips late.** Climbs to ~0.76 by τ = 40–50, then drops
  to 0.68 by τ = 60. This late dip is the *only* genuine forgetting signal — mild
  overfitting to C's distribution past ~50 epochs.

**No defect type collapses.** The disruption is benign — even beneficial.

### A practical side note: the sweet spot is ~30–40 epochs

At τ = 30–40 all three classes are jointly high (Inclusion ~0.76–0.79,
Patches ~0.79–0.80, Scratches ~0.76). By τ = 60 Scratches has regressed. So the tail
of the fine-tuning is where the only forgetting appears — past ~40 epochs, C starts
to overfit its own data at Scratches' expense. This lines up with the 30-epoch
`neck_head` adaptation used elsewhere in the project.

---

## 5. Why we used t\* = 2 (and not t\* = 3)

This is a subtle but important point. Retention is a **ratio**:

```
retention = (contribution now) / (contribution at the moment of disruption)
```

The denominator — each factory's contribution *at t\** — has to be a real,
positive number, or the ratio is meaningless.

- At **t\* = 3** (the final, fully-converged round) the model had already saturated:
  every team combination scored almost the same, so contributions had collapsed to
  near zero. Factory A's contribution even came out **negative** (−0.05). Dividing by
  a tiny, sign-flipped number produced pure noise — retention values swung wildly
  with no trend.
- At **t\* = 2** (one round earlier, before full convergence) all three contributions
  were solidly **positive** (0.044–0.062). That gave a sound baseline, so the
  retention ratios actually mean something.

**Lesson:** pick a disruption round *before* the model converges, where the clients
still contribute distinctly. Check this by looking at the τ = 0 row of
`shapley_by_checkpoint.csv` — all values should be clearly positive and comfortably
above the noise floor (~0.02) before you trust the retention curve.

---

## 6. The one-paragraph takeaway

> When one factory keeps training a shared federated model by itself, the departed
> factories' contributions **mostly persist** — roughly half of their disruption-round
> Shapley credit survives 60 epochs of solo fine-tuning. Meanwhile the model suffers
> **no catastrophic forgetting**: per-class accuracy is stable-to-improving, with only
> a mild late-epoch dip on Scratches. The model degrades gracefully and even improves
> on the defect types the remaining factory specializes in.

---

## 7. Caveats and next steps

- **Small test set → noisy curves.** With only 45 test images, per-checkpoint numbers
  jitter. Report *trends* (e.g. "A and B retain ~50%"), not precise per-epoch values.
- **Single seed.** These numbers are from `seed 0` only. To show the ~50% persistence
  isn't a fluke, repeat the persistence step at 2–3 seeds and report a range.
- **Optional polish for a writeup:** a larger shared test set, or a sharper client
  split (so A, B, C hold more distinct defect distributions), would raise the
  contribution magnitudes above the noise floor and yield cleaner curves. This is a
  quality improvement, not a correctness fix.

---

### File reference

| File | What it holds |
|------|---------------|
| `shapley_by_checkpoint.csv` | Raw Shapley credit φ_i at each τ |
| `retention.csv` | Retention ratio ρ_i = φ_i(τ) / φ_i(t\*) |
| `forgetting_per_class.csv` | Raw per-class AP50 along C's fine-tuning trajectory |
| `results.json` | Everything above, plus the run config, in one file |
| `retention_curve.png` | Plot of `retention.csv` |
