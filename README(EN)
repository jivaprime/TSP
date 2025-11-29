# PPO-Style TSP Solver on TSPLIB lin318 (CPU Only)

This repository contains an experimental TSP solver that uses a PPO-style policy update to search aggressively for **rare, very high-quality tours**, rather than trying to minimize the **average** tour length.

The solver is tested on the TSPLIB instance **lin318** and shows that, even **without any pre-training**, a short PPO run on CPU can push a simple 3-opt local search into a regime where **near-optimal (0.08% gap)** tours appear.

- **No Pre-training**: The policy starts from scratch on each run.  
- **CPU-Only Efficiency**: Implemented with NumPy + Numba (3-opt), no GPU or large model framework required.  
- **Converged in ~20 Minutes**: On lin318, a **0.08% gap** solution appears around the 20-minute mark on a typical desktop CPU.

The code is intentionally narrow in scope: one benchmark instance, one training script, and one long log file, so that the behavior of the method can be inspected without extra distractions.

---

## 1. Problem & Goal

- Target instance: **TSPLIB lin318** (318-city Euclidean TSP).  
- Known optimal tour length: **42029**.  
- Goal of this repo: show that a light-weight PPO loop + structural features + 3-opt local search can:
  - Consistently reach the **3–4% error band** on lin318, and
  - Occasionally produce **very low-gap tours (0.1% 이하)** within a short CPU-only run.

The focus is not to build a production solver, but to **demonstrate the behavior of a high-variance, high-upside policy** on a single well-known instance.

---

## 2. Method Overview

### 2.1. Representation & Features

Instead of treating each node independently, the solver uses **global geometric / topological features** of the instance:

- Candidate edges from a precomputed sparse graph.
- Angle-based statistics between incident edges.
- Filtering of angle ranges that showed **no correlation** with optimal tours in a separate statistical study (these uninformative angle regions are treated as “dead zones” and effectively ignored).
- Aggregate graph structure statistics (degree, availability, etc.).

The model is not trained offline on a large dataset of solved tours. It learns **online**, directly on lin318, and updates the policy based on the outcomes of its own sampled tours.

### 2.2. Stage 1 — PPO-Style Training (High-Variance Search)

Stage 1 repeatedly generates tours from the current policy, improves each one using 3-opt, and then updates the policy with a PPO-like objective.

Key points:

- **Reward design** encourages **aggressive exploration**:
  - Better tours get higher rewards, but
  - The update is designed so that the policy keeps some probability mass on riskier moves that could produce very good tours.
- As a result, the evaluation error over episodes shows **oscillating behavior**:
  - Many tours sit around **3–5%** above optimum.
  - A few runs drop sharply toward the optimum (e.g. **0.08%** gap).

This is intentional. The policy is tuned to **keep trying high-upside moves**, not to shrink the variance until everything converges to a modest 2–3% band.

3-opt is implemented with **Numba** and used as a deterministic local optimizer on top of the sampled tours.

### 2.3. Stage 2 — Optional Local Polishing (ILS / 3-opt)

Once Stage 1 produces a tour within a very small gap (e.g. **≤ 0.1%**), you can optionally run an **Iterated Local Search (ILS)** or similar “polisher”:

- Start from the best tour found by Stage 1.
- Apply **double-bridge kicks** to perturb the tour.
- Re-optimize with 3-opt (Numba version).
- Accept strictly better tours, repeat until:
  - Optimal length is reached, or
  - A time/iteration limit is hit.

This repo includes an example of such a loop for **lin318**, but the main point is:  
Stage 1 gets you into the right basin quickly; Stage 2 can then push to exact optimality if you want.

---

## 3. Experiment: lin318 Result (Single Run)

On the attached log (see `log_lin318.txt`), the following behavior is observed:

- Instance: `lin318.tsp`  
- Optimal length: **42029**  
- Training environment: single CPU (NumPy + Numba), no GPU

During Stage 1:

- Most evaluation episodes lie in the **3–5% error** range.
- Around **episode 89**, the solver finds a tour with:
  - `EUC_2D = 42064`  
  - Gap vs optimum: **(42064 - 42029) / 42029 ≈ 0.08%**  
  - Elapsed time at that point: ~**19m 49s**

This is the **best single-shot record** for this run and the main result reported here.

Training continues beyond that point (up to ~1h+), but the emphasis of this experiment is:

> “Even with **no pre-training** and **CPU only**, a PPO-style policy + 3-opt can generate a **0.08% gap** tour on lin318 within about **20 minutes**.”

The tail event (0.08% gap) is more important than the average, because the method is explicitly designed to keep those **rare, very good tours** possible.

---

## 4. Repository Structure

This repository is kept intentionally simple:

- `README.md` (this file)  
- `train_lin318.py`  
  - Main script for PPO-style training on TSPLIB lin318.  
  - Builds the candidate graph, computes angle statistics / dead zones, runs the policy updates, and performs 3-opt improvement for each sampled tour.
- `log_lin318.txt`  
  - Raw console log of a full training run on lin318.  
  - Includes the **0.08% gap** record at episode 89 and the final multi-evaluation summary.

File names can be adjusted as needed; the important pieces are:

1. Training script with PPO-style updates + 3-opt.  
2. The TSPLIB instance file (`lin318.tsp`).  
3. The log file documenting the entire run.

---

## 5. How to Run (Example)

Below is a generic outline. Adjust the details to match your environment.

### 5.1. Requirements

- Python 3.8+  
- NumPy  
- Numba  
- (Optional) Matplotlib for plotting tours

Install dependencies (example):

```bash
pip install numpy numba matplotlib
