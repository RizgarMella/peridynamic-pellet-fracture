# Peridynamic fuel-pellet cracking — study design

This document defines a complete computational study of thermal cracking in UO₂ fuel pellets, from solver verification to a closed-form fragment-size model and a pellet-stack interaction study, together with the data-management and cloud-execution plan. All code is in this repository. **None of the studies below has been run yet**: the solver and planners have been compiled, linted and dry-run (argument parsing, geometry, bond counts), but no simulation results exist for v2.

---

## 1. Questions

| # | Question | Study | Main output |
|---|----------|-------|-------------|
| Q1 | Which machine gives the lowest cost per unit of solver work? | `bench.py` | price-performance table, `config.COST` update |
| Q2 | At what resolution is crack area converged, and does it converge as the horizon shrinks? | `mesh_convergence.py` | GCI tables, `convergence.json` (calibration used downstream) |
| Q3 | How do the Weibull modulus and the spatial correlation of strength change crack onset, crack count and fragment size, and their scatter? | `weibull_study.py` | onset CV vs m, seed bands, fragment distributions |
| Q4 | Can fragment size be predicted in closed form from a heating/cooling history, geometry and material properties? | `campaign.py` → `fit_pysr.py` → `predict_fragments.py` | dataset, Pareto front, model cards, predictor CLI |
| Q5 | What does the converged crack pattern look like at high resolution, including cool-down cracking? | `highres.py` + `crackmesh.py` | hero run, crack-aware meshes, restart for Q6 |
| Q6 | Does a cracked pellet seed cracks in its neighbours, and how do pressure, friction and contact area control this? | `stack_study.py` | sympathetic-cracking metrics vs factors |

## 2. Model (pellet_pd v2)

* **Constitutive model:** 3D ordinary state-based peridynamics, Linear Peridynamic Solid. It allows arbitrary Poisson ratio (UO₂ ν ≈ 0.32).
* **Thermal strain:** carried in the bond extension through the bond-mean temperature, so uniform heating is stress-free.
* **Temperature field:** prescribed steady radial profile T(r) = Ts(q) + q/(4πk)(1 − r²/R²), driven by a power history. Optional axial pressure enters as a third history column.
* **Creep:** viscoelastic LPS (Mitchell 2011). A deviatoric back-extension relaxes with an Arrhenius time constant τ(T), integrated semi-implicitly. This is what makes cool-down cracking appear: the hot core relaxes at power and goes into tension when cooled. Without creep, a cracked pellet only unloads elastically.
* **Fracture:** irreversible bond breaking when the stress-carrying stretch exceeds s_c.
  * Load stepping is quasi-static by adaptive dynamic relaxation.
  * Within each step, only the most critical bonds are broken (ratio ≥ κ·max). The body is then re-equilibrated, repeating until no bond exceeds s_c.
  * This yields discrete cracks. The two simpler alternatives — breaking inside the relaxation, or breaking all overstretched bonds at once — were tested in v1 and produce spurious diffuse damage.
* **Crack-face contact:** broken bonds still resist compression.
* **Stacks:**
  * Separate bodies, joined by column-aligned contact pairs with penalty normal stiffness and Coulomb friction (return mapping).
  * Uses a small-sliding assumption: sliding is µm against a particle spacing of about 0.1 mm.
  * Bottom face pinned in z; nominal pressure applied on the top face.
  * Contact area is controlled by the inner contact radius and by dish or chamfer geometry.
* **Stress recovery:**
  * Uses the non-local deformation gradient of the intact bonds.
  * Thermal strain is removed bond-by-bond inside the same fit. Subtracting α(T_i − T_ref) instead biases stresses by about 90 MPa in the core and several hundred MPa at the surface (found and fixed in v1).
  * Crack debris, and damaged particles with too little intact family left, report NaN.
* **Verified in v1** (same kernels): elastic mid-plane stresses converge to the analytic long-cylinder solution, with hoop RMS error of 6.4 → 4.5 → 4.4 % for nr = 8 → 12 → 16. The remainder is the peridynamic surface effect in the outermost layer.

## 3. The calibration problem — the key design decision

A single critical stretch cannot reproduce both tensile strength σ_f and fracture energy G_c, except at one horizon, the material length:

    δ_mat = G_c / (C (σ_f/E)²),    C = 3μ + (3/4)⁴ (K − 5μ/3)

For UO₂ (G_c ≈ 10 J/m², σ_f ≈ 130 MPa), δ_mat ≈ 0.1 mm. That means R/δ ≈ 40 and about 15 M particles per pellet, which is affordable for one hero run but not for a 640-run campaign. Two calibrations are therefore implemented everywhere.

| Mode | s_c | What converges as δ → 0 | Use |
|------|-----|-------------------------|-----|
| `strength` | σ_f/E, with a Weibull **grain field** of fixed physical size (`--wcorr`) | initiation; G_eff = C s_c² δ is recorded | default for the campaign (thermal cracking is initiation-controlled) |
| `energy` | from G_c at δ | nothing at fixed m: onset ∝ δ^−½ (no length scale) | comparison, physical δ only |

The Weibull scatter is defined on a mesh-independent grain field rather than per bond. Per-bond scatter changes the weakest-link statistics when the mesh is refined, which would contaminate the convergence study. `mesh_convergence.py` runs both modes and both refinement families, and writes the recommended (mode, δ, m) to `derived/convergence.json`. Every later study reads it.

## 4. Studies

Run counts below describe the **full** designs. The planners default to the lean profiles in §6, which keep every decision and cost about 1/6.

Run order and gates. Each arrow requires the previous gate to pass.

    build ─► bench (Q1) ─► meshconv (Q2) ─┬─► weibull (Q3)
                                           ├─► campaign (Q4) ─► fit_pysr ─► predict
                                           └─► highres (Q5) ─► stack (Q6)

### 4.1 Benchmark (Q1)
* **Case:** one fixed fracture case (nr = 24, 15 steps), run on 6 Spot machine types × {all vCPUs, physical cores}.
* **Metric:** pounds per 10¹² bond evaluations, computed from measured throughput and the Spot price per vCPU-hour.
* **Gate:** pick the cheapest machine, then update `config.COST`. All planner cost estimates depend on this.

### 4.2 Mesh convergence on crack area (Q2)
* **m-family:** δ fixed, m = δ/dx ∈ {2, 2.5, 3, 4, 5}.
* **δ-family:** m = 3, nr ∈ {8, 12, 16, 24, 32, 48}.
* Both families are run in both calibration modes, with 4 seeds each (88 runs).
* **History:** heating ramp to 30 kW/m in 1 kW/m steps, no creep, to isolate discretisation.
* **Metrics:** crack area A(q*) at q* = 5…30 kW/m, onset power, radial cracks, final fragment count, and L_eq = 3V/A.
* **Crack area definition:** a plane through intact material cuts Σ x V V = πδ⁵/5 per unit area, so A = 5/(πδ⁵) Σ_cut x V V. This is exact for a planar crack and independent of s_c, so it is comparable across meshes.
* **Analysis:** Celik et al. (2008) GCI on the three finest levels, with iterative order for non-uniform ratios and Richardson extrapolation. A seed-noise test checks whether the finest-two difference is ≤ 2 combined standard errors of the mean.
* **Gate:** the coarsest m-level within 10 % of the extrapolated A(20 kW/m) is written as the calibration.

### 4.3 Weibull distribution (Q3)
* **Design:**
  * m ∈ {∞ (no scatter), 3, 5, 8, 12, 20, 40}
  * Correlation ∈ {per bond, 0.15 mm, 0.40 mm grains}
  * 10 seeds
  * Two scenarios: a 40 kW/m ramp, and a heat-hold-cool cycle with creep
  * 420 runs in total
* **Outputs:** onset-power mean, CV and a Weibull fit, compared against the input strength CV. Also bootstrap CIs, seed bands of A(q) and crack count, and full fragment-volume distributions.
* **Question answered:** how strongly does microstructural scatter, rather than the mean strength, set crack count and fragment size?

### 4.4 Parametric campaign and symbolic regression (Q4)
* **Design:** scrambled Sobol, 512 training + 128 independent validation cases.
* **Variables and assumed distributions** (21 dimensions, in `campaign.py:DISTS`; replace them with measured distributions):

  | Group | Variable | Distribution |
  |-------|----------|--------------|
  | Geometry | R | 3.5–5.0 mm |
  | | H/D | 0.8–1.6 |
  | | dish radius | 0–0.75 R |
  | | dish depth | 0.1–0.4 mm |
  | Material | E | N(200, 8) GPa |
  | | ν | 0.30–0.33 |
  | | α | N(10, 0.5) ×10⁻⁶/K |
  | | k | 2.5–4.5 W/m/K |
  | | σ_f | LN(130 MPa, 0.2) |
  | | G_c | LN(10, 0.3) |
  | | Weibull m | 6–16 |
  | Creep | Q | 350–550 kJ/mol |
  | | τ(1700 K) | 0.2–10 h |
  | | surface coupling | 0.012–0.025 K·m/W |
  | History | peak power | 10–45 kW/m |
  | | cycles | 1–4 |
  | | ramp rate | 1–100 kW/m/h |
  | | hold | 0.1–200 h |
  | | cool-to fraction | 0–0.6 |
  | | scram probability | 0.3 |
  | Microstructure | seed | random |

* **Features** (`features.py`, one implementation shared by dataset, fit and predictor). These are dimensionless groups:
  * Π_T = αEΔT₀/((1−ν)σ_c): peak thermal stress over strength
  * Π_R = R/l_ch: size over Irwin length
  * Π_H: aspect ratio
  * Π_dish: dish volume fraction
  * ν and Weibull m
  * Π_creep = log₁₀(t_hot/τ(T_c,max)): relaxation achieved at power
  * cycle count, largest power drop, and number of shutdowns
* **Targets:** log₁₀ of fragment count, of volume-weighted d_eq/R, and of L_eq/R, plus specific crack area.
* **Fit:**
  * PySR 2.x on the training split.
  * Every Pareto equation is scored on the validation split.
  * The simplest equation within 5 % of the best validation RMSE is selected.
  * A gradient-boosting baseline shows how much accuracy the closed form gives up.
* **Model card:** each JSON carries the expression, features, residual σ, training-domain bounds and calibration.
* **Predictor:** `predict_fragments.py` takes a user's history file, geometry and material. It returns an estimate with a ±1σ band and warns on extrapolation.

### 4.5 High resolution (Q5)
* **Protocol:** heat to 30 kW/m over 3 h, hold 48 h, cool over 3 h, with creep on.
* **Hero run:**
  * nr = 64, m = 3: about 2.0 M particles and 0.34 G bonds (from `--dryrun`), 8 GB.
  * Runs on one c3d-highcpu-180 Spot VM with checkpoints every 2 steps.
  * State dumps every 5 steps feed `crackmesh.py` and the mid-plane montage.
* **Stack reference run:** uses the campaign calibration and writes `restart.bin`, which is the cracked pellet for Q6.

### 4.6 Sympathetic cracking (Q6)
* **Stack:** half pellet (pinned) | cracked pellet (from restart, with rigid translation removed) | half pellet (pressed). The sequence is preload, heat to 30 kW/m, hold 24 h, cool.
* **Factors:**
  * pressure {0, 2, 10, 30} MPa
  * friction μ {0, 0.2, 0.5, 1, stick}
  * nominal contact fraction {1, 0.75, 0.51, 0.28}
  * 2 seeds for the half pellets
  * a matched **intact-middle control** for every combination
* **Size:** 240 runs (`--design oat`: 54 runs for a first pass).
* **Metrics:**
  * Crack area per body at peak and at the end, and new crack area in the cracked pellet.
  * **Sympathetic excess:** half-pellet crack area minus the matched control.
  * **Cross-interface alignment:** Pearson r of the face damage profiles, plus an enrichment ratio. The ratio is P(half-face crack within ±6° of a middle-face crack mouth) divided by the circumferential fraction within ±6° of a mouth: 1 means independent, > 1 means seeded.
  * Interface state: pressure, slip fraction and mean slip.

## 5. Data management and sharing

**Every output is self-describing.**
* `manifest.json` records all inputs, derived quantities, the full load schedule, counts, the solver version, the **build id** (sha256 of the source), and `--label` tags (study, case_id, split, every factor).
* `summary.json` is written only on completion (`"complete": true`).
* CSV column names carry their units (`_mm`, `_MPa`, `_Wpm`, `_h` …).
* `schema.json` (generated from `schema.py`) documents every file and column.
* Every derived table has a `.meta.json` sidecar listing columns, units, description, producing script and source cases. Figures are made only from derived tables, so anything can be re-plotted without re-running.

**Case ids** take the form `<prefix>-<index>-<hash of solver arguments>`, so re-planning the same design is idempotent and a case id identifies its inputs.

**Bucket layout** (`study_common.py` docstring):

    gs://<bucket>/pellet-pd/v2/
      _meta/schema.json  _meta/solver/<build>/  _meta/env/
      benchmarks/<date>/
      studies/<study>/
        study.json  DATASET_CARD.md  cases.csv  catalog.csv
        histories/  inputs/  jobs/
        runs/<case_id>/      raw outputs (manifest, summary, history, cycles, fragments,
                             profile_*, angular_*, interface_*, state_*.bin, restart.bin)
        derived/             tidy tables + .meta.json (+ parquet)
        figures/

* **Sync commands:** `study_common.py push|pull|catalog|publish-meta` wrap `gcloud storage rsync`. `pull --what small` skips the binaries and is enough for every analysis.
* **Lifecycle** (`cloud/lifecycle.json`): leftover checkpoints are deleted after 14 days; state and VTK binaries move to Nearline after 30 days.
* **Sharing:** each study has a `DATASET_CARD.md`. Fill in the licence and citation, then share the study prefix (or `derived/` alone) read-only.

## 6. Cloud execution (Google Cloud)

* **Machines:** Spot VMs on AMD families: c3d-highcpu-30 by default, with c4d, t2d and n2d as the alternatives the benchmark compares.
  * The workload is many independent single-node OpenMP runs, so H4D (RDMA) adds nothing until the solver gets MPI.
  * Use about 30-vCPU tasks for the studies and one large VM for the hero run.
* **Orchestration:** Cloud Batch task arrays (`jobs/batch_job.json`), one line of `jobs/jobs.tsv` per task. `cloud/task.sh` then:
  * compiles the solver with `-march=native` for whatever CPU the VM has;
  * stages inputs and pulls any earlier attempt;
  * runs with `--resume`;
  * syncs to the bucket every 10 min and on SIGTERM;
  * on success, deletes the checkpoint and mirrors the run.
* **Preemption and retries:** the solver writes atomic checkpoints every `--ckpt-every` steps. On SIGTERM it aborts the current step and exits 75; Batch retries on 75 and on 50001 (VM preempted).
* **Region (cheapest first).** Spot prices differ by more than 2× between regions. Batch work doesn't care where it runs, so `config.REGION` defaults to the cheapest region found.

  | Machine (Spot) | Cheapest region | £/vCPU-h | us-central1 £/vCPU-h |
  |---|---|---|---|
  | c3d-highcpu-30 | australia-southeast1 | ~0.0032 | ~0.0071 |
  | t2d-standard-32 | europe-southwest1 | ~0.0055 | – |
  | c4d-highcpu-32 | us-west4 | ~0.0059 | ~0.012 |
  | n2d-highcpu-32 | asia-southeast1 | ~0.0065 | – |

  These are October 2026 values from a third-party tracker, converted from Google's USD list prices at
  `config.GBP_PER_USD` (0.75), and Spot prices change up to daily. `bench.py` runs each family in its cheapest region, then recommends a (machine, region) pair from the measured throughput.
* **Bucket co-location:** create the bucket and the Artifact Registry image repository in the **same region** as the compute. Then nothing crosses regions, and you pay nothing for transfer. Pull only `--what small` (CSV/JSON) to your desk; the binaries stay in the bucket.

### Cost-saving measures built in

* **Lean profiles by default.** Every planner has `--profile lean|full` (or equivalent options). Lean keeps each study's decision intact and drops the redundant levels.
* **Staged campaign.**
  * Stage 1 is 128 training + 32 validation runs.
  * `fit_pysr.py --learning-curve` refits on nested 25/50/100 % subsets.
  * `campaign.py plan --extend` adds 128 + 32 more from the same Sobol sequence, and **only if** the last doubling still cut the validation error by more than 10 %.
* **Lean numerics** (`--numerics lean`): 2 kW/m load steps and coarser creep sub-steps, about half the steps. Check against `--numerics fine` on a handful of cases before relying on it.
* **Looser tolerance between crack cycles.** The solver relaxes at `--tol-cycle` (3e-3) between bond-breaking cycles, then once more at `--tol` (1e-3) for the final equilibrium of each step.
* **Smaller hero run:** nr = 48 by default, about a third of nr = 64's cost. Use `--nr-hero 64` for the full one.
* **One-factor-at-a-time stack design first:** 54 runs. Run the 240-run factorial only where the first pass shows interactions.

**Planner estimates.** These use `config.COST`: 0.2 G bond-evaluations/s per vCPU, about 110 ADR iterations per step per particle across the radius, and c3d Spot at £0.0032/vCPU-h. Expect ±2–3×, and replace them with benchmark numbers.

| Study | Lean: runs | vCPU-h | £ | Full: runs | vCPU-h | £ |
|-------|-----------|--------|-----|-----------|--------|-----|
| meshconv | 42 | ~1,300 | ~£4 | 88 | ~3,400 | ~£11 |
| weibull | 40 | ~530 | ~£2 | 420 | ~5,600 | ~£18 |
| campaign | 160 (stage 1) | ~4,100 | ~£13 | 640 (fine numerics) | ~32,000 | ~£101 |
| highres (hero + stackref) | 2 | ~290 | ~£1 | 2 | ~820 | ~£3 |
| stack | 54 | ~2,300 | ~£7 | 240 | ~10,400 | ~£33 |
| **Total** | | **~8,500** | **~£27** | | **~52,000** | **~£165** |

Each campaign extension stage adds about £13. The benchmark costs a few pounds. Storage is minor: the campaign keeps summary data only (use `--keep-state` for dumps), and the hero state series is about 0.2 GB per dump at nr = 48.

## 7. Limitations and risks

* **Prescribed temperature:** cracks do not alter conduction, and there is no gap conductance, burnup, swelling, densification or relocation.
* **Creep model:** this is linear viscoelastic relaxation, not UO₂ power-law creep. The parameters are assumed.
* **Material distributions:** these are assumptions. The fragment model is only as good as the distributions and the calibration (§3). Use it inside its training domain; the predictor warns otherwise.
* **Fragment definition:** components of intact bonds among particles below a damage cut. Arrested cracks do not separate fragments. `crackmesh.py` gives an independent mesh-based count, and the two should be compared.
* **Contact:** small sliding, column-aligned pairs, so large relative rotations of fragments are not represented.
* **Lattice anisotropy:** cracks prefer 45°/90° at coarse resolution. Grain-field scatter and refinement reduce this, and `--jitter` is available.
* **Scale:** shared memory only; an MPI domain decomposition would be needed for δ = δ_mat at full pellet scale.

## 8. Reproducing

    ./build.sh                                   # local build with build id
    docker build -f cloud/Dockerfile -t <AR>/pellet-pd:2 .     # cloud image
    python bench.py plan --bucket B --image <AR>/pellet-pd:2   # then submit, analyze
    python mesh_convergence.py plan --bucket B ...             # etc. - see README.md

References: Silling et al., J. Elasticity 88 (2007) — state-based PD and LPS; Mitchell (2011), Sandia report — non-local ordinary state-based viscoelasticity for PD; Foster, Silling & Chen (2011), Int. J. Multiscale Comput. Eng. — energy-based failure criterion for peridynamic states (critical stretch for LPS); Kilic & Madenci, Theor. Appl. Fract. Mech. 53 (2010) — adaptive dynamic relaxation; Celik et al., J. Fluids Eng. 130 (2008) — GCI; Cranmer, arXiv:2305.01582 — PySR.
