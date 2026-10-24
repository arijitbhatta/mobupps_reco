

## 0. Problem statement

Given a `user_id`, return a ranked list of `N` apps the user is likely to engage with next.

Hard interface constraints (from `scripts/evaluate.py`):

- Model exposes `fit(train_df)` and `recommend(user_id: str, n: int) -> list[str]`.
- `recommend` must **never crash** for unknown/cold users; it must always return *something*
  (the harness scores an empty list as 0).
- `fit` receives **only the interactions DataFrame** — so app catalog and user profiles must
  be made available to the model another way (constructor injection / a data service), which is
  a deliberate design decision detailed in §3.

Success is **not** the offline number. It is: frame the problem correctly, pick a defensible
model, evaluate honestly, and reason about offline→online.

---

## 1. What the data actually is (exploration summary)

| Table | Rows | Key facts |
|---|---|---|
| `users.csv` | 10,000 | `platform` android 5,474 / ios 4,526; `age_bucket` complete; 30 null `country_code`; `signup_date` 2024‑05 → 2026‑05 |
| `apps.csv` | 500 | 288 android / 212 ios; `status` active/blocked/paused; ~60 apps never interacted |
| `interactions.csv` | 490,382 | 10,244 unique users, 685 unique apps, 311,849 unique (user,app) pairs, 150 days (2026‑01‑30 → 2026‑06‑29) |

Density & shape (these drive every modelling decision):

- **Sparse implicit feedback**: ~4.4% of the user×app matrix is filled.
- **Shallow per-user**: median 22 interactions but only **20 distinct apps** → ~1.1 events per
  app, so repeat/sequential signal is weak.
- **Heavy item long-tail**: median 389 distinct users per app; **245 apps touched by exactly one
  user** — and those 245 are exactly the apps missing from `apps.csv`.
- **Platform is a near-hard constraint**: 98.7% of interactions are same-platform.

Event types in their **funnel order** (weakest → strongest engagement):

```
af_view → af_click → af_install → af_open → af_complete_registration → af_level_achieved → af_purchase
```

| `event_name` | `interaction_type` | Count | Funnel position / strength |
|---|---|---|---|
| `af_view` | view | 206,764 | top-of-funnel (weakest) |
| `af_click` | click | 107,232 | top-of-funnel |
| `af_install` | install | 24,638 | **acquisition** — the "engage next" barrier |
| `af_open` | open | 97,862 | **post-install** engagement (app already installed → user opens it) |
| `af_complete_registration` | (mapped to `open`) | 19,610 | deep engagement |
| `af_level_achieved` | (mapped to `open`) | 9,724 | deep engagement |
| `af_purchase` | (mapped to `open`) | 24,552 | monetization (strongest) |

**Key implication for labeling:** `af_open` happens *after* `af_install` — it is a
post-install engagement event, not a pre-install discovery signal. So in §2.4 the "anything
≥ install" positive boundary splits the funnel at the *acquisition* barrier, and `af_open` sits
**with** the deep-engagement events (registration/level/purchase), not with clicks.

Note: `af_purchase`, `af_complete_registration`, `af_level_achieved` are all labelled
`interaction_type == "open"` in the raw data — a data-quality finding we must not lose when we
build label weights.

---

## 2. Data preparation plan

### 2.1 Loading & validation (data layer)

Build a `data_prep/` module (`load_config` / `validate` / `prepare`) that:

1. **Loads** the three CSVs from a configurable `data/` path (documented in README).
2. **Validates schema & referential integrity**, emitting a DQ report (see §2.3):
   - PK uniqueness (`user_id`, `app_id`).
   - FK references both directions (interactions → users, interactions → apps).
   - Type coercion (`event_time` → datetime with `errors="coerce"`, then drop/flag NaT).
   - Impossible/duplicate rows, cross-platform (user vs app) mismatches.
3. **Asserts the Model is fed a clean, joined, exploded-by-event training table** (see §3 on
   how the model gets catalog + profiles despite `fit(train_df)` receiving only interactions).

### 2.2 Cleansing decisions (per DQ finding, one line of justification)

| # | Finding (how we found it) | Action | Justification |
|---|---|---|---|
| DQ1 | 245 `user_id`s in interactions absent from `users.csv`, each with **1** event (likely id overflow, e.g. `u_1000046`) | **Flag + keep** in the interaction log; treat as cold users with no profile | They are real traffic; dropping them only hides the cold-start problem the task asks us to handle |
| DQ2 | 245 `app_id`s in interactions absent from `apps.csv`, each touched by exactly 1 user | **Flag + keep**; model them as a "no-metadata" cold item | They are a genuine recall ceiling (no features, no co-occurrence). Document, don't pretend to predict them |
| DQ3 | `af_purchase`/`af_complete_registration`/`af_level_achieved` all carry `interaction_type="open"` | **Don't fix**, but read strength from `event_name`, not `interaction_type` | Fixing the column rewrites history; using `event_name` is lossless |
| DQ4 | ~9,807 null `duration_seconds` (opens + post-install events) | **Ignore** as a feature; only flag | Duration is missing on the highest-value events; imputing would fabricate signal |
| DQ5 | 6,186 cross-platform interactions (~1.3%) | **Keep**, but weight/label honestly; add `is_same_platform` feature rather than deleting | A small % may be real multi-device behaviour; deletion assumes an error we can't verify |
| DQ6 | 30 null `country_code`; ~60 catalog apps never seen | Null-country → `"unknown"` category; unseen apps → cold-item path | Minimal, honest imputation |
| DQ7 | Doc says "~90 days" but data spans ~150 days (2026‑01‑30 → 2026‑06‑29) | **Flag only** | No action needed; just don't bake a wrong 90-day assumption into features |

### 2.3 Feature engineering plan

Three feature families, built **on the train slice only** (no test leakage):

- **User features** (from `users.csv` + train aggregates): `platform`, `country_code`,
  `age_bucket`, `signup_age_days` at prediction time, plus per-user aggregate stats computed
  from *train* interactions only — e.g. `n_events`, `n_distinct_apps`, `n_installs`, `preferred_category`
  (mode), `preferred_subcategory`, `install_rate` (installs / events), `recency` (days since last event).
- **App features** (from `apps.csv`): `platform`, `category` + `subcategory_*`, `size_mb`,
  `age_rating`, `avg_rating`, `total_ratings` (log-scaled), `status` (active/paused/blocked),
  `app_age_days` relative to prediction time. For the 245 metadata-less apps → a `has_metadata`
  flag and embedding-less cold-item handling.
- **Interaction / cross features** (in the training table):
  - **Per-(user, app) interaction weight** derived from `event_name`, used in the loss (see §2.4).
  - `is_same_platform` (user platform == app platform).
  - Category-affinity signals: e.g. user's event count per `category`, app's event count per country.
  - **Segment popularity priors**: global popularity, plus `country×platform`, `age×platform`, and
    `country×age×platform` popularity — the last one is also the cold-user fallback signal.
  - Optional light temporal signal (recency-weighted counts).

### 2.4 Positive / negative / weight construction (drives §5.1 of the write-up)

Signal definition is **model-variant-specific** (all three share one LightGBM feature pipeline):

- **Pointwise (binary classifier)**:
  - **Positive**: `event_name ∈ {install, open, complete_registration, level_achieved, purchase}`
    — i.e. **anything at or past the install/acquisition barrier** in the funnel. `af_open`
    is *post*-install, so it is a positive. Below the barrier (`view`, `click`) is **negative** —
    the model predicts *"will this user install (or already has installed and thus engage) this
    app next"*, not "will they weakly engage".
  - **Loss weight per (user, app) row** from `event_name`, **strictly following the funnel order**
    so a misclassification deeper in the funnel costs more than one earlier: `install < open <
    complete_registration < level_achieved < purchase`. Concretely `sample_weight` e.g. install=2,
    open=3, registration=4, level=5, purchase=6, and the negative class carries a floor weight (1)
    so negatives still shape the decision boundary.
  - **`af_view` is negative** in pointwise (not just unweighted — it does not count as a positive).
- **Pairwise / listwise (LightGBM rank-pair / lambdarank)**:
  - Each *user* is a query/group; *apps the user interacted with* are the candidate documents,
    ordered by the user's interaction (implicitly by strength / recency).
  - **`af_view` is down-weighted**, not dropped: it is a low-relevance document that still helps
    shape the *order* (views rank below clicks/installs/opens), rather than a negative label.
  - Document relevance tiers follow the funnel: `af_view < af_click < af_install < af_open <
    af_complete_registration < af_level_achieved < af_purchase` (never collapsed — the funnel
    order is preserved as distinct grades where practical). Unobserved candidate apps per user
    are sampled as grade-0 / low-relevance negatives.

- **Unobserved pairs (all variants)**: treated as **unknown, not negative**, but sampled as
  negatives/grades because the objectives need them. We document the cost: unobserved ≠ disliked,
  so naive negatives suppress long-tail/cold items that are merely *unseen* — a §5.1 assumption
  to defend, and one of the §3.4 hyperparameter axes.

### 2.5 Train/test split plan

- **Primary**: use the provided `evaluate.py` per-user leave-last-20% split, so numbers are
  directly comparable to the popularity baseline.
- **Also implement a stricter global time-cutoff split** (train on interactions with `event_time
  < T`, test on `>= T`) as a robustness check, because the per-user split has a known temporal
  leakage: a user's test events are interleaved *globally* with others' train events, so
  train contains "future" popularity signal relative to some test events (see §4 critique).
- Cold users (fewer than 5 events) stay train-only; they are handled by the fallback path, not scored.

### 2.6 Training-grain decision: (user, app) pairs, not events

The raw log is **(user, app, event)** grain — one row per event, so a user can emit view+click+
install on the same app. All three models train on a **flat (user, app)-level** feature matrix
(one row per pair):

- **Collapse rule**: group events by (user, app) → one row, where the label/weight is the
  **strongest** event in the pair (never a count/average):
  - *Pointwise*: label `= 1` if any event in the pair is ≥ install, else `0`; `sample_weight` =
    strongest funnel weight among the pair's events (install=2 < open=3 < registration=4 < level=5
    < purchase=6; negatives floor = 1).
  - *Pairwise/listwise*: relevance grade = strongest funnel tier in the pair (an install outranks
    a view on the same app). Rows are then grouped **by user** into query lists for training.
- **Synthesized negatives** (sampled unobserved pairs) are injected at the *same pair grain*, so
  the whole training table is uniformly (user, app)-level.
- **Difference in loss scope**: pointwise computes loss **per row** (pair-independent); ranking
  objectives compute it **per user-group** (the user's rows form one ordered list). The feature
  matrix is identical; only the grouping changes.

This means at prediction time candidate scoring is exactly the same pair-grain join, so the
training/serving grain is consistent end-to-end.

### 2.7 Handling users with long/spread-out histories (time anchoring)

The provided split is **count-based, not time-based** — it sorts each user's events and holds
out the last 20% *by count*, so a user active across a month (or the full 150-day span) is
handled purely by their event count and order, never their calendar span. That is fine for the
split, but it forces **time-anchoring discipline** on my features so the model can't peek at its
own answers:

- **(a) User aggregates = train-only.** `preferred_category`, `n_installs`, `install_rate`,
  `n_distinct_apps`, etc. are computed from events **before the user's holdout point** only —
  never from the held-out tail. Otherwise the feature vector contains the answer.
- **(b) Recency/app-age relative to an anchor time, not the file's last date.** "Days since last
  event" and `app_age_days` for a user scoring at point `t` measure from `t` (a user's last train
  event at serve time; the row's own timestamp at train time), not from 2026‑06‑29.
- **(c) Point-in-time features within train (default).** Each (user, app, t) training row gets
  user-aggregate features computed from events **strictly before t** (cumulative groupby), so a
  user's early rows don't borrow from their own future. Falls back to static train-only aggregates
  only if these misbehave, with the lookahead documented.
- **(d) Negative-sampling leak guard.** The "unobserved negative" pool for a user excludes **every
  app they touch anywhere in their observed history (train ∪ test)** — if a future install is app
  X, sampling X as "negative" because it isn't in train would teach the model to down-rank the
  very thing it should recommend. Conservative direction: over-excludes plausible negatives, but
  eliminates the poisoning bug.
- **(e) Warm/cold should be recency-aware.** A user with many events months ago and silence since
  is effectively cold at test time even if their lifetime count clears `< 5`. Flagged as a
  possible refinement to the fallback trigger; not necessarily implemented first.

---

## 3. Modelling plan

### 3.1 How the model sees catalog + profiles (given `fit(train_df)`)

`evaluate.py` calls `fit(train_df)` with interactions only. Design decision:

- The recommender is constructed with a **`DataService`** (catalog + user profiles + split
  config), either injected in `__init__` or lazy-loaded from `data/` paths stored on the instance.
- `fit(train_df)` then receives the interaction slice and *internally* joins profiles/catalog to
  build user & item feature vectors. This keeps the required protocol intact while giving the
  model access to side information.

### 3.2 Recommended model (primary)

**LightGBM, tried in all three objective variants, sharing one feature pipeline** (classical ML,
no latent-factor / embedding model):

1. **Hard-eligibility filter**: only apps on the user's platform are candidates (98.7% of signal
   agrees; nearly free Recall, removes garbage). Apps with `status != active` are demoted/excluded.
2. **Cold-user / weak-history fallback**: **`country×age×platform` segment popularity** for users
   with too little history to score (< ~5 train events). This is a *specific* fallback: the
   candidate set is the top apps within the user's country × age-bucket × platform segment.
   Never return an empty list.
3. **Personalized core — a LightGBM ranker, three variants over the same feature rows** (built per
   *(user, app)* from §2.3): the candidate apps for a user, ordered by predicted score. The three
   variants differ only in the **training objective**, not the features or inference:
   - **(a) Pointwise (primary candidate)**: binary classification, `positive = event_name ≥ install`
     (the acquisition barrier; `af_open` and beyond are positives). Each row carries a
     **`sample_weight` derived from the funnel order** (`install < open < registration < level <
     purchase`) so a misclassification deeper in the funnel costs proportionally more; `af_view`
     is negative here. Objective `binary`, tuned on Recall@10/NDCG@10.
   - **(b) Pairwise**: `objective=rank_xendcg` (LightGBM rank-pair loss). Query = user; documents
     = interacted apps ordered by interaction strength along the funnel
     (view < click < install < open < … < purchase); `af_view` **down-weighted** (low-grade
     document, not negative).
   - **(c) Listwise**: `objective=lambdarank`, `label_gain` from the graded funnel tier,
     `eval_metric=ndcg@10`. Query = user; documents = interacted apps graded by the funnel order
     + sampled grade-0 negatives; `af_view` **down-weighted**.
   We build the feature/ranking pipeline once, run **all three, report all three against the same
   baselines, and commit to one** as the shipped model with justification (this is the "try pairwise
   and listwise both" requirement, plus the pointwise build).

**Why this family, reasoned from *this* data** (not from popularity of the method):
- **The real signal here is tabular / structural, not latent.** Platform, category,
  subcategory, age_rating, popularity, recency and country affinity are all strong predictors —
  and GBDTs are unbeatable at squeezing exactly this kind of heterogeneous, mixed-type tabular
  feature space. With 685 items and ~10k users there is little room for a latent model to "learn"
  silently beyond what we can state explicitly as features.
- **The hard constraints are features, not priors.** A GBDT learns `same_platform`, `status`,
  and category-affinity *interactions* automatically; we don't have to hand-code a hybrid of a
  latent model + rules to get the easy wins first.
- **Cold-start is handled by content features natively.** A cold app with no co-occurrence still
  has category/rating/size features the tree can score — no warm-start machinery needed. Cold
  users fall back to the `country×age×platform` popularity path (they have no feature history),
  which is exactly the honest default.
- **Why three objectives instead of one:** the harness scores us on list order (NDCG/Recall);
  pointwise logloss optimizes per-pair calibration, while pairwise/listwise optimize order
  directly. Running all three isolates *whether the objective matters* for this data, and the
  weight/label discipline (§2.4) makes them comparable rather than an apples-to-oranges pile.

**Why not a pure latent competitor as the primary?** We could ship an item-item co-occurrence or
pure popularity model, but they cannot use app metadata or user profile at all, and would waste
the catalog we've been given. The GBDT ranker is the single model that uses *all three* tables —
users, apps, and interactions — with one coherent training objective.

### 3.3 Model I deliberately reject (for §5.2)

**Latent-factor / embedding models — both matrix-factorization (ALS/BPR) and deep sequential
recommenders (GRU4Rec, SASRec, two-tower ranking).** Rejected *for this dataset*:

- **MF**: at ~10k users × 685 items with a 4.4%-dense matrix and a median of only ~20 apps per
  user, there is too little per-user co-occurrence structure for latent factors to find anything
  a few tabular features can't state directly. MF would smooth the long tail away and would
  need the same content warm-start we can fold into features for free — so the GBDT gets the
  content *and* the interaction signal in one objective.
- **Deep sequential/two-tower**: ~1.1 events per (user,app) means almost no sequence signal; the
  item set is far too small to train a neural model without overfitting; and the wins in this
  data come from hard constraints + cold-start (which learned user/item towers can't capture on
  unseen items/users). ("It's SOTA" is not a reason — it's the trap §5.2 exists to avoid.)

### 3.4 Hyperparameters we will discuss (§5.4) — chosen, with directionality

For a LightGBM ranker (shared across the three variants), the two that matter most:

1. **Tree capacity — `max_depth` / `num_leaves` / `n_estimators` (we treat "capacity" as one
   knob)**: controls how much the model can memorize vs generalise.
   - *Offline*: as capacity ↑, train loss falls and Recall/NDCG rise to a plateau, then the
     model memorises one-off co-occurrences and overfits the long tail → metrics fall (and
     diverge from the global-cutoff re-check).
   - *Real quality*: too shallow → the model is near a logistic rule on popularity + platform,
     missing genuine personalization; too deep → it over-specialises on rare category-combos
     (the 245 single-user apps become pathological "lookalike" anchors), producing
     non-generalising, low-engagement recs.
2. **The `event_name → weight/tier` mapping (negative floor & sample_weight in pointwise; the
   graded `label_gain` tiers + negative-document count in pairwise/listwise)**: controls how
   confident the model is that an *unobserved* pair is negative, and how much a strong event
   outranks a weak one — the "unknown vs not-liked" and "install vs view" trade-off.
   - *Offline*: too few negatives / too flat tiers → dominated by popularity priors, Recall
     collapses toward head apps; too many negatives / too steep tiers → unseen long-tail apps
     get treated as "disliked", harming Recall and ranking on cold items.
   - *Real quality*: the extreme failure mode is *miscalibrated confidence* — either the model
     confidently ranks only the top-20 apps for everyone (no discovery), or it confidently
     ranks novelty junk; both show up as an engagement/install drop that only the online A/B
     (§5) can catch, because offline metrics can't see it.

---

## 4. Evaluation plan

### 4.1 What we run

- **Primary metrics**: Recall@10, NDCG@10 (as the harness defines them) on the provided split,
  reported against: (a) global popularity baseline, (b) segment popularity baseline, (c) random
  baseline. We report **coverage** (fraction of users with non-empty predictions) and the
  **recall ceiling imposed by the 245 one-touch apps / cold items**.
- **Variant comparison**: all three LightGBM objectives (pointwise / pairwise / listwise) on the
  same features and split, so the shipped-model choice is evidence-based, not a preference.
- **Robustness check**: same metrics on a global time-cutoff split, to quantify the leakage we
  believe exists in the per-user split.
- **Slice diagnostics**: metric by user-activity bucket (warm vs cold), by platform, by
  app-popularity decile (head vs long-tail) — to show *where* the model wins/loses, not just a
  headline number.

### 4.2 What this evaluation fails to capture (§5.3) — the critique we will write

- Recall/NDCG reward *retrieval of any held-out app*, not *engagement depth* or *install/revenue*
  — offline "wins" may be weak-event views the business doesn't care about.
- The per-user **split leaks temporally** (test events interleaved globally inside the train
  window → popularity/recency signal from "future" is available at train time).
- It **can't score cold items** (245 apps, ~1/3 of the catalog) — a hard ceiling on Recall that
  no model can beat, and it **ignores the business funnel** (we care about installs, but the
  label mix is dominated by views).
- **Would we trust it for a rollout decision?** No — it is a necessary but insufficient signal;
  it says "retrieves reappeared apps," not "drives installs." Decision gate must be the online
  test (§5). We state this plainly.

---

## 5. Experimentation plan (§5.5) — offline → online A/B

- **Hypothesis**: serving the hybrid recommender's ranked list increases *downstream installs*
  (engaged acquisition) vs. the current popularity/rule baseline.
- **Primary metric**: install rate per shown list (installs → post-install events per impression),
  or installs-per-session for acquired users — because that is the outcome MobUpps monetizes
  next; Not Recall.
- **Unit of randomization**: user (device-id), hash-bucketed; no cross-user contamination in
  recommender output.
- **Sizing**: powered on the primary metric's expected lift using historical per-user variance
  and a chosen MDE (we'll compute n from observed install-rate baseline, ~2–5% relative MDE,
  two-sided α=0.05, β=0.2).
- **Guardrails**: (1) **latency / impression-load** (p99 rec-serve time must not regress) and
  (2) **top-funnel engagement or uninstall/dau** — an engagement or retention guardrail that
  must not drop while installs rise.
- **Biggest validity threat**: **novelty/positioning bias & carryover** — the new list changes
  what users see (exploration), so early install-lift may be a *novelty effect* or a
  *ranking-position* artifact rather than genuine preference; mitigated by a holdout/ramp
  design and long enough read horizon.

---

## 6. Deliverables & repo structure (what we will produce)

```
README.md            # exact copy-paste setup/run commands
ANALYSIS.md          # the §5 five-question write-up, plus §6 DQ findings
plan.md              # this plan
data_prep/           # load, validate, prepare; DQ report; one or two .sql feature queries
models/              # recommender class (fit / recommend), fallback logic
eval/                # harness wrapper, baselines, slice metrics, global-time-split check
notebooks/           # exploration only (never the runnable path)
scripts/evaluate.py  # provided (used as-is for the headline number)
```

**SQL deliverables** (assignment asks for 1–2): we include, in `data_prep/`, the warehouse
queries that would build (a) the **per-user interaction-aggregate feature table** and (b) the
**candidate (user, app) feature join / training table** (e.g. `GROUP BY user_id` producing
counts per `event_name`, distinct apps, recency; and the user×app pair rows with
`SUM(CASE WHEN event_name=...)` strength-weighted counts + negative sampling flag).

---

## 7. Order of work (~5h budget)

1. **Data layer + DQ report** (validate, join, DQ table above) — ~1h.
2. **Feature pipeline + train/test both splits** — ~45m.
3. **Baselines first** (popularity / segment-popularity / random) to set the floor — ~20m.
4. **Model fit + recommend + fallback** (feature pipeline → LightGBM in all three objectives —
   pointwise / pairwise / listwise — then commit to one) — ~1.5h.
5. **Evaluation** (headline + variant comparison + slice diagnostics + global-cutoff check) — ~45m.
6. **WRITE ANALYSIS.md** (the five questions + DQ findings) — ~1h. This is the priority
   deliverable; if time runs short we cut tuning, never the write-up.

---

## 8. Open decisions to confirm before I build

1. **Pointwise label boundary = install-inclusive confirmed** (with `af_open` and beyond also
   positive, since open is post-install); the remaining open sub-choice is the exact
   `sample_weight` scale strictly following the funnel (e.g. install=2, open=3, registration=4,
   level=5, purchase=6, negatives=1). Same question for the pairwise/listwise relevance tiers.
   These are the §3.4 axis — I'll pick sensible defaults and expose them, not tune them into the
   ground.
2. **`af_view` role confirmed**: **negative** in pointwise, **down-weighted** (low-grade, not
   dropped, not negative) in pairwise/listwise.
3. **Cold-user fallback confirmed**: **`country×age×platform`** segment popularity as the floor.