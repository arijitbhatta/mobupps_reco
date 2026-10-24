# MobUpps — Recommendation Model (Take-Home)

A feature-based LightGBM recommender. Given a `user_id`, it returns a ranked
list of apps the user is likely to engage with next. This repo is the runnable
model plus the write-up; the longer reasoning lives in `plan.md` and
`ANALYSIS.md`.

---

## Setup & run

```bash
# from the repository root; Python 3.10+ (I developed on 3.14)
brew install libomp            # macOS only: LightGBM's OpenMP runtime

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q                    # 7 tests
python -m reco.data_to_report                 # data-quality report
python -m reco.evaluate --objective all       # headline evaluation
python -m reco.feature_importance --objective all

# exploratory analysis as a notebook
jupyter notebook scripts/data_analysis.ipynb
```

Data is auto-discovered at `./data`; override with `--data-dir /path/to/data`.

The model exposes the interface `scripts/evaluate.py` expects:

```python
from reco.model import AppRecommender
from scripts.evaluate import load_test_split, evaluate

train_df, test_df = load_test_split("data/interactions.csv")
model = AppRecommender(objective="pointwise")
model.fit(train_df)
model.recommend("u_000000", n=10)     # -> list[str], never crashes / never empty
evaluate(model, test_df, k=10)
```

---

## 1. Data preparation

Three tables, three different grains:

| Table | Rows | One row is | Columns I use |
|---|---|---|---|
| `users.csv` | 10,000 | a user | platform, age_bucket, country_code, signup_date |
| `apps.csv` | 500 | an app | platform, category, age_rating, status, size_mb, avg_rating, total_ratings, app_date_inserted |
| `interactions.csv` | ~490k | a single event | user_id, app_id, event_time, event_name |

The raw log is per-event, so a user can `view` and then `install` the same app
in separate rows. Prep does four things (`reco/data.py`, `reco/features.py`):

1. **Load** the CSVs. The timestamp column is parsed with `format="mixed"`
   because the log has two timestamp formats (with and without microseconds) and
   pandas' default parser silently drops the 493 no-microsecond rows. Annoying
   bug, cost me an afternoon.
2. **Validate** schema, primary keys, and referential integrity. This produces
   the data-quality report (8 findings, summarised in `ANALYSIS.md` §6). The
   ones that matter most: 245 users and 245 apps appear in interactions but not
   in their own tables (actually the same 245 single-touch items, which puts a
   hard ceiling on recall), and `af_purchase` / `af_complete_registration` /
   `af_level_achieved` are all written as `interaction_type == "open"`. So I read
   engagement depth from `event_name`, not `interaction_type`.
3. **Collapse to (user, app) pairs.** Each distinct pair becomes one row,
   labelled by the strongest event in that pair. This is the training grain.
4. **Sample negatives.** There are no explicit "disliked" rows, so I sample 5
   unobserved apps per user as label-0 negatives for training. The pool is
   restricted to same-platform, non-`blocked` apps the user has never touched
   anywhere. That last bit matters: it keeps a held-out future install from ever
   being sampled as a negative.

The funnel ordering that drives every label and weight:

```
af_view -> af_click -> af_install -> af_open -> af_complete_registration -> af_level_achieved -> af_purchase
```

Note `af_open` is post-install (you're opening an app you already installed), so
the acquisition barrier is between `af_click` and `af_install`.

---

## 2. Features and label

Every training row and every inference candidate is a 24-column vector over a
`(user, app)` pair. Same columns at train and serve time, no drift.

17 numeric features (`reco/config.py`):

| Family | Features | Source |
|---|---|---|
| User engagement | `user_n_events`, `user_n_distinct_apps`, `user_n_installs`, `user_install_rate`, `user_recency_days` | train events, per user |
| App static | `app_size_mb`, `app_avg_rating`, `app_log_total_ratings`, `app_age_days`, `has_metadata` | apps.csv + horizon |
| User static | `user_signup_age_days` | users.csv + horizon |
| Category affinity | `user_cat_affinity` | share of the user's train events in this app's category |
| Popularity priors | `app_pop_log1p`, `seg_pop_log1p`, `seg_age_pop_log1p`, `seg_cap_pop_log1p` | train events |
| Platform match | `is_same_platform` | user vs app platform |

7 categorical: `user_platform`, `user_age_bucket`, `user_country`,
`app_category`, `app_age_rating`, `app_status`, `user_pref_category`.

The label depends on the variant, but both are built from the same collapsed
pairs:

| Variant | Label | Weight |
|---|---|---|
| `pointwise` | binary `is_positive` = event in {install, open, registration, level, purchase} | `sample_weight` by funnel position: install=2 < open=3 < registration=4 < level=5 < purchase=6, negatives at 1. `view`/`click` are negative |
| `pairwise` / `listwise` | graded tier 0-7 (`view`=1 .. `purchase`=7), `label_gain=[0,1,3,7,15,31,63,127]` | implicit in the gain ladder; `view` is low-grade positive, not negative |

Leak rules: all counts and priors are computed on train events only, and time
features (`user_recency_days`, `app_age_days`, `user_signup_age_days`) are
measured from the train horizon (latest train timestamp), not the file's last
date.

---

## 3. Training and test data

The harness uses a per-user leave-last-20% split, not a random one:

```python
n_test = max(1, int(len(user_events) * 0.20))   # 6 events -> 1 held out
train  = events[:-n_test]                       # 5 oldest
test   = events[-n_test:]                       # the "future"
```

### Worked example: one real user

Here is `u_000000` (iOS, France, age 25-34) as it appears in the raw
`interactions.csv`:

| time | app | event | interaction_type | duration_s | source |
|---|---|---|---|---|---|
| 2026-04-06 13:21 | a_0241 | af_install | install | 70 | facebook_ads |
| 2026-04-11 19:41 | a_0278 | af_click | click | 6 | organic |
| 2026-04-22 02:22 | a_0379 | af_view | view | 11 | facebook_ads |
| 2026-04-23 02:23 | a_0061 | af_click | click | 55 | facebook_ads |
| 2026-04-28 16:36 | a_0069 | af_click | click | 24 | organic |
| 2026-04-29 10:19 | a_0092 | af_purchase | open | 123 | googleadwords_int |

The split cuts this list by count. Six events, so `n_test = max(1, int(6 *
0.20)) = 1`: the last event (`a_0092`, a purchase) is held out as test, the
first five are train.

```
train: a_0241  af_install
       a_0278  af_click
       a_0379  af_view
       a_0061  af_click
       a_0069  af_click
test:  a_0092  af_purchase   <- "the future" the model must predict
```

The train events then collapse to five `(user, app)` pairs, one per distinct
app, labelled by the strongest event in that pair:

| app | strongest event | pointwise label | pointwise weight |
|---|---|---|---|
| a_0241 | af_install | 1 | 2 |
| a_0278 | af_click | 0 | 1 |
| a_0379 | af_view | 0 | 1 |
| a_0061 | af_click | 0 | 1 |
| a_0069 | af_click | 0 | 1 |

Five sampled negatives (unobserved apps, label 0) get added to those five rows,
so this user contributes 10 rows to the training matrix. The model is evaluated
on whether `a_0092` shows up in its top-k for this user.

| Split | Interactions |
|---|---|
| Train | 396,677 |
| Test | 93,212 (8,776 test users) |

Users with under 5 interactions stay entirely in train; they're cold and go
through the segment-popularity fallback rather than the model.

There is no separate validation set here, only train/test. That's a constraint
of the provided evaluation and I left it as is rather than hiding it: nothing
gets tuned against test, so test doubles as the held-out evaluation. In a real
system I'd add a point-in-time validation split for hyperparameter choice, and
I'd be more careful about the per-user split's temporal leakage (a user's test
events are interleaved with other users' train events). That critique is in
`ANALYSIS.md` §5.3, along with a global time-cutoff check.

---

## 4. Modelling

LightGBM in three objective variants over one shared feature pipeline, plus a
country x age x platform popularity fallback for cold users.

| Variant | LightGBM objective | Loss scope |
|---|---|---|
| `pointwise` | `binary` + `sample_weight` | per (user, app) row |
| `pairwise` | `rank_xendcg` | per-user pair preference |
| `listwise` | `lambdarank` + `label_gain` | per-user list |

The fallback (`MIN_HISTORY_FOR_MODEL = 5`) ranks cold users by the top apps in
their country-age-platform segment, then global popularity. So `recommend`
never raises and never returns empty.

Hyperparameters (defaults used for every number in this README):

| Parameter | Value |
|---|---|
| `objective` | binary / rank_xendcg / lambdarank |
| `n_estimators` | 200 |
| `num_leaves` | 63 |
| `learning_rate` | 0.1 |
| `n_neg_per_user` | 5 |
| `label_gain` | [0,1,3,7,15,31,63,127] |
| `MIN_HISTORY_FOR_MODEL` | 5 |

The two that actually matter are tree capacity (`num_leaves` x `n_estimators`)
and the funnel weight/tier mapping with the negative count. I picked sensible
defaults and reasoned about their extremes in `ANALYSIS.md` §5.4 rather than
tuning them into the ground.

---

## 5. Feature importance

LightGBM native importance, gain = total loss reduction from splits on a
feature, split = how many times it was used. Default config, 200 trees.

| Rank | Pointwise | Pairwise | Listwise |
|---|---|---|---|
| 1 | `user_cat_affinity` | `user_cat_affinity` | `user_cat_affinity` |
| 2 | `user_install_rate` | `app_pop_log1p` | `user_recency_days` |
| 3 | `seg_cap_pop_log1p` | `user_n_installs` | `app_pop_log1p` |
| 4 | `user_n_installs` | `seg_cap_pop_log1p` | `seg_cap_pop_log1p` |
| 5 | `app_pop_log1p` | `user_n_distinct_apps` | `user_signup_age_days` |
| 6 | `app_category` | `user_install_rate` | `user_n_events` |
| 7 | `user_pref_category` | `user_recency_days` | `user_install_rate` |
| 8 | `user_country` | `user_signup_age_days` | `user_n_distinct_apps` |
| 9 | `user_recency_days` | `seg_pop_log1p` | `seg_pop_log1p` |
| 10 | `seg_pop_log1p` | `seg_age_pop_log1p` | `user_pref_category` |

Full tables (gain and split, per variant) via
`python -m reco.feature_importance --objective all`.

A few things worth noting:

- `user_cat_affinity` is #1 in all three. The most informative signal is
  whether the app is in a category the user already engages with. That's
  personalisation, not popularity.
- Behavioural aggregates beat app metadata. Install rate, install count,
  recency and popularity priors all outrank size/rating/age-rating. Content
  helps, but what the user has already done matters more.
- `is_same_platform` doesn't show up, which is expected: the hard platform
  filter removes cross-platform candidates before scoring, so the feature is
  near-constant and there's nothing for the tree to split on.
- Gain values aren't comparable across objectives (binary logloss and ranking
  losses are on different scales), so read within a column, not across columns.

---

## 6. Evaluation

Recall@10 and NDCG@10 on distinct held-out apps per user, averaged over 8,776
test users, k=10, every model returning non-empty. I also report an
acquisition-aligned recall restricted to install-or-beyond held-out apps,
because the business cares about acquisition, not retrieving a view.

Default config, `python -m reco.evaluate --objective all`:

**Retrieval:**

| Model | Recall@10 | NDCG@10 |
|---|---|---|
| Random | 0.0200 | 0.0224 |
| Popularity | 0.0790 | 0.0876 |
| Segment popularity | 0.1339 | 0.1508 |
| **Pointwise** | **0.1435** | **0.1798** |
| Pairwise | 0.1291 | 0.1621 |
| Listwise | 0.1227 | 0.1593 |

**Acquisition (install-or-beyond):**

| Model | Acq-Recall@10 |
|---|---|
| Popularity | 0.0805 |
| Segment popularity | 0.1396 |
| **Pointwise** | **0.1557** |
| Pairwise | 0.1382 |
| Listwise | 0.1377 |

Pointwise is the only model that beats both non-trivial baselines on both
metrics. The ranking objectives trail the segment baseline at this capacity,
and I don't think that's noise: at 50 trees all three beat the segment
baseline, at 200 only the binary pointwise still does. The ranking objectives
overfit the graded-funnel loss faster than the binary objective. That's exactly
the direction `ANALYSIS.md` §5.4 predicts for the capacity knob.

The aggregate recall margin also looks thin partly because ~64% of held-out
events are views/clicks, where the segment baseline scores cheap hits. The
acquisition column is the fairer comparison, and it's where the gap is widest.
None of this is treated as a rollout decision, for reasons spelled out in
`ANALYSIS.md` §5.3 and §5.5.

---

## 7. Conclusion

The model I'd ship is the pointwise variant: binary label at the install
barrier, funnel-weighted loss, segment-popularity cold start. It's the only one
that beats both baselines on both metrics at default capacity, and its dominant
feature is category affinity rather than raw popularity, so the win isn't just
"popularity with extra steps."

What offline can't decide: whether the ranking actually causes more installs in
production. Recall/NDCG on a leaky per-user split doesn't settle that. The
decision would come from the online A/B in `ANALYSIS.md` §5.5 (primary metric:
install rate per shown list; guardrails: latency and retention/uninstalls).

If I kept going, I'd add a point-in-time validation split, re-check on a strict
global time cutoff, and figure out how to lift the 245 single-touch,
metadata-less cold items, which cap offline recall no matter what model sits on
top.

---

## Repository layout

```
data/                       # mounted dataset (NOT committed)
scripts/
  evaluate.py               # provided harness
  data_analysis.ipynb       # exploratory data analysis
reco/
  config.py                 # funnel, weights, features, objectives
  data.py                   # load + validate + DQ report
  features.py               # (user, app) feature builder
  model.py                  # AppRecommender (fit / recommend / cold start)
  evaluate.py               # baselines, split, acquisition + slice metrics
  feature_importance.py     # §5 report
sql/
  user_features.sql
  user_app_training.sql
tests/test_reco.py
```

Notes for the reviewer: the dataset is not committed (assume `data/` is mounted
at the documented path); `.venv/`, `__pycache__/`, `.pytest_cache/` are ignored;
everything is in English.