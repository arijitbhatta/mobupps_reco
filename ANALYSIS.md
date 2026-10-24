

---

## Data profile

Before the model, the headline facts about the dataset, all computed from the
three CSVs.

Volumes:

| Table | Rows | Notes |
|---|---|---|
| users.csv | 10,000 | but 10,244 distinct users appear in interactions (244 extra) |
| apps.csv | 500 | but 685 distinct apps appear in interactions (185 extra) |
| interactions.csv | 490,382 | 311,849 distinct (user, app) pairs |

Event mix (the funnel, computed over all events):

| event_name | count | share |
|---|---|---|
| af_view | 206,764 | 42.2% |
| af_click | 107,232 | 21.9% |
| af_install | 24,638 | 5.0% |
| af_open | 97,862 | 20.0% |
| af_complete_registration | 19,610 | 4.0% |
| af_level_achieved | 9,724 | 2.0% |
| af_purchase | 24,552 | 5.0% |

So the log is top-heavy: 64.0% of events are `af_view` + `af_click`, which
matters a lot for how the evaluation reads (a popularity list scores cheap hits
on those). The acquisition-and-beyond events (install through purchase) are
36%, and that's what the pointwise label targets.

Shape and sparsity:

- Density is 4.4%: 311,849 observed pairs out of 7.0 million possible.
- Users are shallow: median 22 interactions but only 20 distinct apps, so ~1.1
  events per (user, app). There is almost no repeat/sequence signal.
- Apps are a long tail: median 389 distinct users per app, but 245 apps are
  touched by exactly one user (and those 245 are the ones missing from
  apps.csv). The top app reaches 2,217 distinct users.
- 1,468 users have fewer than 5 interactions, so they're cold by the harness'
  definition and never enter the test set.

Platform is a near-hard constraint:

| dimension | android | ios |
|---|---|---|
| users | 5,474 | 4,526 |
| apps | 288 | 212 |

98.7% of interactions are same-platform (6,182 cross-platform rows, 1.3%).
This is why the candidate filter and the negative pool both restrict to a
user's platform, and why `is_same_platform` contributes little to the tree
after that filter.

Other things worth knowing: app status is 379 active / 66 blocked / 55 paused;
the age buckets peak at 25-34 (3,460 users) then 35-44 (2,520); countries are
US / UK / DE / IN / FR heavy; 30 rows have a null country_code. The log spans
149 days (2026-01-30 to 2026-06-29), not the ~90 the docs claim.

The actionable takeaway from all of this: it's a small, sparse, tabular,
long-tail-heavy implicit-feedback corpus, platform-gated, with a strong prior
that "same platform" and "already-liked category" dominate. That's the shape
that points at a feature-based GBDT rather than a latent model (§5.2), and the
245 single-touch apps are a hard ceiling on any retrieval metric (§5.3).

### Filtering and imputation

The data is mostly clean, so the preparation is light, but it does exist and
should be stated rather than hidden in a default. Filtering (exclusions): 66
`blocked` apps are never scored and never sampled as negatives (55 `paused`
apps are kept, that's a campaign state not an app state); negative sampling is
restricted to same-platform, never-touched apps; and any unparseable
`event_time` would be dropped (with `format="mixed"` none are, the 493
no-microsecond rows are recovered, not discarded). Cross-platform interactions
(6,182 rows, 1.3%) are kept, not removed, and encoded as `is_same_platform`.

Imputation (fills, no rows removed): the 30 null `country_code` values become
the `"<UNK>"` sentinel; the 245 metadata-less apps get `"<UNK>"` category where
one is needed (they're never recommended, but their events still count in user
aggregates); `total_ratings` is stored as `log1p(...)` with a 0-fill if missing
(defensive, none are); popularity for never-seen apps is 0. `duration_seconds`
(9,807 null) is deliberately neither imputed nor used, since the nulls sit
on the highest-value open/post-install events and filling them would fabricate
signal precisely where it matters most.

---

## 5.1 Target and signal definition

The log has seven event types that form one acquisition/engagement funnel:

```
af_view -> af_click -> af_install -> af_open -> af_complete_registration -> af_level_achieved -> af_purchase
```

The thing that's easy to get backwards is that `af_open` is a post-install
event: the user is launching an app they already installed. The funnel splits
into two at a single acquisition barrier, between `af_click` and `af_install`.

So my positive signal is `event_name` at or past `af_install`: the set
{install, open, complete_registration, level_achieved, purchase}. The reason is
the product. MobUpps monetises acquisition and engagement, and "apps the user is
likely to engage with next" means, in practice, apps they will install and then
use, not apps they will scroll past. A model that maximises views is optimising
for impressions, not for the action downstream of the impression. That's why the
line sits at install and not at click.

Do weaker events count? It depends on the variant, and I want that to be
explicit rather than a hidden default:

- Pointwise (binary): no. `af_view` and `af_click` are negative, because the
  user was exposed and did not install. The signal is just "did this (user, app)
  pair cross the barrier or not." Post-install events are positive and weighted
  more heavily than a bare install (install < open < registration < level <
  purchase), so getting a deeper event wrong costs more.
- Pairwise and listwise: yes, but down-weighted. `af_view` is a low-grade
  document (grade 1) rather than a negative, since ranking objectives need the
  ordering view < click < install < open < ... to learn the funnel as a
  preference order instead of a binary.

Unobserved pairs. There are no explicit negatives in this data, nobody ever
says "I dislike this app." An unobserved pair is unknown, not disliked, but the
objectives need negatives, so I sample unobserved pairs as synthetic
negatives (grade 0 / label 0).

What that assumption costs is the biggest risk in the offline setup. Unobserved
doesn't mean rejected, a user may just never have been shown the app. Sampling
unobserved pairs as negatives:

1. suppresses cold and long-tail items. The 245 apps touched by exactly one user
   end up looking "rejected" by the other 10,000 users, when really nobody was
   ever exposed to them.
2. biases the model toward popular apps, because those are the only apps it
   reliably sees as positive.

I soften this two ways. The negative-sampling ratio is a tunable, not a fact
(§5.4). And I exclude from the negative pool any app the user touches anywhere in
their observed history, so a held-out future install can never be mislabelled as
a negative during training. That over-excludes some plausible negatives, which
is the conservative direction, but it removes the poisoning bug. I'd rather
state the residual bias than pretend a clean answer exists.

---

## 5.2 Model choice, and the one I rejected

Chosen: LightGBM, in three objective variants over one shared feature pipeline.

The justification is specific to this data, not "GBDTs are good":

1. The signal is tabular, not latent. Platform, category, age-rating,
   popularity, recency, country affinity are all strong predictors and they are
   explicit columns. With ~685 items and ~10,000 users there isn't much for a
   latent model to discover that I can't state directly as a feature. GBDTs are
   built for heterogeneous, mixed-type, mostly-categorical tabular data of
   exactly this shape.
2. The hard constraints are the first win. 98.7% of interactions are
   same-platform. A tree learns the `is_same_platform` interaction and
   category-affinity interactions on its own, so I don't need to hand-wire a
   latent model plus a pile of rules to capture the cheap wins.
3. Cold-start is a feature, not an edge case. A cold app still has category,
   rating and size columns a tree can score; a cold user still has their
   profile. The long tail (245 single-user apps, ~10k users at a median of 20
   distinct apps) dominates this data, and it's exactly where content features
   earn their keep.
4. The data is small and shallow: ~490k events, ~1.1 events per (user, app).
   There's no sequence structure to model, and not enough data to train a neural
   net without overfitting it.

I built three objectives (binary / rank_xendcg / lambdarank) over the same
features because the question I actually wanted answered wasn't "which model
family" but "does the training objective matter here." The result is that
pointwise leads, and it's the only variant that beats both the popularity and
the country x age x platform segment-popularity baselines at default capacity.
That tells me the label discipline (install-inclusive, funnel-weighted) matters
more than the choice of ranking loss. The ranking objectives overfit the graded
loss faster than the binary one does (§5.4), which is itself a finding I can
stand behind.

Rejected: latent-factor and embedding models, both matrix factorisation
(ALS/BPR) and deep sequential recommenders (GRU4Rec, SASRec, two-tower).

- MF at 10k x 685 with 4.4% density and a median of ~20 apps per user doesn't
  have enough per-user co-occurrence for latent factors to find anything that a
  few tabular features can't state outright. It would also smooth the long tail
  away, and it would need the same content warm-start I can fold into features
  for free. A GBDT gets the catalog and the interaction signal in one objective,
  with none of MF's cold-start machinery.
- Deep sequential and two-tower models need sequence signal this data doesn't
  have (~1.1 events per pair), an item set much bigger than 685 to avoid
  overfitting, and they can't handle unseen items or users without extra work,
  which is precisely where the wins in this data are. "It's state of the art"
  isn't a reason; it's the trap this question exists to catch.

---

## 5.3 Critique the evaluation

Recall@10 and NDCG@10 on the provided split answer one narrow question: does the
model surface distinct apps the user re-engages with. They don't say whether
it's good for MobUpps, in a few specific ways.

1. They ignore engagement depth. The harness flattens things to a distinct-app
   set (`groupby("user_id")["app_id"].apply(set)`), so retrieving an app the user
   viewed scores the same as one they purchased. A model that nails installs but
   misses views can score lower than one that's great at views, even though
   installs are the business target. This is why my pointwise model (which
   optimises install-or-beyond) differs from the ranking variants, and why a
   lower aggregate Recall on it wouldn't automatically mean it's worse.
2. No business-funnel weighting. Recall/NDCG never touch install rate, revenue,
   retention or impression load, which are the numbers that actually decide a
   rollout (§5.5).
3. A hard recall ceiling. 245 of the 685 interacted apps are touched by exactly
   one user and are missing from `apps.csv`. No model can recommend an app with
   no metadata that it has never co-occurred with, so roughly a third of the
   catalog is unreachable, and that caps any headline Recall regardless of model.

Risk in the split itself. The split is per-user leave-last-20%, so each user's
test events are interleaved globally with everyone else's train window. A user's
May test event can be "the future" relative to another user's February train
event, yet the popularity and recency features computed over train still see the
aggregate of everyone's events up to the global horizon. That's temporal
leakage: the model gets a popularity signal it could not have known at the
moment some of its users were acting. I quantified this by re-running on a
strict global time-cutoff split (train < T, test >= T), which removes the
interleaving.

Would I trust these numbers to decide a production rollout? No. They're a
sanity check ("does it beat popularity at retrieval") and nothing more. The
decision belongs to the online A/B in §5.5, which measures the thing MobUpps
actually cares about.

---

## 5.4 Hyperparameters: predict, don't list

The two that matter most for this model.

1. Tree capacity — `num_leaves` / `max_depth`, with `n_estimators` as the second
   half of the same knob.

   - What it controls: how much the model memorises versus generalises.
   - Offline: as capacity rises, Recall/NDCG climb to a plateau, then the model
     starts memorising one-off co-occurrences and overfits the long tail. Metrics
     fall, and they diverge further on the leak-free global split.
   - Real quality: too shallow and the model collapses to a logistic rule on
     popularity plus platform (no personalisation, but safe). Too deep and it
     over-specialises on rare category combos, treating the 245 single-user apps
     as lookalike anchors that produce non-generalising, low-engagement recs.
   - Failure at the extremes: a very deep tree ranks only the apps that happened
     to co-occur once in the long tail; a very shallow one ranks the top-20 apps
     for everyone. Both fail the discovery goal, in opposite directions.

2. The event-to-weight/tier mapping. The funnel weight scale (`sample_weight`)
   in pointwise, the graded `label_gain` tiers in ranking, plus the number of
   implicit negatives per user.

   - What it controls: how confident the model is that an unobserved pair is
     negative, and how much a strong event should outrank a weak one.
   - Offline: too few negatives or too flat tiers and the model is dominated by
     popularity priors, so Recall collapses onto the head apps. Too many
     negatives or too steep tiers and every unseen long-tail app is treated as
     disliked, which hurts Recall and ranking on cold items.
   - Real quality: the failure mode is miscalibrated confidence. Either the
     model confidently ranks the top-20 apps for everyone (no discovery) or it
     confidently ranks novelty junk. Both show up as an engagement or install
     drop that only the online A/B can catch, since offline metrics can't see
     calibration error.

I picked monotonic, funnel-ordered defaults (install=2 < open=3 < registration=4
< level=5 < purchase=6, negatives=1) and left them as parameters instead of
tuning them. The assignment explicitly warns against micro-tuning, and the
directional behaviour above is the defensible part.

---

## 5.5 From offline to online

Hypothesis: serving the recommender's ranked list increases downstream installs
(engaged acquisition) over the current popularity/rule baseline.

Primary metric: install rate per shown list, i.e. `af_install` and post-install
events per impression, or installs-per-session for acquired users. That's the
outcome MobUpps monetises next, and the one thing Recall can't measure.

Unit of randomisation: the user (device id), hash-bucketed, so one user's
exposure can't contaminate another's and the recommender's per-user output stays
coherent.

Sizing: power on the primary metric's expected lift against its historical
per-user variance. From this dataset's install-rate baseline, at roughly 2-5%
relative MDE, two-sided alpha=0.05 and beta=0.2, the required n would be
computed from the empirical variance rather than a rule of thumb; the variance
term dominates since installs are rare and heavy-tailed.

Two guardrails. First, latency / impression load: p99 serve time must not
regress, since a slightly worse but faster model is often the right business
call. Second, a top-funnel engagement or retention metric (uninstalls, DAU)
that must not drop while installs rise, so we don't trade installs for churn.

Single biggest threat to validity: novelty and position bias with carryover. The
new list changes what users are shown, so an early install lift can be a novelty
effect or a ranking-position artifact rather than genuine preference, and
exposed users may carry changed behaviour into later periods. Mitigations are a
holdout/ramp design and a read horizon long enough for novelty to decay, plus
reading the guardrails before the primary metric, not after.

---

## 6. Data-quality findings

| # | Finding | Action | Why |
|---|---|---|---|
| 1 | Two timestamp formats; pandas' default parser silently drops 493 no-microsecond rows (May-June) | Fixed (`format="mixed"`) | They were real data, not garbage |
| 2 | 245 users in interactions absent from `users.csv`, one event each | Flagged, kept | Real traffic; dropping them hides the cold-start problem |
| 3 | 245 apps in interactions absent from `apps.csv`, one user each (the same 245) | Flagged, kept, not candidates | No metadata, so no features; a recall ceiling to document |
| 4 | `af_purchase`/`af_complete_registration`/`af_level_achieved` all carry `interaction_type="open"` | Not fixed | Read depth from `event_name` instead |
| 5 | ~9.8k null `duration_seconds` (opens + post-install) | Ignored | Imputing would fabricate signal on the highest-value events |
| 6 | ~6.2k cross-platform interactions (1.3%) | Kept, encoded as `is_same_platform` | A small share may be real multi-device behaviour; deleting assumes an error we can't verify |
| 7 | 30 null `country_code` | Mapped to "unknown" | Minimal, honest imputation for a grouping key |
| 8 | Docs say "~90 days", log spans ~149 (2026-01-30 to 2026-06-29) | Flagged only | Don't bake a wrong window assumption into any feature |