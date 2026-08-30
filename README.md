# GymVibe — Context-Aware Hybrid Music Recommender 🏋️‍♂️

GymVibe is a two-stage recommender that selects music for a workout based on the current workout context and the user's preferred tempo.

The project extends my [Gym Vibe Archetype Classifier](https://github.com/sweeteri/gym-vibe-archetype-classifier): the original project classified tracks into seven workout-oriented music archetypes, while GymVibe uses these archetypes as context for personalized recommendations.

> **Data note:** the original dataset contains track features, but no real user interactions. For this experiment, I generate synthetic implicit-feedback logs with controlled dependencies on workout context and BPM. Therefore, the reported metrics are benchmarks for this experimental setup, not production metrics.

## How it works

```text
Workout context + preferred BPM ──► User Tower ─┐
                                                ├─ similarity ─► Top-K candidates
Track audio/text features ────────► Item Tower ─┘
                                                               │
                         context, genre, BPM, item features,
                              neural similarity score
                                                               ▼
                                                CatBoostRanker (YetiRank)
                                                               ▼
                                                        Final Top-N tracks
```

### Retrieval — Two-Tower / DSSM

The PyTorch model maps users and tracks into the same embedding space.

**User features:**

* preferred BPM
* workout archetype (one-hot)

**Track features:**

* BPM
* loudness (`gain`)
* lyric word count
* word density
* `Power Index`
* `Street Score`

The model uses an in-batch contrastive objective: the observed user–track interaction is treated as positive, while other items in the batch provide negatives.

The full catalogue is scored by embedding similarity and the top 50 tracks are passed to the ranking stage.

### Ranking — CatBoost

CatBoostRanker with `YetiRank` re-ranks the retrieved candidates using:

* neural similarity score;
* workout context;
* genre;
* preferred BPM;
* audio/text features.

`session_id` is used as the ranking group and `like` as the relevance label.

## Workout Archetypes

The seven contexts come from the original classifier project:

* Boss Fight / Doom Slayer
* Zen / Lo-Fi Recovery
* Golden Era (Old School)
* Pilates Girl
* Pure Rage / Metalhead
* Sigma Grindset
* Пацанский Вайб

## Features

A few custom features are used to describe the tracks:

* **Power Index** — simple energy proxy based on BPM and loudness.
* **Street Score** — count of predefined street-culture markers in lyrics.
* **Word Density** — lyric word count divided by track duration.

## Offline Evaluation

The notebook uses a **user-level train/test split**, so test users are not present during training.

Current experiment:

| Metric              |     Result |
| ------------------- | ---------: |
| Retrieval Recall@50 | **0.2781** |
| Ranking NDCG@5      | **0.5633** |
| Context Hit Rate@5  | **0.8450** |

**Recall@50** is evaluated against the full catalogue and measures how many positive tracks are retrieved by the Two-Tower model.

**NDCG@5** is evaluated on the displayed candidate sets and measures how well CatBoost orders those candidates.

**Context Hit Rate@5** is an additional sanity check showing whether the final recommendations match the requested workout context. It is not used as the main relevance metric.

Exact values may vary with the random seed, generated synthetic logs, and package versions.

## Run

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\activate

pip install -r requirements.txt

jupyter notebook recsys_songs_gym.ipynb
```

Run all notebook cells to generate the logs, train both stages, evaluate the model, inspect recommendations, feature importance, and t-SNE embeddings.

## Tech Stack

Python · PyTorch · CatBoost · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

## Limitations

This is a portfolio prototype rather than a production recommender.

The main limitations are synthetic feedback, a small catalogue, and the absence of real listening history.

A production version could use timestamped interactions, temporal splits, richer user histories, hard-negative mining, approximate nearest-neighbor retrieval, online A/B testing, and diversity/novelty metrics.
