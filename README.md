# GymVibe — Context-Aware Hybrid Music Recommender 🏋️‍♂️

GymVibe is a two-stage recommender system that selects music for a workout based on the current workout context and the user's aggregated listening history.

The project extends my [Gym Vibe Archetype Classifier](https://github.com/sweeteri/gym-vibe-archetype-classifier): the original project classified tracks into seven workout-oriented music archetypes, while GymVibe uses these archetypes as real-time context for personalized recommendations.

> **Data & Architecture Note:** The original dataset contains track features but no real user interactions. For this experiment, I generate synthetic implicit-feedback logs. Crucially, **user profiles are not hardcoded**. Instead, a simulated **Feature Store** aggregates historical positive interactions to build the user representation, strictly preventing data leakage. Therefore, the reported metrics are robust benchmarks for this experimental setup.

## How it works

```text
Aggregated User History (avg/std BPM, count) + Context ──► User Tower ─┐
                                                                        ├─ similarity ─► Top-50 candidates
Track audio/text features ───────────────────────────────► Item Tower ──┘
                                                                           │
                         context, genre, aggregated user features,
                              neural similarity score (nn_score)
                                                                           ▼
                                                CatBoostRanker (YetiRank, grouped by session)
                                                                           ▼
                                                                    Final Top-N tracks

```
# Retrieval — Two-Tower / DSSM

The PyTorch model maps users and tracks into the same embedding space.

**User features (10 dimensions total):**
- Aggregated historical features: `user_avg_bpm`, `user_std_bpm`, `user_listens_count` (normalized).
- Current workout archetype (one-hot encoded).

**Track features:**
- BPM
- Loudness (gain)
- Lyric word count
- Word density
- Power Index
- Street Score

The model uses an **in-batch contrastive objective**: the observed user–track interaction is treated as positive, while other items in the batch provide negatives.

The full catalogue is scored by embedding similarity, and the **top 50 tracks** are passed to the ranking stage.

---

# Ranking — CatBoost

`CatBoostRanker` with **YetiRank** re-ranks the retrieved candidates using:

- Neural similarity score (`nn_score`)
- Workout context
- Genre
- Aggregated user features (`user_avg_bpm`, `user_std_bpm`, `user_listens_count`)
- Audio/text features

`session_id` is strictly used as the ranking **group**, and `like` as the relevance **label**, ensuring the model learns to order tracks within a specific workout session.

---

# Workout Archetypes

The seven contexts come from the original classifier project:

1. Boss Fight / Doom Slayer
2. Zen / Lo-Fi Recovery
3. Golden Era (Old School)
4. Pilates Girl
5. Pure Rage / Metalhead
6. Sigma Grindset
7. Пацанский Вайб

---

# Features

A few custom features are used to describe the tracks:

- **Power Index** — simple energy proxy based on BPM and loudness.
- **Street Score** — vectorized count of predefined street-culture markers in lyrics.
- **Word Density** — lyric word count divided by track duration.

---

# Offline Evaluation

The notebook uses a strict **session-level train/test split**. This ensures that no future sessions from a user leak into the training data, accurately simulating a real-world temporal deployment.

**Current experiment (leak-free baseline):**

| Metric | Result |
|---|---|
| Retrieval Recall@50 | ~0.259 |
| Ranking NDCG@5 | ~0.240 |

**Recall@50** is evaluated against the full catalogue and measures how many positive tracks the Two-Tower model successfully retrieved into the Top-50 shortlist.

**NDCG@5** is evaluated on the retrieved candidates and measures how well CatBoost orders those 50 candidates, heavily penalizing relevant tracks that are pushed to the bottom of the list.

> **Note:** Exact values may vary slightly with the random seed, generated synthetic logs, and package versions.

---

# Run

Run all notebook cells to generate the logs, build the Feature Store, train both stages, evaluate the model, and inspect recommendations, feature importance, and t-SNE embeddings.

---

# Tech Stack

Python · PyTorch · CatBoost · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

---

# Limitations & Next Steps

This is a **portfolio prototype** demonstrating production-grade RecSys patterns, rather than a full-scale production recommender.

The main limitations are the **synthetic nature of the feedback** and a **relatively small catalogue**.

A production version would:

- Ingest real timestamped event streams into a real Feature Store (e.g., **Feast**)
- Implement **hard-negative mining** during Two-Tower training
- Replace brute-force retrieval with an **Approximate Nearest Neighbor (ANN)** index (e.g., **FAISS**) for sub-millisecond latency
- Add **diversity/novelty** metrics
