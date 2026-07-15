# GymVibe: Personalized Music for Your Workout 🏋️‍♂️

**GymVibe** is a recommendation system that picks the best songs for your workout. It understands the vibe of your training session and your personal musical taste.

### Evolution of the Project
This project is an advanced extension of the [Gym Vibe Archetype Classifier](https://github.com/sweeteri/gym-vibe-archetype-classifier). 
*   **Original Project:** Focused on classifying songs into "Vibes" (archetypes).
*   **Current Project:** Uses those vibes as a starting point to build a **Personalized Ranking System** that predicts which songs you will actually like during your workout.

---

## How it works

1.  **Step 1: Selection (Retrieval)**
    **Neural Network (Two-Tower model)** is used to filter the library and find songs that match the specific gym "vibe" you want (e.g., only "Boss Fight" music).
    
2.  **Step 2: Sorting (Ranking)**
    **CatBoost** is used to sort those songs. It looks at your favorite tempo (BPM) and your past likes to put the best songs at the very top of your list.

---

## The 7 Gym "Vibes" (Archetypes)
The music was categorized into 7 psychological profiles to match your training:

1.  **Zen / Lo-Fi Recovery** – Relaxing beats for Yoga or stretching.
2.  **Pilates Girl** – Chill R&B and smooth rhythms.
3.  **Golden Era (Old School)** – 80s rock and action movie energy.
4.  **Street Vibe** – Gritty street rap and deep bass.
5.  **Sigma Grindset** – Dark synth-pop and "lone wolf" vibes.
6.  **Boss Fight / Doom Slayer** – High-intensity metal and epic choirs.
7.  **Pure Rage / Metalhead** – Maximum aggression for your heaviest lifts.

---

## Smart Features
To understand the music better, custom metrics were created:
*   **Power Index:** Measures how "energetic" a song feels based on speed and volume.
*   **Street Score:** Analyzes lyrics to find "street/thug" culture vibes.
*   **Personal BPM:** Learns the specific speed of music you prefer while training.

---

## Performance & Results

*   **NDCG@5: 0.2055** — High ranking quality; the most relevant songs consistently appear at the top of the recommendations.
*   **Retrieval Loss: 0.3396** — The "Two-Tower" neural network successfully learned to map user preferences to audio features.
*   **Key Insight** — Feature Importance analysis proves that combining **Neural Scores** with **Gym Context** creates the most accurate recommendations.
---

## Tech Stack
*   **Python** (Data processing)
*   **PyTorch** (Neural Network for finding songs)
*   **CatBoost** (Smart ranking of songs)
*   **NLP** (Analyzing song lyrics)
*   **Matplotlib & Seaborn** (Charts and visuals)
