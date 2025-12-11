# Spotify Song Popularity Prediction 🎵  

**Author:** Stephanie Harris  
**Language:** Python  

This project predicts whether a Spotify track will be **popular** based on its audio features and playlist metadata. I treat popularity as a **binary classification problem** rather than regression and explore the full workflow: **EDA → clustering → modeling → evaluation & validation**.

---

## Project Overview

- **Goal:** Predict if a song is popular (`track_popularity > 50`) using audio and playlist features.
- **Problem Type:** Binary classification (popular vs. not popular).
- **Dataset:** [TidyTuesday Spotify Songs](https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-01-21) with ~28k unique tracks after cleaning.
- **Key Idea:** Raw popularity is a bounded 0–100 score with a spike at 0, so regression would produce unrealistic predictions and violate assumptions. Converting to a binary label makes the problem better suited to logistic regression.

---

## Tech Stack & Libraries

All analysis is done in **Python**, using:

- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling & metrics:** `statsmodels`, `scikit-learn`
- **Clustering & decomposition:** `KMeans`, `StandardScaler`, `PCA` (imported), `scipy.cluster.hierarchy`

This repo showcases my ability to work end-to-end in the Python data stack, from raw CSV to validated models.

---

## Repository Structure

- `Harris_Stephanie_Supporting_EDA.ipynb`  
  Supporting **Exploratory Data Analysis** notebook with detailed visualizations and data cleaning steps.

- `Harris_Stephanie_Main.ipynb`  
  Main notebook containing:
  - Data preprocessing and feature engineering  
  - K-means clustering  
  - Logistic regression model fitting and interpretation  
  - Cross-validation and performance comparison  

- `README.md`  
  Project overview (this file).

---

## Data & Feature Engineering

**Selected Features**

- **Continuous inputs**  
  `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,  
  `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`

- **Categorical inputs**  
  `playlist_genre`, `key` (mapped to musical labels C, C#, D, …), `mode` (major/minor)

- **Target variable**  
  `track_popularity_binary` = 1 if `track_popularity > 50`, else 0

**Key Preprocessing Steps**

- Removed duplicate tracks by grouping on `track_id`.
- Parsed `track_album_release_date` into **year** and **decade** for EDA.
- Converted `track_popularity` into a **binary label** to avoid issues with bounded regression.
- Converted `key` and `mode` into human-readable categories.
- Handled **skewed variables** (`speechiness`, `acousticness`, `instrumentalness`, `liveness`, `duration_ms`) using log / log1p transforms.
- **Standardized** continuous features before clustering and modeling.

---

## Exploratory Data Analysis (EDA)

EDA focused on understanding:

- **Distributions** of audio features and popularity.
- **Genre differences** in loudness, energy, danceability, speechiness, etc.
- **Correlations** between continuous variables.
- How features differ across:
  - `playlist_genre`
  - `track_popularity_binary` (popular vs. not popular)

**Key Insights from EDA**

- Popularity has a spike at 0 and is clearly **non-Gaussian** and bounded.
- Genres such as **pop, rap, latin, r&b, rock** show higher proportions of popular songs.
- Audio features alone (e.g., danceability, tempo, valence) show only **modest separation** between popular and non-popular tracks.
- Closer genre-level patterns emerge for features like **loudness, energy, speechiness**, and **danceability**.

A more detailed EDA narrative and plots are available in the supporting EDA notebook.

---

## Clustering

I used **K-Means clustering** on the 10 standardized continuous features:

- Initially explored **k = 2**, then used:
  - **Elbow method** on within-cluster sum of squares
  - **Ward dendrogram**  
  Both suggested **k = 3** as a natural number of clusters.

**Cluster Interpretations (k = 3)**

- **Cluster 0:** High-energy, loud tracks with faster tempos (EDM/rock-heavy).
- **Cluster 1:** Softer, acoustic-leaning, lower-energy songs (more r&b).
- **Cluster 2:** Middle group with balanced features, more latin and rap.

Clusters aligned more strongly with **genre** than with **popularity**, reinforcing the idea that popularity depends on external factors (like curation, artist reputation) beyond audio features alone.

---

## Modeling Approach

I framed the prediction task as **logistic regression** with increasing complexity:

1. **Model 1** – Intercept-only baseline  
2. **Model 2** – Categorical predictors only  
3. **Model 3** – Continuous predictors only  
4. **Model 4** – Additive model: all continuous + categorical  
5. **Model 5** – Continuous predictors + pairwise interactions  
6. **Model 6** – Categorical × continuous interactions (full interaction model)  
7. **Model 7** – Additive model + quadratic terms  
8. **Model 8** – Additive model + pairwise interactions (continuous + categorical)

I evaluated models with:

- **Accuracy**
- **Sensitivity & specificity**
- **ROC AUC**
- **Precision & F1**
- **Confusion matrices**
- **ROC curves**

All models used the same **binary popularity target** and were built using `statsmodels` and `scikit-learn`.

---

## Key Results

**Most Important Predictors**

Across models, the most influential predictors were:

- **`playlist_genre`** (especially pop, rock, latin, rap, r&b)  
- **`loudness`** (higher loudness → higher odds of popularity)  
- **`energy`** (important, but often negative once adjusting for other features)

Other features like **danceability, tempo, acousticness, instrumentalness** contributed smaller but interpretable effects.

**Best Model**

- **Model 6** (all linear main effects + interactions between categorical and continuous variables)
  - ~**65.4% accuracy** on training set  
  - **ROC AUC ≈ 0.674**  
  - **198 coefficients**, highly flexible but complex  

- **Model 4** (additive model, all features)
  - Performed nearly as well as Model 6 with only **28 coefficients**, making it a strong trade-off between performance and interpretability.

**Cross-Validation**

Using **5-fold stratified cross-validation**:

- Model 6 achieved the highest **testing accuracy**, confirming it as the best-performing model overall.
- Model 4 was a close second with much fewer parameters.
- Model 3 (continuous-only model) had lower accuracy, showing that combining **genre + audio features** is crucial.

---

## What I Learned

This project strengthened my skills in:

- **End-to-end EDA**: cleaning, feature engineering, reshaping to long format, and building insightful visualizations.
- **Handling skewed data**: log/log1p transformations and standardization.
- **Clustering**: applying K-Means, using the elbow method and dendrograms, and interpreting clusters.
- **Supervised learning**:  
  - Building and interpreting **logistic regression** models  
  - Working with **interactions** and **nonlinear terms**  
  - Comparing models by complexity vs. performance
- **Model validation**: designing and running **stratified cross-validation**, and interpreting metrics like ROC AUC, sensitivity, specificity, and F1-score.
- **Communicating results**: translating technical findings (e.g., regression coefficients, cluster patterns) into clear, high-level insights.

These skills are directly applicable to real-world **data analytics** and **data science** workflows, especially when turning messy, high-dimensional datasets into reliable, interpretable prediction tools.

---

## How to Run This Project

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
