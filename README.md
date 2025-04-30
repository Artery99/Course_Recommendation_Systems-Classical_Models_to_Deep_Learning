#  Course Recommendation Systems: From Classical Models to Deep Learning

This project implements a wide range of course recommendation techniques, progressing from exploratory data analysis (EDA) and content-based filtering to collaborative filtering and advanced neural network-based predictors. The aim is to understand, compare, and evolve course recommendation strategies using real user-course interaction data.

---

##  Project Structure

The project is organized into a series of Jupyter Notebooks, each representing a major component of the recommender system pipeline:

---

### 1.  `EDA_Online_Course_Enrollment_Data`
- Performs exploratory data analysis on the course enrollment dataset.
- Summarizes distributions of user activity, course ratings, and sparsity.
- Identifies important trends and potential preprocessing needs.

---

### 2.  `Extract_BoW_Features_from_Course_Textual_Content`
- Extracts Bag-of-Words (BoW) features from course descriptions.
- Applies tokenization, lowercasing, and basic preprocessing to prepare text data for similarity calculations.

---

### 3.  `Calculating_Course_Similarity_using_BoW_Features`
- Computes pairwise cosine similarity between courses based on BoW vectors.
- Builds a similarity matrix to power item-item content-based recommendations.

---

### 4.  `Content-based_Course_Recommender_System_Using_User_Profile_and_Course_Genres`
- Constructs user profiles based on rated course genres.
- Recommends new courses by comparing user profiles with available courses using similarity scores.

---

### 5.  `Clustering_Based_Course_Recommender_System`
- Uses unsupervised learning (e.g., KMeans) to cluster courses into thematic groups.
- Recommends courses from the same cluster as user-rated items.

---

### 6.  `Collaborative_Filtering_based_Recommender_System_using_KNN`
- Implements user-based collaborative filtering using K-Nearest Neighbors (KNN).
- Calculates similarity between users to suggest new courses based on neighbors' preferences.

---

### 7.  `Collaborative_Filtering_based_Recommender_System_using_Non-negative_Matrix_Factorization`
- Implements NMF for matrix factorization of user-course interactions.
- Decomposes the sparse user-item matrix into latent user and course features for rating prediction.

---

### 8.  `Course_Rating_Prediction_using_Neural_Networks`
- Trains a basic neural recommender (`RecommenderNet`) with embedding layers for users and courses.
- Uses a dense regression head to predict rating scores.
- Evaluates performance using RMSE.

---

### 9.  `Regression-based_Rating_Score_Prediction_using_Embedding_Features`
- Implements an improved regression model:
  - Deeper architecture with concatenated embeddings.
  - L2 regularization and ReLU activations.
- Shows performance improvement over the base model.

---

### 10.  `Classification-based_Rating_Mode_Prediction_using_Embedding_Features`
- Reformulates the task as a classification problem:
  - Predicts the most likely discrete rating using softmax.
  - Uses categorical cross-entropy loss and accuracy metrics.
- Suitable for scenarios with discrete rating modes (e.g., Likert scales).

---

## Implementation via Streamlit
The current implementation focuses on a content-based course recommendation system using Bag-of-Words (BoW) features and cosine similarity.

## Dataset

The dataset consists of:
- `user`: Unique user identifiers.
- `item`: Course identifiers.
- `rating`: User ratings for courses (1–5 scale).
- `course_description`: Textual data used for content-based filtering.

> Note: Course descriptions are assumed to be pre-cleaned for NLP tasks.

---

## Techniques Summary

| Type                     | Techniques / Models                                  |
|--------------------------|------------------------------------------------------|
| **Content-Based**        | BoW Similarity, Genre Profiles, Clustering           |
| **Collaborative Filtering** | KNN, Non-negative Matrix Factorization (NMF)     |
| **Neural Networks**      | Embedding + Dense (Regression & Classification)      |

---

## Evaluation Metrics

- **RMSE** for regression models.
- **Classification Accuracy** for mode prediction.
- **Cosine Similarity** for BoW-based recommenders.
- **Top-N Recommendations** using similarity rankings (optional).

---

## Libraries Used

- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib / Seaborn (for visualizations)
- NLTK / Scikit-learn (for text processing)

---

## Future Works

- Integrate the rest of the recommender models into `recommender_app.py` and `backend.py`
- Modify necessary changes to the UI and add extra options for the dropdown menu (Model Selection) based on model parameters for each model.
  
---

##  Author

**Hadi Mekdad**

---

## Thank you

---

If you found this repository useful, please star it and share it with others interested in recommender systems!
