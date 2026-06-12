# Personalized Learning Recommendation System

## Project Overview

This project is a **Personalized Learning Recommendation System** developed as part of my **Machine Learning Internship at Innovexa Catalyst**.

The system recommends online courses based on user skills, interests, course category, difficulty level, platform, and ratings. It helps learners find suitable courses according to their learning goals.

---

## Project Details

| Field | Details |
|---|---|
| **Project Title** | Personalized Learning Recommendation System |
| **Name** | Sai Bhavani Yedla |
| **Internship Domain** | Machine Learning |
| **Task** | Machine Learning Project 1 |
| **Dataset** | Coursera.csv / Online Courses Across Platforms Dataset |

---

## Dataset Used

The project uses the **Coursera.csv / Online Courses Across Platforms Dataset**, which contains course-related information such as:

- Course title
- Skills
- Course category
- Difficulty level
- Platform
- Ratings
- Course metadata

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- MinMaxScaler
- Pickle

---

## Recommendation Techniques

### 1. Content-Based Filtering

Recommends courses by analyzing course features such as title, category, skills, difficulty level, and platform.

### 2. Skill-Based Recommendation

Matches user skills and interests with relevant courses to provide suitable learning recommendations.

### 3. Hybrid Recommendation System

Combines multiple recommendation methods to improve the accuracy and relevance of course recommendations.

---

## Evaluation Metrics

| Metric | Score |
|---|---:|
| **Precision@10** | 0.80 |
| **Recall@10** | 0.80 |
| **F1-Score@10** | 0.80 |
| **MAP@10** | 0.80 |
| **NDCG@10** | 0.866 |

---

## Output Files

| File Name | Description |
|---|---|
| `processed_courses.csv` | Cleaned and processed course dataset |
| `tfidf_vectorizer.pkl` | Saved TF-IDF vectorizer model |
| `cosine_similarity.pkl` | Saved cosine similarity matrix |
| `scaler.pkl` | Saved MinMaxScaler object |

---

## How to Run

### **Step 1: Open the Project Folder**

Open the project folder in **VS Code**.

### **Step 2: Add Dataset**

Make sure `Coursera.csv` is in the same folder as `task1.py`.

```bash
Coursera.csv
task1.py
```

### **Step 3: Run the Project**

Run the following command in the terminal:

```bash
python task1.py
```

### **Step 4: Check Output Files**

After running the project, the following files will be generated:

```bash
processed_courses.csv
tfidf_vectorizer.pkl
cosine_similarity.pkl
scaler.pkl
```

## Project Outcome

The project successfully recommends personalized online courses based on user preferences and course features. It demonstrates the use of machine learning, text vectorization, similarity measurement, and evaluation metrics in building a recommendation system.

---

## Submitted By

**Sai Bhavani Yedla**  
**Email:** saibhavaniyedla35@gmail.com  
**Domain:** Machine Learning Internship
