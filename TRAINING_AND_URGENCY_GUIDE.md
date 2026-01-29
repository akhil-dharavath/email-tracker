# Training & Urgency Guide

This guide provides a complete overview of how to train the **Fusion-Based Email Triaging** model and how the system calculates **urgency** based on emotions.

## 1. System Overview

The core of the system is a **Fusion Model** that combines:
1.  **Text Analysis**: Using `DistilBERT` to understand the content of the email.
2.  **Behavioral Analysis**: Using a custom `FeatureExtractor` to analyze metadata (punctuation, capitalization, etc.).

The model is trained to predict the **Emotion** of the email (Angry, Anxious, Neutral, Happy).

**Urgency** is not directly predicted by the neural network. Instead, it is **derived** from the predicted emotion and the behavioral features using a logic defined in the backend.

---

## 2. Prerequisites

Before training or running the model, ensure you have the necessary environment.

1.  **Python 3.8+** installed.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 3. Training the Model

The training logic is located in `ml/train_pipeline.py`.

### 1. Data Source (Important!)
**Currently, the system uses SYNTHETIC DATA generated inside the code.** 
It does **NOT** load data from an external CSV or text file by default.

The data is generated in `ml/train_pipeline.py` by the `SyntheticEmailDataset` class.
*   It randomly selects an emotion (Angry, Anxious, Neutral, Happy).
*   It generates a simple sentence for that emotion (e.g., "I am extremely furious!").
*   **To use your own data**, you must modify the `train_pipeline.py` file to load your dataset (see below).

### 2. How to Run Training
To train the model using the default synthetic dataset, run:

```bash
python ml/train_pipeline.py
```

This will:
1.  Generate a **Synthetic Dataset** of emails labeled with emotions (Angry, Anxious, Neutral, Happy).
2.  Train the `FusionModel` for a few epochs.
3.  Save the trained weights to `fusion_model.pth`.

### 3. How to Train with REAL Data
To train with your own dataset (e.g., `my_emails.csv`), you need to modify `ml/train_pipeline.py`.

1.  **Prepare your CSV**:
    ```csv
    text,label
    "I am so angry about this",0
    "I am worried about the deadline",1
    "Just checking in",2
    "Great news!",3
    ```
    *(Label definition: 0=Angry, 1=Anxious, 2=Neutral, 3=Happy)*

2.  **Modify `ml/train_pipeline.py`**:
    *   Change the `SyntheticEmailDataset` to load pandas or standard CSV.
    *   Example logic to replace:
        ```python
        # Instead of generating random text:
        # self.data = pd.read_csv("my_emails.csv")
        ```


---

## 4. How Urgency is Calculated

The "Urgency" score is calculated in `backend/app.py` inside the `analyze_emails` function. It works as follows:

### Step 1: Predict Emotion
The trained model predicts one of 4 emotions:
*   **Angry**
*   **Anxious**
*   **Neutral**
*   **Happy**

### Step 2: Base Urgency Mapping
The predicted emotion determines the starting urgency score:
*   **Angry** → **0.8** (High Urgency)
*   **Anxious** → **0.7** (High Urgency)
*   **Happy** → **0.2** (Low Urgency)
*   **Neutral** → **0.1** (Low Urgency)

### Step 3: Behavioral Boosts
The system adds a "boost" to the urgency score based on behavioral features extracted by `ml/feature_extractor.py`:
*   **Exclamation Marks**: Adds `0.1 * count`
*   **Capitalization Ratio**: Adds `0.5 * ratio`
*   **Off-Hours**: Adds `0.3` if sent outside business hours

### The Formula
```python
Final Urgency = Base Urgency (from Emotion) + Behavioral Boosts
```
*The result is capped at a maximum of 1.0.*

---

## 5. Summary

To "get urgency with emotions":
1.  **Train the model** to accurately detect **Angry** and **Anxious** emails (using `ml/train_pipeline.py`).
2.  The system automatically translates these emotions into **High Urgency** scores.
3.  Additional urgency signals (like ALL CAPS or !!!) increase the score further via the heuristic logic.
