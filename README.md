# cs421
Research study

# README: Reproducibility Study for Semantic Change Detection with BERT and XLM-R

This repository contains the code and setup used to reproduce the experiments from the paper on **semantic change detection** using **average self-embedding distances (SED)** and **Spearman correlation metrics**. The study evaluates the effectiveness of pre-trained models like **BERT** and **XLM-Roberta** in detecting artificial semantic change over multiple layers.

---

## **1. Project Structure**
```
project_folder/
├── asc-lr-main/
│   ├── bert/
│   │   ├── cosine_distances/
│   │   ├── target_index/
│   │   ├── special_token_mask/
│   ├── xlmr/
│   ├── src/
│   │   ├── wic.py  # Embedding extraction and evaluation logic
│   │   ├── __init__.py
│   ├── sed_plots.py  # Visualization script for metrics
│   ├── wic_stats.py  # Main script for evaluation
│   ├── README.md
├── WiC/
│   ├── wic_en/  # Dataset folder
│   │   ├── train.txt
│   │   ├── dev.txt
│   │   ├── test.txt
│   ├── target_embeddings/
```

---

## **2. Installation**

### **Dependencies**
- Python 3.8+
- Required libraries:  
  - `torch`
  - `numpy`
  - `scipy`
  - `pandas`
  - `sklearn`
  - `matplotlib`

### **Setup**
1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd project_folder
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up a GPU environment for faster computation.

---

## **3. Dataset Preparation**
### **Dataset Structure**
- Place dataset files (`train.txt`, `dev.txt`, `test.txt`) in the folder `WiC/wic_en/`.
- Ensure the dataset format matches JSON lines, where each line represents a sample.

### **Embedding Files**
- Pre-extracted embeddings are stored in `WiC/wic_en/target_embeddings/bert/` and `WiC/wic_en/target_embeddings/xlmr/`.

---

## **4. Running the Experiments**

### **Step 1: Extract Target Embeddings**
Run the script to extract embeddings for BERT:
```bash
python asc-lr-main/src/wic.py -d WiC/wic_en -m bert-base-uncased --test_set --train_set --dev_set
```

### **Step 2: Compute Metrics**
Evaluate **SED** and **Spearman correlation** for semantic change detection:
```bash
python asc-lr-main/sed_plots.py
```

### **Step 3: Visualize Results**
Generate plots for average SED and Spearman correlation:
```bash
python asc-lr-main/sed_plots.py
```
Output plots will be saved in the folder:  
`/asc-lr-main/bert/cosine_distances/bert_sed_plot.png`.

---

## **5. Results**
### **Metrics Evaluated**
- **Average Self-Embedding Distance (SED):** Tracks semantic stability over layers.
- **Spearman Correlation:** Measures similarity alignment over model layers.

### **Visualization**
Generated plots:
- **Figure 1:** Average SED over layers.
- **Figure 2:** Spearman Correlation over layers for artificial semantic change.

---
