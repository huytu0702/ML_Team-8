# Machine Learning Project:
Kaggle competition: Child Mind Institute — Problematic Internet Use

## 📘 Introduction
* The main aim of the competition is to use our training data to predict **sii** or **Severity Impairment Index**, which is a standard measure of Problematic Internet Use (PIU).
* The training data comprises 3,960 records of children and young people with 81 columns (not including the ID column).
* Of particular importance in the data are results of the **Parent-Child Internet Addiction Test (PCIAT)**.
* The target is actually derived from the field PCIAT-PCIAT_Total (scored out of 100).
* We can therefore choose to predict the PCIAT Total and convert this to sii (making this a regression problem) or stick with sii (making this a classification problem).
* The test data is really just formatted sample data. The actual test data of about 3,800 instances is hidden.
* In the sample data none of the 22 PCIAT fields are available (in addition to the target feature). Hence the sample data format has 58 columns compared to 81 in the train data.
* In 1,224 records in the train data the sii target and all the PCIAT columns are missing - presumably because not available.
* Overall there are > 100,000 missing values in the train data.
* Only 2,736 records have a target, the rest are missing.
* 996 of the young people also have sensor data from a worn device which measures gross motor activity.

The file `BTLML.ipynb` is the main Jupyter notebook that includes all steps with code and visualizations.

---

## 📋 Key Features
### 1. Data Preprocessing

* Handle missing values efficiently.
* Analyze feature correlations and perform feature selection.
* Normalize and clean the dataset for better performance.  
### 2. Visualization

* Generate meaningful visualizations for data distributions and relationships.
* Plot feature importances and correlation images.
### 3. Model Training  

* Use **XGBoost Classifier** with optimized hyperparameters for classification.
* Evaluate using **Stratified K-Fold Cross-Validation and Quadratic Weighted Kappa (QWK)**.
* Hyperparameter tuning using **Optuna**.
### 4. Feature Importance 

* Extract insights on the most influential features using built-in and permutation importance techniques.
### 5. Result Submission

* Predict target values and export the results in  submission.csv .

---
## 🔧Technologies Used
This project leverages the following libraries and frameworks:  

* **Data Manipulation**: pandas, numpy
* **Visualization**: matplotlib, seaborn
* **Machine Learning**: xgboost, scikit-learn, eli5
* **Hyperparameters Tuning**: optuna

---

## 📊 Key Steps in the Notebook
### 1. Data Preprocessing
**- Encoding data:**
  - Encode object data into numeric format (float64)

**- Missing Value Handling:**

  - Drop features with more than 50% missing values.
  - Fill missing values in important features with median or mode.
    
**- Feature Selection:**

  - Analyze correlations between features and the target variable.
  - Remove redundant and irrelevant features.
### 2. Exploratory Data Analysis (EDA)
**- Visualizations:**

  - Distribution plots for features.
  - Correlation images for feature analysis.
  - Stacked bar charts for categorical distributions.

**- Statistical Summaries:**

  - Display statistics for selected features (min, max, mean, etc.).
### 3. Model Training
* **Algorithm**: XGBoost Classifier
* **Hyperparameters**: The model was tuned offline using Optuna  
`xgb_params = {
    'max_depth': 3,
    'n_estimators': 229,
    'learning_rate': 0.07102997599091539,
    'subsample': 0.7883429477946611,
    'colsample_bytree': 0.5249446655930966,
}`
* **Evaluation Metric**: Quadratic Weighted Kappa (QWK)
### 4. Feature Importance Analysis
* **Built-in Feature Importance**: Visualize feature importance from the XGBoost model.

* **Permutation Importance**: Use permutation importance to validate the robustness of the feature rankings.

### 5. Submission
* Generate predictions for the test dataset.
* Save predictions in submission.csv.

--- 

## 🎨 Visual Outputs
### 1. Correlation with PCIAT Total Score
![1](https://github.com/user-attachments/assets/e96f1dca-3fe7-45f8-bb29-18ad45cfba36)

### 2. Distribution of "sii" Across PCIAT Score Ranges
![2](https://github.com/user-attachments/assets/013598f4-cff2-4594-ac1d-2bce34fdc229)

### 3. PIU
![3](https://github.com/user-attachments/assets/3fbb176f-f063-4f72-8209-32f26cf6ab4e)
![4](https://github.com/user-attachments/assets/d40c214f-8472-470b-9220-5f7bfce66400)
![5](https://github.com/user-attachments/assets/3e1e4bfd-a92f-47b0-854c-6a9ce6f70576)
![6](https://github.com/user-attachments/assets/6538e715-6258-4e3c-9cd9-b052be1dde6a)

### 4. Correlations
![10](https://github.com/user-attachments/assets/d9ee2808-86f1-4f89-baec-9949cd7c2701)
![11](https://github.com/user-attachments/assets/8aa08b61-492e-4491-a245-539b1ddc22bb)
![12](https://github.com/user-attachments/assets/9bec4ad0-9df8-436a-8aaf-15f9dcc39c5a)

### 5. Missing values
![13](https://github.com/user-attachments/assets/86acad96-41e4-4da6-95c9-877a43bfc6c0)
![14](https://github.com/user-attachments/assets/34ffa53b-85d0-4eee-911c-f991e19d2ccf)


### 6. Selected Features
![9](https://github.com/user-attachments/assets/12edc8b1-9b6c-4ea4-9c64-4029118e8771)
![7](https://github.com/user-attachments/assets/d9b17b77-9595-4631-8d6d-c6feab44ae44)

### 7. Features Importances
![8](https://github.com/user-attachments/assets/33a2ddd9-f54b-4547-8ed4-075f4dd263d0)

---
## 📌 Project Highlights
* End-to-end machine learning pipeline for classification tasks.
* Detailed visualizations for better insights into the dataset.
* Use of advanced evaluation metrics (Quadratic Weighted Kappa) to ensure robust model evaluation.

---
## ❤️ Acknowledgments  
This project is built with the help of open-source libraries and frameworks. Special thanks to:

* **XGBoost** for providing an efficient boosting algorithm.
* **Pandas & NumPy** for seamless data manipulation.
* **Seaborn & Matplotlib** for creating stunning visualizations.
* **Eli5** for showing permutation importance among features

---

## 🚀 How to Run the Project
### Prerequisites
* Install Python 3.8 or higher.
* Install Jupyter Notebook.
* Data from: https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/data

### Steps
1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/huytu0702/ML_Team-8
   cd <project-directory>
   ```

2. Set up a virtual environment and activate it:

   ```bash
   python -m venv env
   source env/bin/activate  # On macOS/Linux
   env\Scripts\activate    # On Windows
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Jupyter Notebook:

   ```bash
   jupyter notebook btl-ml.ipynb
   ```

5. Run the notebook cells sequentially to process the data and train the model.

---
## 📊 Member Contributions
| STT  | Name |  Student ID | Task | Contribute |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1  | Nguyễn Huy Tú (leader)  | 22028126  | Data processing, slides  | 33,3%  |
| 2  | Kiều Đức Long  | 22028277  | Building model, presentation  | 33,3%  |
| 3  | Trương Sỹ Đạt  | 22028317  | Data processing, slides  | 33,3%  |

---
## 📬 Contact
For any queries or feedback, feel free to open an issue in this repository or reach out via email: 22028126@vnu.edu.vn




