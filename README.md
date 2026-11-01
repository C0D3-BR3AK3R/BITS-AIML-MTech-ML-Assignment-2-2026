# Student dropout and academic success

## a. Problem statement

From a student's record at the end of the first year, can we tell if they will drop out, stay enrolled, or graduate? That is the question. Five classifiers are trained on the same students and compared on one held-out test set.

## b. Dataset description

The data is [Predict Students' Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) (UCI id 697; Realinho et al., 2022). It has 4,424 students from a Portuguese institute and 36 features: demographics, parents' background, admission grades, first- and second-semester results, and a few economic indicators. Nothing is missing.

The target has three classes. Graduate is about half the students (2,209), Dropout is 1,421, and Enrolled is the small group (794). The split is a stratified 80/20, so 3,539 rows for training and 885 for testing. Precision, recall, and F1 in the table are weighted. AUC is weighted one-vs-rest.

## c. Github Repository Link

https://github.com/<your-username>/<your-repo>

## d. Models used

Same preprocessing and the same test students for every model. Tree depth is 8 and KNN uses 11 neighbours, both chosen on the training set only.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7254 | 0.8967 | 0.7652 | 0.7254 | 0.7384 | 0.5745 |
| Decision Tree | 0.6610 | 0.8223 | 0.7425 | 0.6610 | 0.6837 | 0.4973 |
| KNN | 0.7085 | 0.8464 | 0.6932 | 0.7085 | 0.6825 | 0.5116 |
| Naive Bayes | 0.2441 | 0.7005 | 0.5531 | 0.2441 | 0.1675 | 0.1242 |
| Support Vector Machine | 0.7277 | 0.9016 | 0.7595 | 0.7277 | 0.7382 | 0.5723 |

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Best F1 and MCC. Graduate and Dropout are mostly right. Enrolled is the messy class: the model finds many of them, but it also dumps dropouts and graduates into that label. |
| Decision Tree | Weaker than the top two. It catches more Enrolled students, then spoils it by calling a lot of graduates and dropouts Enrolled as well. |
| KNN | Accuracy looks fine only because it usually says Graduate. It misses most of the Enrolled students. |
| Naive Bayes | Not usable as a labeler. It calls almost everyone Enrolled, so accuracy falls below just guessing the majority class. The probability ranking is better than the labels, but still last. |
| Support Vector Machine | Neck and neck with logistic regression. Slightly better accuracy and the best AUC. Same weakness on Enrolled. |
| Overall, Winner for your dataset? | Logistic regression, by a hair on F1 and MCC. The SVM is close enough that either would do, and neither really solves Enrolled. |

## Setup

Use Python 3.12. From this folder:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The trained models are already in `model/`. Run `python train.py` only if you want to fit them again.
