# UTS-Machine-Learning-Citrus-Classification

Projek ini mengelompokkan buah citrus antara :

- Orange
- Grape

menggunakan 3 machine learning algorithms:

1. Decision Tree
2. Naive Bayes
3. Support Vector Machine (SVM)

## Dataset

File used:

- `citrus.csv`

Fitur:

- diameter
- weight
- red
- green
- blue

Target:

- name (orange / grape)

## Extension tools : 

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Heatmap

Pada Heatmap, kita dapat melihat prediksi dan realita data, dimana 0 adalah grape, dan 1 adalah orange. Maka,
- Jika 0 memprediksi 0, maka algoritma benar memprediksi grape (top-left)
- Jika 0 memprediksi 1, maka algoritma salah memprediksi grape (bottom-left)
- Jika 1 memprediksi 0, maka algoritma salah memprediksi orange (top-right)
- Jika 1 memprediksi 1, maka algoritma benar memprediksi orange (bottom-right)

1. Decision Tree
<img width="518" height="393" alt="decision_tree" src="https://github.com/user-attachments/assets/b396114a-0318-47f0-b827-c797b453ec28" />
Dari gambar heatmap, kita mendapatkan nilai grape yang benar adalah 1163 dan yang salah adalah 89. Sedangkan nilai orange yang benar adalah 1183 dan yang salah adalah 65.

2. Naive Bayes
<img width="518" height="393" alt="naive_bayes" src="https://github.com/user-attachments/assets/7ac807be-26b8-4a76-bd45-1feb619d8b9c" />
Dari gambar heatmap, kita mendapatkan nilai grape yang benar adalah 1143 dan yang salah adalah 113. Sedangkan nilai orange yang benar adalah 1159 dan yang salah adalah 85.

3. Support Vector Machine (SVM)
<img width="518" height="393" alt="svm" src="https://github.com/user-attachments/assets/18d93de3-25bd-4c93-8311-66bd33c4cb26" />
Dari gambar heatmap, kita mendapatkan nilai grape yang benar adalah 1141 dan yang salah adalah 99. Sedangkan nilai orange yang benar adalah 1173 dan yang salah adalah 87.

## Hasil

| Algorithm | Accuracy |
|----------|----------|
| Decision Tree | 93.84% |
| Naive Bayes | 92.08% |
| SVM | 92.56% |

Best model: **Decision Tree**

## How to Run

```bash
python citrus.py

//cape wkwkw ribet banget setup jupyternya
