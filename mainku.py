import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

# Metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model Select
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import metrics
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler

from streamlit_option_menu import option_menu

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",  # required
        options=["Home", "Datasets", "Pre-Processing",
                 "Modelling", "Implementation"],  # required
        icons=["house", "folder", "file-bar-graph",
               "card-list", "calculator"],  # optional
        menu_icon="menu-up",  # optional
        default_index=0,  # optional
    )


if selected == "Home":
    st.title(f'Aplikasi Web Data Mining')
    st.write(""" ### Klasifikasi tingkat penyakit diabetes yang sangat tinggi di negara india menggunakan Metode Decision tree, Random forest, dan SVM
    """)
    st.write(""" ###Dataset terdiri dari beberapa variabel prediktor medis dan satu variabel target, Outcome. Variabel prediktor meliputi jumlah kehamilan yang dialami pasien, IMT, kadar insulin, usia, dan sebagainya.""")

if selected == "Datasets":
    st.title(f"{selected}")
    data_hf = pd.read_csv(
        "https://raw.githubusercontent.com/Arbiansa/webpendat/main/diabetes.csv")  # dataset github
    st.write("Pima Indians Diabetes - EDA & Prediction (0.1206)  : ", data_hf)
    st.write('Jumlah baris dan kolom :', data_hf.shape)
    X = data_hf.iloc[:, 0:11].values
    y = data_hf.iloc[:, 0:11].values
    st.write('Dataset Description :')
    st.write('1. Pregnancies: Number of times pregnant')
    st.write(
        '2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
    st.write('3. BloodPressure: Diastolic blood pressure (mm Hg)')
    st.write('4. SkinThickness: Triceps skin fold thickness (mm)')
    st.write('5. Insulin: 2-Hour serum insulin (mu U/ml)')
    st.write('6. BMI: Body mass index (weight in kg/(height in m)^2)')
    st.write('7. DiabetesPedigreeFunction: Diabetes pedigree function')
    st.write('8. Age: Age (years)')
    st.write('9. Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0')

    st.write("Dataset Fetal Health Download : (https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-1206) ")

if selected == "Pre-Processing":
    st.title(f"{selected}")
    data_hf = pd.read_csv(
        "https://raw.githubusercontent.com/Arbiansa/webpendat/main/diabetes.csv")
    X = data_hf.iloc[:, 0: 9].values
    y = data_hf.iloc[:, 0: 9].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # y = le.fit_transform(y)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    st.write("Hasil Preprocesing : ", scaled)

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        scaled, y, test_size=0.3, random_state=0)


if selected == "Modelling":
    st.title(f"{selected}")
    st.write("""  # Decision Tree, Random Forest, SVM """)
    data_hf = pd.read_csv(
        "https://raw.githubusercontent.com/Arbiansa/webpendat/main/diabetes.csv")
    X = data_hf.iloc[:, 0:9].values
    y = data_hf.iloc[:, 0:9].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    # y = le.fit_transform(y)

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    Y_pred = decision_tree.predict(X_test)
    accuracy_dt = round(accuracy_score(y_test, Y_pred) * 120, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 120, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test, Y_pred)
    precision = precision_score(y_test, Y_pred, average='micro')
    recall = recall_score(y_test, Y_pred, average='micro')
    f1 = f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for DecisionTree\n', cm)
    print('accuracy_DecisionTree: %.3f' % accuracy)
    print('precision_DecisionTree: %.3f' % precision)
    print('recall_DecisionTree: %.3f' % recall)
    print('f1-score_DecisionTree : %.3f' % f1)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=120)
    random_forest.fit(X_train, y_train)
    Y_prediction = random_forest.predict(X_test)
    accuracy_rf = round(accuracy_score(y_test, Y_prediction) * 120, 2)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 120, 2)

    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test, Y_prediction)
    precision = precision_score(y_test, Y_prediction, average='micro')
    recall = recall_score(y_test, Y_prediction, average='micro')
    f1 = f1_score(y_test, Y_prediction, average='micro')
    print('Confusion matrix for Random Forest\n', cm)
    print('accuracy_random_Forest : %.3f' % accuracy)
    print('precision_random_Forest : %.3f' % precision)
    print('recall_random_Forest : %.3f' % recall)
    print('f1-score_random_Forest : %.3f' % f1)

    # SVM
    SVM = svm.SVC(kernel='linear')
    SVM.fit(X_train, y_train)
    Y_prediction = SVM.predict(X_test)
    accuracy_SVM = round(accuracy_score(y_test, Y_pred) * 120, 2)
    acc_SVM = round(SVM.score(X_train, y_train) * 120, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test, Y_pred)
    precision = precision_score(y_test, Y_pred, average='micro')
    recall = recall_score(y_test, Y_pred, average='micro')
    f1 = f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for SVM\n', cm)
    print('accuracy_SVM : %.3f' % accuracy)
    print('precision_SVM : %.3f' % precision)
    print('recall_SVM : %.3f' % recall)
    print('f1-score_SVM : %.3f' % f1)
    st.write("""
    #### Akurasi:""")
    results = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest', 'SVM'],
        'Score': [acc_decision_tree, acc_random_forest, acc_SVM],
        'Accuracy_score': [accuracy_dt, accuracy_rf, accuracy_SVM]})

    result_df = results.sort_values(by='Accuracy_score', ascending=False)
    result_df = result_df.reset_index(drop=True)
    result_df.head(12)
    st.write(result_df)

    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(['Decision Tree', 'Random Forest', 'SVM'],
           [accuracy_dt, accuracy_rf, accuracy_SVM])
    plt.show()
    st.pyplot(fig)


if selected == "Implementation":
    st.title(f"{selected}")
    st.write("""
            ### Pilih Metode yang anda inginkan :"""
             )
    algoritma = st.selectbox(
        'Pilih', ('Decision Tree', 'Random Forest', 'SVM'))

    data_hf = pd.read_csv(
        "https://raw.githubusercontent.com/Arbiansa/webpendat/main/diabetes.csv")
    X = data_hf.iloc[:, 0:9].values
    y = data_hf.iloc[:, 9].values
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train and Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    # Decision Tree
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, y_train)
    Y_pred = decision_tree.predict(X_test)
    accuracy_dt = round(accuracy_score(y_test, Y_pred) * 120, 2)
    acc_decision_tree = round(decision_tree.score(X_train, y_train) * 120, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test, Y_pred)
    precision = precision_score(y_test, Y_pred, average='micro')
    recall = recall_score(y_test, Y_pred, average='micro')
    f1 = f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for DecisionTree\n', cm)
    print('accuracy_DecisionTree: %.3f' % accuracy)
    print('precision_DecisionTree: %.3f' % precision)
    print('recall_DecisionTree: %.3f' % recall)
    print('f1-score_DecisionTree : %.3f' % f1)

    # Random Forest
    random_forest = RandomForestClassifier(n_estimators=120)
    random_forest.fit(X_train, y_train)
    Y_prediction = random_forest.predict(X_test)
    accuracy_rf = round(accuracy_score(y_test, Y_prediction) * 120, 2)
    acc_random_forest = round(random_forest.score(X_train, y_train) * 120, 2)

    cm = confusion_matrix(y_test, Y_prediction)
    accuracy = accuracy_score(y_test, Y_prediction)
    precision = precision_score(y_test, Y_prediction, average='micro')
    recall = recall_score(y_test, Y_prediction, average='micro')
    f1 = f1_score(y_test, Y_prediction, average='micro')
    print('Confusion matrix for Random Forest\n', cm)
    print('accuracy_random_Forest : %.3f' % accuracy)
    print('precision_random_Forest : %.3f' % precision)
    print('recall_random_Forest : %.3f' % recall)
    print('f1-score_random_Forest : %.3f' % f1)

    # SVM
    SVM = svm.SVC(kernel='linear')
    SVM.fit(X_train, y_train)
    Y_prediction = SVM.predict(X_test)
    accuracy_SVM = round(accuracy_score(y_test, Y_pred) * 120, 2)
    acc_SVM = round(SVM.score(X_train, y_train) * 120, 2)

    cm = confusion_matrix(y_test, Y_pred)
    accuracy = accuracy_score(y_test, Y_pred)
    precision = precision_score(y_test, Y_pred, average='micro')
    recall = recall_score(y_test, Y_pred, average='micro')
    f1 = f1_score(y_test, Y_pred, average='micro')
    print('Confusion matrix for SVM\n', cm)
    print('accuracy_SVM : %.3f' % accuracy)
    print('precision_SVM : %.3f' % precision)
    print('recall_SVM : %.3f' % recall)
    print('f1-score_SVM : %.3f' % f1)

    st.write("""
            ### Input Data :"""
             )
    Pregnancies = st.sidebar.number_input(
        "Pregnancies =", min_value=0, max_value=17)
    Glucose = st.sidebar.number_input(
        "Glucose =", min_value=0, max_value=11212)
    BloodPressure = st.sidebar.number_input(
        "BloodPressure =", min_value=0, max_value=122)
    SkinThickness = st.sidebar.number_input(
        "SkinThickness =", min_value=0, max_value=1212)
    Insulin = st.sidebar.number_input(
        "Insulin =", min_value=0, max_value=846)
    BMI = st.sidebar.number_input(
        "BMI =", min_value=0, max_value=68)
    DiabetesPedigreeFunction = st.sidebar.number_input(
        "DiabetesPedigreeFunction =", min_value=0, max_value=3)
    Age = st.sidebar.number_input(
        "Age =", min_value=21, max_value=81)
    Outcome = st.sidebar.number_input(
        "Outcome =", min_value=0, max_value=1)
    submit = st.button("Submit")
    if submit:
        if algoritma == 'Decision Tree':
            X_new = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                             Age, Outcome]])
            prediksi = decision_tree.predict(X_new)
            if prediksi == 1:
                st.write(
                    """ ## Hasil Prediksi : resiko terkena penyakit diabetes tinggi""")
            else:
                st.write(
                    """## Hasil Prediksi : resiko terkena penyakit diabetes rendah""")
        elif algoritma == 'Random Forest':
            X_new = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                             Age, Outcome]])
            prediksi = random_forest.predict(X_new)
            if prediksi == 1:
                st.write(
                    """## Hasil Prediksi : resiko terkena penyakit diabetes tinggi""")
            else:
                st.write(
                    """## Hasil Prediksi : resiko terkena penyakit diabetes rendah""")
        else:
            X_new = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
                             Age, Outcome]])
            prediksi = SVM.predict(X_new)
            if prediksi == 1:
                st.write(
                    """## Hasil Prediksi : resiko terkena penyakit diabetes tinggi""")
            else:
                st.write(
                    """## Hasil Prediksi : resiko terkena penyakit diabetes rendah""")
