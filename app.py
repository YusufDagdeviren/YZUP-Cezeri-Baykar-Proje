import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler


class App:
    def __init__(self):
        self.dataset_name: None
        self.classifier_name: None
        self.params = dict()
        self.uploaded_file = None
        self.clf = None
        self.X, self.y = None, None
        self.Init_Streamlit_page()

    def Init_Streamlit_page(self):
        st.title("YZUP-Cezeri-Baykar-Proje")
        self.classifier_name = st.sidebar.selectbox(
            'Select Classifier',
            ("KNN", "SVM", "Naive Bayes")
        )

    def run(self):
        self.get_dataset()
        self.generate()

    def get_dataset(self):
        self.uploaded_file = st.file_uploader("Upload Breast Cancer CSV file", type=["csv"])
        if self.uploaded_file is not None:
            data = pd.read_csv(self.uploaded_file)
            self.dataset_name = st.sidebar.selectbox(
                'Select Dataset',
                (self.uploaded_file.name,)
            )
            st.write("# Görev 1")
            st.write("### DataFrame'in ilk 10 satiri")
            st.table(data.head(10))
            st.write("### DataFrame'in Sütunları")
            st.write(data.columns.tolist())
            data = data.drop(['id', 'Unnamed: 32'], axis=1)
            st.write("# Görev 2")
            st.write("### DataFrame'in gereksiz sütunlarını sildikten sonra son 10 satiri")
            st.table(data.tail(10))
            data['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)
            self.y = data['diagnosis']
            self.X = data.drop('diagnosis', axis=1)
            # Verisetinde imbalanced classification var sınıfları dengelemek için veri ürettim
            st.write("### Aşağıda görüldüğü üzere verisetinde imbalanced classification problemi var.")
            fig = self.create_bar_graph()
            st.pyplot(fig)
            sampler = RandomOverSampler()
            self.X, self.y = sampler.fit_resample(self.X, self.y)
            st.write("### Verisetindeki imbalanced classification problemini OverSampler ile çözdükten sonra.")
            fig = self.create_bar_graph()
            st.pyplot(fig)

            fig = self.create_correlation_matrix(data)
            st.write("### Korelasyon Matrisi Çizimi")
            st.pyplot(fig)

    def generate(self):
        if self.uploaded_file is not None:
            self.get_classifier()
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)
            self.clf.fit(X_train, y_train)
            y_predict = self.clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_predict)
            precision = precision_score(y_test, y_predict)
            recall = recall_score(y_test, y_predict)
            f1 = f1_score(y_test, y_predict)
            st.write(f"# Görev 4")
            st.write(f"Classifier = {self.classifier_name}")
            st.write(f"Model Accuracy: {accuracy}")
            st.write(f" Model Precision: {precision}")
            st.write(f" Model Recall: {recall}")
            st.write(f" Model F1 Score: {f1}")
            st.write("### Confusion Matrix")
            f = self.create_confusion_matrix(y_predict, y_test)
            st.pyplot(f)

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            svm = SVC()
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                          'kernel': ['rbf']}
            self.clf = GridSearchCV(svm, param_grid, refit=True, verbose=0)
            st.write("### Veri normalizasyonundan önce X değişkenleri")
            st.table(self.X.head(5))
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
            st.write("### Veri normalizasyonundan sonra X değişkenleri")
            st.table(self.X.head(5))
        elif self.classifier_name == 'KNN':
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': np.arange(1, 10), 'weights': ['uniform', 'distance'],
                          'metric': ['euclidean', 'manhattan']}
            self.clf = GridSearchCV(knn, param_grid, cv=5)
            st.write("### Veri normalizasyonunda önce X değişkenleri")
            st.table(self.X.head(5))
            self.X = (self.X - self.X.min()) / (self.X.max() - self.X.min())
            st.write("### Veri normalizasyonundan sonra X değişkenleri")
            st.table(self.X.head(5))
        else:
            mnb = MultinomialNB()
            self.clf = mnb

    def create_confusion_matrix(self, y_predict, y_test):
        cm = confusion_matrix(y_test, y_predict)
        f, ax = plt.subplots(figsize=(5, 5))
        sns.heatmap(cm, annot=True, linewidths=0.5, linecolor="red", fmt=".0f", ax=ax)
        plt.xlabel("y_pred")
        plt.ylabel("y_true")
        return f

    def create_correlation_matrix(self, data):
        malignant_data = data[data['diagnosis'] == 1]
        benign_data = data[data['diagnosis'] == 0]
        fig, ax = plt.subplots()
        sns.scatterplot(data=malignant_data, x='radius_mean', y='texture_mean', color='red', label='kotu',
                        ax=ax, alpha=0.4)
        sns.scatterplot(data=benign_data, x='radius_mean', y='texture_mean', color='green', label='iyi', ax=ax,
                        alpha=0.4)
        ax.legend()
        return fig

    def create_bar_graph(self):
        df_graph = pd.DataFrame({
            "Siniflar": self.y.value_counts().index,
            "Sinifların Miktarı": self.y.value_counts().values
        })
        fig, ax = plt.subplots()
        sns.barplot(data=df_graph, x="Siniflar", y="Sinifların Miktarı")
        return fig
