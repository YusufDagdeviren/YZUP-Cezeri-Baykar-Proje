import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class App:
    def __init__(self):
        self.dataset_name: None
        self.classifier_name: None
        self.Init_Streamlit_page()
        self.params = dict()
        self.clf = None
        self.X, self.y = None, None

    def Init_Streamlit_page(self):
        st.title("YZUP-Cezeri-Baykar-Proje")
        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer', )
        )
    def run(self):
        self.get_dataset()
        # self.add_parameter_ui()
        self.generate()

    def get_dataset(self):
        data = None
        if self.dataset_name == 'Breast Cancer':
            data = pd.read_csv("data.csv")
            st.write("# Görev 1")
            st.write("### DataFrame'in ilk 10 satiri")
            st.table(data.head(10))
            st.write("### DataFrame'in Sütunları")
            st.write(data.columns.tolist())
            data = data.drop(['id', 'Unnamed: 32'], axis=1)
            st.write("# Görev 2")
            st.write("### DataFrame'in gereksiz satirlarini sildikten sonra son 10 satiri")
            st.table(data.tail(10))
            data['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)
            fig = self.create_correlation_matrix(data)
            st.write("### Korelasyon Matrisi Çizimi")
            st.pyplot(fig)

    def create_correlation_matrix(self, data):
        malignant_data = data[data['diagnosis'] == 1]
        benign_data = data[data['diagnosis'] == 0]
        self.y = data['diagnosis']
        self.X = data.drop('diagnosis', axis=1)
        fig, ax = plt.subplots()
        sns.scatterplot(data=malignant_data, x='radius_mean', y='texture_mean', color='red', label='kotu',
                        ax=ax, alpha=0.4)
        sns.scatterplot(data=benign_data, x='radius_mean', y='texture_mean', color='green', label='iyi', ax=ax, alpha=0.4)
        ax.legend()
        return fig
    def get_classifier(self):
        pass
    def generate(self):
        self.get_classifier()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)

