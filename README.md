# YZUP-Cezeri-Baykar-Proje

Bu proje, Milli Teknoloji Akademisi Yapay Zeka Uzmanlık Programı kapsamında Baykar-Cezeri  şirketlerinin  4 haftalık eğitimi sonrasında modül projesi olarak istemiş olduğu Python ve Streamlit kullanılarak bir Streamlit uygulaması oluşturmayı ve bir veri seti üzerinde makine öğrenmesi modellerini eğitmeyi ve analiz etmeyi içerir.

## Görevler

### Görev  1: Data Analizi

Veri setini yüklemek ve ilk 10 satırı ve sütunları göstermek için get_dataset fonksiyonu kullanılır. Bu fonksiyon, kullanıcının bir CSV dosyasını yüklemesini sağlar ve veri setinin ilk 10 satırını ve sütunlarını gösterir.
```python
def get_dataset(self):
    self.uploaded_file = st.file_uploader("Upload Breast Cancer CSV file", type=["csv"])
    if self.uploaded_file is not None:
        data = pd.read_csv(self.uploaded_file)
        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            (self.uploaded_file.name,)
        )
        st.write("# Görev  1")
        st.write("### DataFrame'in ilk  10 satiri")
        st.table(data.head(10))
        st.write("### DataFrame'in Sütunları")
        st.write(data.columns.tolist())
        ...
```

### Görev  2: Data Ön  İşlemleri

Veri setindeki gereksiz sütunları silmek ve 'diagnosis' sütununu 1 ve 0 değerlerine dönüştürmek için get_dataset fonksiyonu içindeki kodlar kullanılır. Ayrıca, veri setindeki imbalanced classification problemi, RandomOverSampler kullanılarak çözülür. X değişkenlerine SVM ve KNN modelleri için veri normalizasyonu uygulanır
```python
...
data = data.drop(['id', 'Unnamed:  32'], axis=1)
st.write("# Görev  2")
st.write("### DataFrame'in gereksiz sütunlarını sildikten sonra son  10 satiri")
st.table(data.tail(10))
data['diagnosis'].replace({'M':  1, 'B':  0}, inplace=True)
self.y = data['diagnosis']
self.X = data.drop('diagnosis', axis=1)
...
```
```python
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

```

### Görev  3: Model Gerçekleşmesi

Model eğitimi ve test seti oluşturma işlemleri generate fonksiyonu içinde gerçekleştirilir. Bu fonksiyon, kullanıcının seçtiği sınıflandırıcıya göre bir model eğitir ve veri seti üzerinde bir test seti oluşturur.
```python
def generate(self):
    if self.uploaded_file is not None:
        self.get_classifier()
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=1234)
        self.clf.fit(X_train, y_train)
        ...
```

### Görev  4: Model Analizi

Modelin performansı, doğruluk, duyarlılık, hatırlama ve F1 skoru gibi metriklerle analiz edilir. Ayrıca, bir karmaşık matris çizimi ile modelin karışıklığını gösterir.
```python
...
y_predict = self.clf.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict)
recall = recall_score(y_test, y_predict)
f1 = f1_score(y_test, y_predict)
st.write(f"# Görev  4")
st.write(f"Classifier = {self.classifier_name}")
st.write(f"Model Accuracy: {accuracy}")
st.write(f" Model Precision: {precision}")
st.write(f" Model Recall: {recall}")
st.write(f" Model F1 Score: {f1}")
...
```

### Görev  5: Streamlit Entegrasyonu

Streamlit, veri analizi ve model eğitimi için kullanılan bir web uygulamasıdır. Kullanıcılar, veri setini yükleyebilir ve sınıflandırıcıyı seçebilir. Model eğitildikten sonra, sonuçlar Streamlit uygulaması üzerinden görüntülenir.
```python
class App:
    def __init__(self):
        ...
        self.Init_Streamlit_page()

    def Init_Streamlit_page(self):
        st.title("YZUP-Cezeri-Baykar-Proje")
        self.classifier_name = st.sidebar.selectbox(
            'Select Classifier',
            ("KNN", "SVM", "Naive Bayes")
        )
        ...
```

## Docker ile Proje  Çalıştırma

Projeyi Docker kullanarak  çalıştırmak için aşağıdaki adımları izleyin:

1. Docker'ın yüklü olduğundan ve  çalıştığından emin olun.
2. Terminalde projenin kök dizinine gidin.
3. Aşağıdaki komutu  çalıştırarak Docker görüntüsünü oluşturun:

   ```
   docker build -t yzup-cezeri-baykar-proje .
   ```

4. Docker görüntüsünü  çalıştırmak için aşağıdaki komutu kullanın:

   ```
   docker run -p  8501:8501 yzup-cezeri-baykar-proje
   ```

5. Tarayıcınızda `http://localhost:8501` adresine giderek Streamlit uygulamasını açın.

## Yerel  Çalıştırma

Projeyi yerel olarak  çalıştırmak için aşağıdaki adımları izleyin:

1. Python  3.9 ve gerekli paketlerin yüklü olduğundan emin olun.
2. Terminalde projenin kök dizinine gidin.
3. Gerekli paketleri yüklemek için aşağıdaki komutu  çalıştırın:

   ```
   pip install -r requirements.txt
   ```

4. Streamlit uygulamasını başlatmak için aşağıdaki komutu  çalıştırın:

   ```
   streamlit run main.py
   ```

5. Tarayıcınızda `http://localhost:8501` adresine giderek Streamlit uygulamasını açın.
