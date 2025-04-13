#!/usr/bin/env python
# coding: utf-8

# In[165]:


pip install numpy==1.23.5


# In[166]:


pip install pandas==1.5.3


# In[167]:


pip install scikit-learn==1.2.2   


# In[168]:


pip install scipy==1.10.1          


# In[169]:


pip install tensorflow==2.10.1    


# In[10]:


pip install keras==2.10.0       


# In[11]:


pip install numexpr==2.8.4        


# In[12]:


pip install bottleneck==1.3.7   


# In[170]:


import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import nltk

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# In[11]:



nltk.download('stopwords')

# Инициализация морфологического анализатора и стоп-слов
morph = MorphAnalyzer()
russian_stopwords = stopwords.words('russian')
additional_stopwords = ['тд', 'тп', 'тк', 'зарплата', 'вакансия', 'работодатель', 
                       'компания', 'работ', 'работа', 'обязанность', 'требование']
russian_stopwords.extend(additional_stopwords)

def preprocess_text(text):
    """Функция для предобработки текста"""
    if not isinstance(text, str):
        return ""
    
    # Удаление HTML-тегов
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Удаление спецсимволов и цифр
    text = re.sub(r'[^а-яА-ЯёЁa-zA-Z\-]', ' ', text)
    
    # Приведение к нижнему регистру
    text = text.lower()
    
    # Токенизация
    words = text.split()
    
    # Лемматизация и удаление стоп-слов
    processed_words = []
    for word in words:
        if len(word) > 2 and word not in russian_stopwords:
            parsed_word = morph.parse(word)[0]
            lemma = parsed_word.normal_form
            processed_words.append(lemma)
    
    return ' '.join(processed_words)

def prepare_dataset(data):
    """Подготовка датасета для анализа"""
    # Создаем DataFrame из данных
    df = pd.DataFrame(data)
    
    # Обработка текстовых полей
    text_fields = ['name_desc', 'description_desc', 'branded_description_desc','area.name_desc', 'key_skills','vacancies_url','employer.name','salary.from']
    
    for field in text_fields:
        if field in df.columns:
            if field == 'key_skills':
                # Обработка ключевых навыков (список словарей)
                df[field] = df[field].apply(lambda x: ' '.join([skill['name'] for skill in x]) if isinstance(x, list) else '')
            df[f'processed_{field}'] = df[field].apply(preprocess_text)
    
    # Создаем объединенное текстовое поле для анализа
    df['combined_text'] = ''
    for field in text_fields:
        if f'processed_{field}' in df.columns:
            df['combined_text'] += ' ' + df[f'processed_{field}']
    
    return df

def calculate_similarity(df, target_vacancy_idx):
    """Расчет косинусной схожести между вакансиями"""
    # Инициализация TF-IDF векторайзера
    vectorizer = TfidfVectorizer()
    
    # Преобразование текстов в TF-IDF матрицу
    tfidf_matrix = vectorizer.fit_transform(df['combined_text'])
    
    # Расчет косинусной схожести
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # Получение схожести для целевой вакансии
    sim_scores = list(enumerate(cosine_sim[target_vacancy_idx]))
    
    # Сортировка вакансий по схожести
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    return sim_scores

def recommend_vacancies(df, sim_scores, top_n=5):
    """Рекомендация вакансий на основе схожести"""
    # Получаем индексы топ-N вакансий
    top_indices = [i[0] for i in sim_scores[1:top_n+1]]
    
    # Возвращаем информацию о рекомендуемых вакансиях
    return df.iloc[top_indices][['id_desc', 'name_desc', 'description_desc']]

def load_and_concatenate_excel_files(directory):
    """Загрузка и объединение всех Excel файлов в указанной директории"""
    all_data = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.xlsx'):
            file_path = os.path.join(directory, file_name)
            df = pd.read_excel(file_path)
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Загрузка и объединение данных из Excel файлов
directory_path = r"C:\Users\Victoria\OneDrive - НИУ Высшая школа экономики\Рабочий стол\Python\analyst"
data = load_and_concatenate_excel_files(directory_path)

# Подготовка данных
df = prepare_dataset(data)
df_vacancies = df
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Создаем и обучаем векторайзер на ваших данных
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(df['combined_text'])

# Сохраняем векторайзер в файл
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
import ast

# Assuming df_vacancies is already defined
df_vacancies['salary.from'] = df_vacancies['salary.from'].fillna(50000)

def extract_name(work_format):
    # Check if the input is NaN
    if pd.isna(work_format) or work_format == '[]':
        return None
    else:
        # Convert the string to a list of dictionaries
        work_format_list = ast.literal_eval(work_format)
        if work_format_list:
            return work_format_list[0]['name']  # Extract 'name' from the first element
        else:
            return None

# Apply the function to the 'work_format' column
df_vacancies['name'] = df_vacancies['work_format'].apply(extract_name)

# Print the result
print(df_vacancies[['work_format', 'name']])

# Drop unnecessary columns
df_vacancies = df_vacancies.drop(columns=['relations_desc', 'description_desc', 'branded_description_desc'])

# Display the first 5 rows of the DataFrame
pd.set_option('display.max_columns', None)

columns_to_drop = [
    'salary.currency',
    'salary.gross',
    'salary_range.currency',
    'salary_range.gross',
    'salary_range.mode.id',
    'salary_range.mode.name',
    'contacts.phones',
    'test.required',
    'salary_range.frequency.id',
    'salary_range.frequency.name',
    'trusted',
    'accredited_it_employer',
    'type',
    'relations_comp',
    'branding.type',
    'processed_key_skills'
]

df_vacancies.drop(columns=columns_to_drop, inplace=True)

# Очищаем строки от символов \xa0
df_vacancies = df_vacancies.replace('\xa0', ' ', regex=True)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Function to create and train a simple LSTM model
def create_and_train_lstm_model(vacancies_df):
    # Ensure that the columns contain valid data
    vacancies_df['combined_text'] = vacancies_df['combined_text'].astype(str)
    vacancies_df['name_desc'] = vacancies_df['name_desc'].astype(str)

    # Drop any rows where 'name_desc' is NaN or empty
    vacancies_df = vacancies_df[vacancies_df['name_desc'].notna() & (vacancies_df['name_desc'] != '')]

    texts = vacancies_df['combined_text'].values
    labels = vacancies_df['name_desc'].values

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=100)
    y = pd.get_dummies(labels).values  # One-hot encoding

    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(len(np.unique(labels)), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    model.save('lstm_model.h5')
    return model, tokenizer


# In[176]:


# Загрузка и объединение данных из Excel файлов
directory_path = r"C:\Users\Victoria\OneDrive - НИУ Высшая школа экономики\Рабочий стол\Python\analyst"
data = load_and_concatenate_excel_files(directory_path)

# Подготовка данных
df = prepare_dataset(data)


# In[31]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Function to create and train a simple LSTM model
def create_and_train_lstm_model(vacancies_df):
    # Ensure that the columns contain valid data
    vacancies_df['combined_text'] = vacancies_df['combined_text'].astype(str)
    vacancies_df['name_desc'] = vacancies_df['name_desc'].astype(str)

    # Drop any rows where 'name_desc' is NaN or empty
    vacancies_df = vacancies_df[vacancies_df['name_desc'].notna() & (vacancies_df['name_desc'] != '')]

    texts = vacancies_df['combined_text'].values
    labels = vacancies_df['name_desc'].values

    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=100)
    y = pd.get_dummies(labels).values  # One-hot encoding

    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(len(np.unique(labels)), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X, y, epochs=5, batch_size=32, validation_split=0.2)

    model.save('lstm_model.h5')
    return model, tokenizer


# # C Макколлом

# In[194]:


import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import re
import os
import json
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tkinter.constants import BOTH
from scipy.special import expit as sigmoid


class JobSearchSystem:
    def __init__(self, root, vacancies_df):
        self.root = root
        self.root.title("Система поиска вакансий с моделью МакКолла")
        self.root.geometry("900x650")
        
        # Загрузка данных
        self.vacancies_df = vacancies_df
        self.preprocess_data()
        
        # Инициализация моделей
        self.init_models()
        
        # Параметры модели МакКолла
        self.gamma = 0.5  
        self.beta = 0.95  # Коэффициент дисконтирования
        self.k = 1.0      # Параметр крутизны сигмоиды
        
        # Переменные интерфейса
        self.job_title = tk.StringVar()
        self.resume_text = ""
        self.experience = tk.StringVar(value="Нет опыта")
        self.city = tk.StringVar()
        self.unemployed = tk.BooleanVar(value=False)
        self.same_field = tk.BooleanVar(value=True)
        self.desired_job = tk.StringVar()
        self.desired_salary = tk.StringVar()
        self.employment_type = tk.StringVar(value="Полный день")
        self.work_format = tk.StringVar(value="Не важно")
        self.search_costs = tk.DoubleVar(value=10.0)  # Затраты на поиск (% от з/п)
        self.unemployment_benefit = tk.DoubleVar(value=30.0)  # Пособие по безработице (% от з/п)
        # Создание интерфейса
        self.create_widgets()
    
    def preprocess_data(self):
        """Предварительная обработка данных"""
        # Заполнение пропущенных значений
        self.vacancies_df['salary.from'] = self.vacancies_df['salary.from'].fillna(0)
        self.vacancies_df['salary.to'] = self.vacancies_df['salary.to'].fillna(0)
        self.vacancies_df['experience.name'] = self.vacancies_df['experience.name'].fillna('Нет опыта')
        
        # Очистка текстовых полей
        text_columns = ['name', 'processed_description_desc']
        for col in text_columns:
            if col in self.vacancies_df.columns:
                self.vacancies_df[col] = self.vacancies_df[col].fillna('').apply(self.clean_text)
        
        # Создаем комбинированный текст для анализа
        self.vacancies_df['combined_text'] = (
            self.vacancies_df.get('name', '') + " " + 
            self.vacancies_df.get('processed_description_desc', '')
        ).apply(self.clean_text)
        
        # Нормализация зарплат
        self.vacancies_df['salary_mid'] = (self.vacancies_df['salary.from'] + self.vacancies_df['salary.to']) / 2
        
        # Подготовка данных для Item-Based CF
        self.prepare_item_based_data()
        
        # Подготовка данных для LSTM
        self.prepare_lstm_data()

        self.calculate_competition_level()   
    def calculate_competition_level(self):
        """Расчет уровня конкуренции на рынке труда"""

        competition_levels = {
            'Разработка': 2.5, 
            'Аналитика': 1.8,
            'Менеджмент': 1.2,
            'Дизайн': 1.5,
            'Другое': 1.0
        }
        
        self.vacancies_df['gamma'] = self.vacancies_df['category'].map(competition_levels)
    
    def prepare_lstm_data(self):
        """Подготовка данных для LSTM модели"""
        # Создаем категории на основе названий должностей
        self.vacancies_df['category'] = self.vacancies_df['name'].apply(
            lambda x: self.detect_category(x)
        )
        
        # Кодируем категории
        self.label_encoder = LabelEncoder()
        self.vacancies_df['category_encoded'] = self.label_encoder.fit_transform(
            self.vacancies_df['category']
        )
        
        # Сохраняем количество классов
        self.num_classes = len(self.label_encoder.classes_)
    
    def detect_category(self, job_title):
        """Определение категории на основе названия должности"""
        if not isinstance(job_title, str):
            return "Другое"
            
        job_title = job_title.lower()
        if any(word in job_title for word in ['разработчик', 'developer', 'программист', 'engineer']):
            return 'Разработка'
        elif any(word in job_title for word in ['аналитик', 'analyst', 'data scientist']):
            return 'Аналитика'
        elif any(word in job_title for word in ['менеджер', 'manager', 'руководитель']):
            return 'Менеджмент'
        elif any(word in job_title for word in ['дизайн', 'designer']):
            return 'Дизайн'
        else:
            return 'Другое'
    
    def prepare_item_based_data(self):
        """Подготовка данных для Item-Based Collaborative Filtering"""
        # Создаем матрицу признаков для вакансий
        self.job_features = pd.get_dummies(
            self.vacancies_df[['experience.name', 'schedule.name']], 
            columns=['experience.name', 'schedule.name'])

        # Добавляем нормализованную зарплату
        self.job_features['salary_norm'] = (
            self.vacancies_df['salary_mid'] - self.vacancies_df['salary_mid'].min()
        ) / (self.vacancies_df['salary_mid'].max() - self.vacancies_df['salary_mid'].min())

        # Заполняем пропуски
        self.job_features = self.job_features.fillna(0)

        # Создаем словарь для быстрого поиска индексов по названию должности
        self.job_title_to_index = {
            self.clean_text(title): idx 
            for idx, title in enumerate(self.vacancies_df['name'])
        }

        # Создаем матрицу сходства для Item-Based CF
        self.item_similarity = 1 - pairwise_distances(
            self.job_features, 
            metric='cosine'
        )
    
    def clean_text(self, text):
        """Очистка текста"""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def init_models(self):
        """Инициализация моделей машинного обучения"""
        self.lstm_model = None
        self.tokenizer = Tokenizer(num_words=10000)
        
        # Обучаем токенизатор на всех текстах
        texts = self.vacancies_df['combined_text'].tolist()
        self.tokenizer.fit_on_texts(texts) 
        
        # Пытаемся загрузить предобученную модель
        try:
            if os.path.exists('lstm_model.h5'):
                self.lstm_model = load_model('lstm_model.h5')
                # Загружаем историю обучения для отображения графиков
                if os.path.exists('training_history.json'):
                    with open('training_history.json', 'r') as f:
                        self.training_history = json.load(f)
            else:
                self.train_lstm_model()
        except Exception as e:
            print(f"Ошибка при загрузке модели: {e}")
            self.train_lstm_model()
        
        # Подготовка модели рекомендаций
        self.prepare_recommendation_model()
    
    def train_lstm_model(self):
        """Обучение LSTM модели на текстах вакансий"""
        texts = self.vacancies_df['combined_text'].tolist()
        labels = self.vacancies_df['category_encoded'].values
        
        # Преобразование текста в последовательности
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=200)
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, labels, test_size=0.2, random_state=42
        )
        
        # Преобразование меток в one-hot encoding
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        y_test = to_categorical(y_test, num_classes=self.num_classes)
        
        # Создание модели
        self.lstm_model = Sequential([
            Embedding(input_dim=10000, output_dim=128, input_length=200),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
            LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        self.lstm_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Коллбэки
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('lstm_model.h5', save_best_only=True)
        ]
        
        # Обучение модели
        history = self.lstm_model.fit(
            X_train,
            y_train,
            batch_size=64,
            epochs=20,
            validation_data=(X_test, y_test),
            callbacks=callbacks
        )
        
        # Сохраняем историю обучения
        self.training_history = history.history
        with open('training_history.json', 'w') as f:
            json.dump(self.training_history, f)
        
        # Визуализация обучения
        self.plot_training_history()
    
    def plot_training_history(self):
        """Визуализация процесса обучения модели"""
        plt.figure(figsize=(12, 5))
        
        # График точности
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['accuracy'], label='Точность на обучении')
        plt.plot(self.training_history['val_accuracy'], label='Точность на валидации')
        plt.title('Точность модели')
        plt.ylabel('Точность')
        plt.xlabel('Эпоха')
        plt.legend()
        
        # График потерь
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['loss'], label='Потери на обучении')
        plt.plot(self.training_history['val_loss'], label='Потери на валидации')
        plt.title('Потери модели')
        plt.ylabel('Потери')
        plt.xlabel('Эпоха')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
    
    def prepare_recommendation_model(self):
        """Подготовка модели рекомендаций на основе TF-IDF и косинусного сходства"""
        # Создаем TF-IDF матрицу для текстов вакансий
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.vacancies_df['combined_text'])
        
        # Модель KNN для рекомендаций
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.knn_model.fit(self.tfidf_matrix)
    
    def create_widgets(self):
        """Создание интерфейса пользователя"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=BOTH, expand=True)

        # Блок названия должности
        ttk.Label(main_frame, text="Название текущей/желаемой должности:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.job_title, width=50).grid(row=0, column=1, sticky=tk.W, pady=5)

        # Блок резюме
        resume_frame = ttk.LabelFrame(main_frame, text="Резюме", padding=10)
        resume_frame.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=10)

        ttk.Button(resume_frame, text="Ввести резюме", command=self.open_resume_editor).pack(side=tk.LEFT, padx=5)
        ttk.Button(resume_frame, text="Загрузить из файла", command=self.load_resume_file).pack(side=tk.LEFT, padx=5)

        # Блок опыта работы
        ttk.Label(main_frame, text="Опыт работы:").grid(row=2, column=0, sticky=tk.W, pady=5)
        experience_options = ["Нет опыта", "От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"]
        ttk.Combobox(main_frame, textvariable=self.experience, values=experience_options, state="readonly").grid(
            row=2, column=1, sticky=tk.W, pady=5)

        # Блок города
        ttk.Label(main_frame, text="Город:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.city, width=30).grid(row=3, column=1, sticky=tk.W, pady=5)

        # Чекбокс поиска в той же сфере
        ttk.Checkbutton(main_frame, text="Искать в той же сфере деятельности", variable=self.same_field,
                       command=self.toggle_job_field).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Блок типа занятости
        ttk.Label(main_frame, text="Тип занятости:").grid(row=5, column=0, sticky=tk.W, pady=5)
        employment_options = ["Полный день", "Удаленная работа", "Не важно"]
        ttk.Combobox(main_frame, textvariable=self.employment_type, values=employment_options, state="readonly", width=20).grid(row=5, column=1, sticky=tk.W, pady=5)

        # Блок формата работы
        ttk.Label(main_frame, text="Формат работы:").grid(row=6, column=0, sticky=tk.W, pady=5)
        format_options = ["На месте работодателя", "Удалённо", "Гибрид", "Не важно"]
        ttk.Combobox(main_frame, textvariable=self.work_format, values=format_options, state="readonly", width=20).grid(row=6, column=1, sticky=tk.W, pady=5)

        # Блок желаемой зарплаты
        ttk.Label(main_frame, text="Желаемая зарплата (руб.):").grid(row=7, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.desired_salary, width=15).grid(row=7, column=1, sticky=tk.W, pady=5)

        # Блок параметров модели МакКолла
        params_frame = ttk.LabelFrame(main_frame, text="Параметры модели", padding=10)
        params_frame.grid(row=8, column=0, columnspan=2, sticky=tk.EW, pady=10)

        ttk.Label(params_frame, text="Затраты на поиск (% от з/п):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Scale(params_frame, from_=0, to=100, variable=self.search_costs, orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=tk.W, padx=5)

        ttk.Label(params_frame, text="Пособие по безработице (% от з/п):").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Scale(params_frame, from_=0, to=100, variable=self.unemployment_benefit, orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=tk.W, padx=5)

        # Чекбокс безработного статуса
        ttk.Checkbutton(main_frame, text="Я в настоящее время безработный", variable=self.unemployed).grid(
            row=9, column=0, columnspan=2, sticky=tk.W, pady=5)

        # Кнопка поиска
        ttk.Button(main_frame, text="Найти вакансии", command=self.search_jobs).grid(
            row=10, column=0, columnspan=2, pady=20)

        # Кнопка выхода
        ttk.Button(main_frame, text="Выход", command=self.root.quit).grid(
            row=11, column=0, columnspan=2, pady=10)
    
    def toggle_job_field(self):
        """Обработчик изменения чекбокса 'Искать в той же сфере'"""
        if not self.same_field.get():
            self.open_desired_job_dialog()
    
    def open_desired_job_dialog(self):
        """Открытие диалога для указания желаемой должности"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Укажите желаемую должность")
        dialog.geometry("400x200")

        ttk.Label(dialog, text="Введите желаемую должность:").pack(pady=10)
        entry = ttk.Entry(dialog, textvariable=self.desired_job, width=40)
        entry.pack(pady=5)
        entry.focus_set()

        def save_and_close():
            if not self.desired_job.get():
                messagebox.showerror("Ошибка", "Пожалуйста, введите желаемую должность")
                return
            dialog.destroy()

        ttk.Button(dialog, text="Сохранить", command=save_and_close).pack(pady=15)
    
    def open_resume_editor(self):
        """Открытие редактора резюме"""
        editor = tk.Toplevel(self.root)
        editor.title("Редактор резюме")
        editor.geometry("600x400")

        text_area = ScrolledText(editor, wrap=tk.WORD, width=70, height=20)
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        if self.resume_text:
            text_area.insert(tk.END, self.resume_text)

        def save_resume():
            self.resume_text = text_area.get("1.0", tk.END)
            editor.destroy()
            messagebox.showinfo("Сохранено", "Резюме успешно сохранено")

        ttk.Button(editor, text="Сохранить", command=save_resume).pack(pady=5)
    
    def load_resume_file(self):
        """Загрузка резюме из файла"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Текстовые файлы", "*.txt"), ("Документы Word", "*.docx"), ("PDF", "*.pdf")])
        if file_path:
            try:
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as file:
                        self.resume_text = file.read()
                else:
                    self.resume_text = f"Файл {file_path} загружен (необходима обработка для извлечения текста)"
                messagebox.showinfo("Успех", "Резюме успешно загружено")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")
    def search_jobs(self):
        """Поиск вакансий"""
        # Проверка обязательных полей
        if not self.job_title.get() and not self.same_field.get():
            messagebox.showerror("Ошибка", "Пожалуйста, укажите название должности или выберите 'Искать в той же сфере деятельности'")
            return

        if not self.resume_text:
            messagebox.showerror("Ошибка", "Пожалуйста, введите или загрузите резюме")
            return

        if not self.city.get():
            messagebox.showerror("Ошибка", "Пожалуйста, укажите город")
            return

        if not self.desired_salary.get() or not self.desired_salary.get().isdigit():
            messagebox.showerror("Ошибка", "Пожалуйста, укажите валидную желаемую зарплату")
            return

        # Проверка для случая, когда снята галочка "Искать в той же сфере"
        if not self.same_field.get() and not self.desired_job.get():
            messagebox.showerror("Ошибка", "Пожалуйста, укажите желаемую должность")
            return

        # Подготовка данных для поиска
        search_data = {
            "job_title": self.job_title.get(),
            "resume_text": self.resume_text,
            "experience": self.experience.get(),
            "city": self.city.get(),
            "same_field": self.same_field.get(),
            "desired_job": self.desired_job.get() if not self.same_field.get() else "",
            "employment_type": self.employment_type.get(),
            "work_format": self.work_format.get(),
            "desired_salary": int(self.desired_salary.get()),
            "unemployed": self.unemployed.get(),
            "search_costs": self.search_costs.get() / 100,  # Преобразуем в долю
            "unemployment_benefit": self.unemployment_benefit.get() / 100  # Преобразуем в долю
        }

        # Инициализация filtered_jobs перед использованием
        filtered_jobs = pd.DataFrame()

        # Поиск вакансий
        if self.same_field.get():
            # Используем текст резюме для поиска в той же сфере
            filtered_jobs = self.find_relevant_jobs_by_resume(search_data)
        else:
            # Используем желаемую должность для поиска в другой сфере
            filtered_jobs = self.find_jobs_in_different_field(search_data)

        # Проверка на пустые результаты
        if filtered_jobs.empty:
            messagebox.showinfo("Результаты", "Подходящих вакансий не найдено")
            return

        # Применяем модель МакКолла для расчета вероятности отклика и пользы
        filtered_jobs = self.apply_mccall_model(filtered_jobs, search_data)

        # Отображение результатов
        self.display_results(filtered_jobs)
    
    def apply_mccall_model(self, jobs_df, search_data):
        """Применение модели МакКолла для расчета вероятности отклика и пользы"""
        # Копируем DataFrame, чтобы избежать предупреждений
        jobs_df = jobs_df.copy()
        
        # Рассчитываем среднюю зарплату для каждой вакансии
        jobs_df['salary_avg'] = jobs_df.apply(
            lambda row: (row['salary.from'] + row['salary.to']) / 2 if row['salary.from'] and row['salary.to'] 
            else row['salary.from'] if row['salary.from'] else row['salary.to'], axis=1)
        
        # Получаем желаемую зарплату
        w_star = search_data['desired_salary']
        
        # Рассчитываем вероятность отклика для каждой вакансии
        jobs_df['response_prob'] = jobs_df.apply(
            lambda row: self.calculate_response_probability(
                row, 
                search_data['resume_text'], 
                w_star,
                search_data['search_costs'],
                search_data['unemployment_benefit']
            ), axis=1
        )
        
        # Рассчитываем полезность перехода для каждой вакансии
        jobs_df['transition_value'] = jobs_df.apply(
            lambda row: self.calculate_transition_value(
                row['salary_avg'],
                search_data['search_costs'],
                search_data['unemployment_benefit'],
                w_star
            ), axis=1
        )
        
        # Сортируем по комбинированному показателю (вероятность * полезностьность)
        jobs_df['combined_score'] = jobs_df['response_prob'] * jobs_df['transition_value']
        jobs_df = jobs_df.sort_values('combined_score', ascending=False)
        
        return jobs_df
    
    def calculate_response_probability(self, job_row, resume_text, w_star, c_t, b_t):
        """Расчет вероятности положительного отклика на вакансию по модели МакКолла"""
        # Рассчитываем косинусное сходство между резюме и вакансией
        job_text = job_row['name'] + " " + job_row.get('processed_description_desc', '')
        job_vector = self.tfidf_vectorizer.transform([job_text])
        resume_vector = self.tfidf_vectorizer.transform([resume_text])
        similarity = cosine_similarity(resume_vector, job_vector)[0][0]
        
        # Получаем уровень конкуренции для категории вакансии
        gamma = job_row.get('gamma', 1.0)
        
        # Получаем предлагаемую зарплату
        w_t = (job_row['salary.from'] + job_row['salary.to']) / 2 if job_row['salary.from'] and job_row['salary.to'] else job_row['salary.from'] if job_row['salary.from'] else job_row['salary.to']
        
        # Рассчитываем вероятность отклика
        # P(vacancy_t) = σ(k * (similarity + γ - c(t) + b(t) - w*/w_t))
        prob = sigmoid(
            self.k * (similarity + gamma - c_t + b_t - (w_star / w_t if w_t > 0 else 1))
        )
        
        return prob
    
    def calculate_transition_value(self, w_t, c_t, b_t, w_star):
        """Расчет полезности перехода на новую работу"""
        # V(w,t) = w + β[(1-β)V_e_current(w,t+1) + β*E_max{V_e_current(w,t+1), V_e_new(w',t+1), Vu(t+1)}]
        
        # Для упрощения модели считаем, что:
        # V_e_current = текущая зарплата (или пособие по безработице)
        # V_e_new = зарплата на новой работе
        # Vu = пособие по безработице
        
        # Текущая ценность (если безработный - пособие, иначе - желаемая зарплата)
        V_current = b_t * w_star if self.unemployed.get() else w_star
        
        # Ценность новой работы
        V_new = w_t / (1 - self.beta) - c_t * w_t
        
        # Ценность безработицы
        V_u = b_t * w_star / (1 - self.beta)
        
        # Максимальная ценность
        max_value = max(V_current, V_new, V_u)
        
        # Общая ценность перехода
        transition_value = w_t + self.beta * (
            (1 - self.beta) * V_current + 
            self.beta * max_value
        )
        
        return transition_value
    def find_jobs_in_different_field(self, search_data):
        """Поиск вакансий в другой сфере деятельности"""
        # Получаем желаемую должность
        target_job = search_data['desired_job']
        if not target_job:
            return pd.DataFrame()

        # Очищаем текст
        target_job_clean = self.clean_text(target_job)

        # Базовые фильтры
        filtered_df = self.vacancies_df.copy()

        # Фильтр по городу
        if search_data['city']:
            filtered_df = filtered_df[filtered_df['area.name_desc'].str.contains(
                search_data['city'], case=False, na=False)]

        # Фильтр по опыту работы
        experience_map = {
            "Нет опыта": ["Нет опыта"],
            "От 1 года до 3 лет": ["От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"],
            "От 3 до 6 лет": ["От 3 до 6 лет", "Более 6 лет"],
            "Более 6 лет": ["Более 6 лет"]
        }
        if search_data['experience'] in experience_map:
            filtered_df = filtered_df[filtered_df['experience.name'].isin(experience_map[search_data['experience']])]

        # Фильтр по типу занятости
        if search_data['employment_type'] != "Не важно":
            filtered_df = filtered_df[filtered_df['schedule.name'].str.contains(
                search_data['employment_type'], case=False, na=False)]

        # Фильтр по желаемой зарплате
        desired_salary = search_data['desired_salary']
        filtered_df = filtered_df[
            (filtered_df['salary.from'] <= desired_salary * 1.5) | 
            (filtered_df['salary.to'] >= desired_salary * 0.7)]

        # Если после базовых фильтров ничего не осталось, возвращаем пустой DataFrame
        if filtered_df.empty:
            return pd.DataFrame()

        # Ищем вакансии, похожие на желаемую должность
        similar_indices = []

        # 1. Поиск по точному или частичному совпадению в названии
        for idx, title in enumerate(filtered_df['name']):
            clean_title = self.clean_text(str(title))
            if target_job_clean in clean_title or clean_title in target_job_clean:
                similar_indices.append(idx)
                if len(similar_indices) >= 5:
                    break

        # 2. Если не нашли по названию, ищем по описанию
        if not similar_indices:
            for idx, desc in enumerate(filtered_df['processed_description_desc']):
                clean_desc = self.clean_text(str(desc))
                if target_job_clean in clean_desc:
                    similar_indices.append(idx)
                if len(similar_indices) >= 3:
                    break

        # 3. Если совсем не нашли, используем TF-IDF
        if not similar_indices:
            target_vector = self.tfidf_vectorizer.transform([target_job_clean])
            job_vectors = self.tfidf_vectorizer.transform(filtered_df['combined_text'])
            similarities = cosine_similarity(target_vector, job_vectors)[0]
            similar_indices = np.argsort(-similarities)[:10].tolist()

        # Фильтруем только по валидным индексам
        valid_indices = [i for i in similar_indices if i < len(filtered_df)]

        if valid_indices:
            filtered_df = filtered_df.iloc[valid_indices].copy()
        else:
            return pd.DataFrame()

        # Вычисляем сходство с желаемой должностью
        if not filtered_df.empty:
            job_texts = filtered_df['name'] + " " + filtered_df['processed_description_desc']
            job_vectors = self.tfidf_vectorizer.transform(job_texts)
            target_vector = self.tfidf_vectorizer.transform([target_job_clean])
            similarities = cosine_similarity(target_vector, job_vectors)[0]
            filtered_df['similarity'] = similarities
            filtered_df = filtered_df.sort_values('similarity', ascending=False)

        return filtered_df.head(10)
    
    
    def find_relevant_jobs_by_resume(self, search_data):
        """Поиск релевантных вакансий по тексту резюме с использованием TF-IDF и LSTM"""
        # Базовые фильтры
        filtered_df = self.vacancies_df.copy()

        # Фильтр по городу
        if search_data['city']:
            if 'area.name' in filtered_df.columns:
                filtered_df = filtered_df[
                    filtered_df['area.name'].str.contains(
                        search_data['city'], 
                        case=False, 
                        na=False)]
        
        # Фильтр по опыту работы
        experience_map = {
            "Нет опыта": ["Нет опыта"],
            "От 1 года до 3 лет": ["От 1 года до 3 лет", "От 3 до 6 лет", "Более 6 лет"],
            "От 3 до 6 лет": ["От 3 до 6 лет", "Более 6 лет"],
            "Более 6 лет": ["Более 6 лет"]
        }
        if search_data['experience'] in experience_map:
            filtered_df = filtered_df[filtered_df['experience.name'].isin(experience_map[search_data['experience']])]
        
        # Фильтр по типу занятости
        if search_data['employment_type'] != "Не важно":
            filtered_df = filtered_df[filtered_df['schedule.name'].str.contains(
                search_data['employment_type'], case=False, na=False)]
        
        # Фильтр по желаемой зарплате
        desired_salary = search_data['desired_salary']
        filtered_df = filtered_df[
            (filtered_df['salary.from'] <= desired_salary * 1.5) | 
            (filtered_df['salary.to'] >= desired_salary * 0.7)]
        
        # Если после базовых фильтров осталось слишком много вакансий, применяем TF-IDF и LSTM
        if len(filtered_df) > 0:
            # Получаем вектор резюме для TF-IDF
            resume_vector = self.tfidf_vectorizer.transform([search_data['resume_text']])
            
            # Получаем векторы вакансий
            job_vectors = self.tfidf_vectorizer.transform(filtered_df['combined_text'])
            
            # Вычисляем косинусное сходство для TF-IDF
            similarities_tfidf = cosine_similarity(resume_vector, job_vectors)[0]
            
            # Получаем предсказания категории LSTM для резюме
            resume_sequence = self.tokenizer.texts_to_sequences([search_data['resume_text']])
            resume_padded = pad_sequences(resume_sequence, maxlen=200)
            resume_category_prob = self.lstm_model.predict(resume_padded)[0]
                        
            # Получаем предсказания категорий для вакансий
            job_sequences = self.tokenizer.texts_to_sequences(filtered_df['combined_text'])
            job_padded = pad_sequences(job_sequences, maxlen=200)
            job_category_probs = self.lstm_model.predict(job_padded)
            
            # Вычисляем сходство категорий (скалярное произведение векторов вероятностей)
            similarities_lstm = np.dot(job_category_probs, resume_category_prob)
            
            # Комбинируем оценки TF-IDF и LSTM
            combined_similarities = 0.7 * similarities_tfidf + 0.3 * similarities_lstm
            
            # Добавляем оценки в DataFrame
            filtered_df['similarity'] = combined_similarities
            filtered_df = filtered_df.sort_values('similarity', ascending=False)
            
            # # Возвращаем топ-10 вакансий
            return filtered_df.head(10)
        else:
            return pd.DataFrame()
    def display_results(self, filtered_jobs):
        """Отображение результатов поиска"""
        if filtered_jobs.empty:
            messagebox.showinfo("Результаты поиска", "К сожалению, не найдено подходящих вакансий.")
        else:
            # Создаем окно с результатами
            results_window = tk.Toplevel(self.root)
            results_window.title("Результаты поиска вакансий")
            results_window.geometry("1200x600")

            # Создаем Treeview
            columns = ('name_desc', 'company', 'employment', 'salary', 'response_prob', 'transition_value', 'url')
            tree = ttk.Treeview(results_window, columns=columns, show='headings')

            # Настраиваем заголовки
            tree.heading('name_desc', text='Должность')
            tree.heading('company', text='Компания')
            tree.heading('employment', text='Тип занятости')
            tree.heading('salary', text='Зарплата')
            tree.heading('response_prob', text='Вероятность отклика')
            tree.heading('transition_value', text='Полезность перехода')
            tree.heading('url', text='Ссылка')

            # Настраиваем ширину
            tree.column('name_desc', width=200)
            tree.column('company', width=150)
            tree.column('employment', width=120)
            tree.column('salary', width=120)
            tree.column('response_prob', width=120)
            tree.column('transition_value', width=120)
            tree.column('url', width=250)

            # Добавляем данные
            for _, row in filtered_jobs.iterrows():
                job_title = row.get('name_desc', row.get('name', 'Не указано'))
                company = row.get('processed_employer.name', row.get('employer.name', 'Не указано'))
                employment_type = row.get('schedule.name', 'Не указано')
                url = row.get('employer.vacancies_url', row.get('alternate_url', 'Нет ссылки'))

                salary_from = row.get('salary.from', 0)
                salary_to = row.get('salary.to', 0)
                salary_avg = (salary_from + salary_to) / 2 if salary_from and salary_to else salary_from if salary_from else salary_to
                salary_str = f"{int(salary_avg):,} руб.".replace(",", " ") if salary_avg else "Не указана"

                response_prob = row.get('response_prob', 0)
                transition_value = row.get('transition_value', 0)

                response_prob_str = f"{response_prob:.1%}" if isinstance(response_prob, (float, np.float64)) else "N/A"
                transition_value_str = f"{transition_value:,.0f}".replace(",", " ") if isinstance(transition_value, (float, np.float64)) else "N/A"

                tree.insert('', tk.END, values=(
                    job_title,
                    company,
                    employment_type,
                    salary_str,
                    response_prob_str,
                    transition_value_str,
                    url))

            # Настройка прокрутки
            scrollbar = ttk.Scrollbar(results_window, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Пояснение
            explanation = ttk.Label(
                results_window,
                text="Вероятность отклика - оценка шансов получить положительный ответ. " +
                     "Полезность перехода - оценка экономической выгоды от смены работы.",
                wraplength=800
            )
            explanation.pack(pady=5)

            # Обработчик клика
            def copy_url(event):
                item = tree.focus()
                url = tree.item(item, 'values')[6]
                if url != 'Нет ссылки':
                    self.root.clipboard_clear()
                    self.root.clipboard_append(url)
                    messagebox.showinfo("Скопировано", "Ссылка скопирована в буфер обмена")

            tree.bind("<Double-1>", copy_url)

            # Кнопка закрытия (исправленная строка)
            ttk.Button(results_window, text="Закрыть", command=results_window.destroy).pack(pady=10)
        


def main():
    # Загрузка данных о вакансиях (пример)
    try:
        vacancies_df = df_vacancies
    except:
        # Создаем пример данных, если файл не найден
        data = {}
        vacancies_df = pd.DataFrame(vacancies_df)
    
    
    root = tk.Tk()
    app = JobSearchSystem(root, vacancies_df)
    root.mainloop()


if __name__ == "__main__":
    main()
                              
                                               


# In[189]:


def main():
    # Загрузка данных о вакансиях (пример)
    try:
        vacancies_df = vacancies_df
    except:
        # Создаем пример данных, если файл не найден
        data = {
            'name': ['Python Developer', 'Data Analyst', 'Project Manager'],
            'salary.from': [100000, 80000, 120000],
            'salary.to': [150000, 100000, 150000],
            'experience.name': ['От 1 года до 3 лет', 'Нет опыта', 'От 3 до 6 лет'],
            'area.name': ['Москва', 'Санкт-Петербург', 'Москва'],
            'processed_description_desc': ['Описание вакансии 1', 'Описание 2', 'Описание 3'],
            'schedule.name': ['Полный день', 'Удаленная работа', 'Полный день'],
            'employer.name': ['Компания А', 'Компания Б', 'Компания В']
        }
        vacancies_df = pd.DataFrame(data)
    
    root = tk.Tk()
    app = JobSearchSystem(root, vacancies_df)
    root.mainloop()

if __name__ == "__main__":
    main()

