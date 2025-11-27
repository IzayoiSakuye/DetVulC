import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import re


#  1. 配置参数
VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 512
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 10
DATASET_PATH = '../data/processed/result.csv'
MODEL_PATH = '../models/vulnerability_scanner_model.keras'
MLB_PATH = '../models/mlb.pkl'

#  2. 数据加载和预处理
print("Loading and preprocessing data...")
try:
    df = pd.read_csv(DATASET_PATH)
except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_PATH}'"); exit()

df['ir_code'].fillna('', inplace=True)
df['specific_vulnerability_type'].fillna('unknown', inplace=True)
df = df[df['ir_code'].str.len() > 0]

features = df['ir_code']
labels = df['specific_vulnerability_type'].apply(lambda x: x.split('|'))


#  3. 标签编码 (多标签)
print("Encoding labels...")
mlb = MultiLabelBinarizer()
encoded_labels = mlb.fit_transform(labels)
NUM_CLASSES = len(mlb.classes_)
print(f"Found {NUM_CLASSES} unique classes: {mlb.classes_[:10]}")


#  4. 创建文本向量化层
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)
vectorizer.adapt(features.values)


#  5. 数据集划分
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    features.values, encoded_labels, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print(f"Test set size: {len(X_test)}")


#  6. 构建集成预处理的深度学习模型
print("Building the model with integrated vectorization layer...")
model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)
model.summary()


#  7. 模型训练
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val)
)


#  8. 模型评估
print("Evaluating model on the test set...")
results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
print(f"Test Loss: {results[0]}, Accuracy: {results[1]}, Precision: {results[2]}, Recall: {results[3]}")


#  9. 保存模型和标签编码器
print("Saving model and label binarizer...")
model.save(MODEL_PATH)
joblib.dump(mlb, MLB_PATH)

print(f"Process finished successfully. Model saved to {MODEL_PATH}")
