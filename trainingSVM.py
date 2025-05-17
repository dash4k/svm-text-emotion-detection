import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from manualSVM import clean_text, preprocess, vectorize_tfidf, LinearSVM, compute_idf, load_data, build_vocab, OneVSRestSVM, evaluate_model

ID_STOPWORDS = {
    "yang", "untuk", "dengan", "pada", "tidak", "dari", "ini", "di", "ke", "dan",
    "adalah", "itu", "saya", "kamu", "dia", "kita", "mereka", "akan", "apa", "bisa",
    "karena", "jadi", "jika", "agar", "dalam", "ada", "sudah", "belum", "lagi", "harus",
    "sangat", "banyak", "semua", "hanya", "saja", "mau", "boleh", "begitu", "lebih",
    "kurang", "seperti", "masih", "namun", "tetapi", "bukan", "bila", "oleh", "setelah",
    "sebelum", "kami", "aku", "engkau", "dirinya", "sendiri", "antar", "antara",
    "sehingga", "berupa", "terhadap", "pula", "tetap", "baik", "sambil", "tersebut",
    "selama", "seluruh", "bagai", "sekali", "supaya", "dapat", "bahwa", "kapan", "sebab",
    "sedang", "terjadi", "mungkin", "saat", "menjadi", "apakah", "dimana", "kemana"
}

# Load and preprocess the data
texts, labels = load_data()
texts = [clean_text(t) for t in texts]
tokens_list = [preprocess(t) for t in texts]

# Encode labels
unique_labels = sorted(set(labels))
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
y_encoded = np.array([label2idx[label] for label in labels])

# Build vocab and compute IDF
vocab, word2idx = build_vocab(tokens_list)
idf = compute_idf(tokens_list, word2idx)

# Convert texts to TF-IDF vectors
X = np.array([vectorize_tfidf(tokens, word2idx, idf) for tokens in tokens_list])
y = y_encoded

# Shuffle and split (80/20)
data = list(zip(X, y))
random.seed(42)
random.shuffle(data)
split_idx = int(0.8 * len(data))
train_data, test_data = data[:split_idx], data[split_idx:]

X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Train the model
num_classes = len(unique_labels)
num_features = X.shape[1]

# Hyperparameter tuning using 5-fold CV
# best_params = grid_search(X_train, y_train, num_classes, num_features, k=5)

# Final training using best parameters
# model = OneVSRestSVM(num_classes, num_features, 
#                      lr=best_params['lr'], 
#                      C=best_params['C'], 
#                      epochs=best_params['epochs'])

model = OneVSRestSVM(num_classes, num_features, 
                     lr=0.001, 
                     C=0.05, 
                     epochs=100)
model.fit(X_train, y_train)

# Predict on test set
predictions = model.predict(X_test)

# Compute accuracy
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# cm = confusion_matrix_np(y_test, predictions, len(unique_labels))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix (Test Set) Manual SVM")
# plt.tight_layout()
# plt.show()

cm, precision, recall, f1 = evaluate_model(y_test, predictions, len(unique_labels), unique_labels)

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix with Evaluation Metrics")
plt.tight_layout()
plt.show()

import pickle

accuracy = np.mean(predictions == y_test)
precision = np.mean(precision)  # average across classes
recall = np.mean(recall)
f1 = np.mean(f1)

# Bundle everything needed for future prediction
model_package = {
    'model': model,
    'word2idx': word2idx,
    'idf': idf,
    'label2idx': label2idx,
    'idx2label': {v: k for k, v in label2idx.items()},
    'stopwords': ID_STOPWORDS,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1,
}

# Save to file
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model_package, f)

print("Model saved to svm_model.pkl")