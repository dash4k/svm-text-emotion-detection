import numpy as np
import random

def load_data():
    emotions = [ 'Anger', 'Fear', 'Joy', 'Love', 'Neutral', 'Sad']
    texts, labels = [], []

    for emotion in emotions:
        with open(f'dataset/{emotion}Data.csv', encoding='utf-8') as file:
            lines = file.readlines()[1:]
            for line in lines:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if parts:
                        texts.append(parts[0])
                        labels.append(emotion)
    return texts, labels


def clean_text(text: str):
    text = text.lower()
    text = ''.join([word if word.isalnum() or word.isspace() else '' for word in text])
    text = text.strip()
    return text


def preprocess(text: str):
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
    tokens = text.split()
    tokens = [word for word in tokens if word not in ID_STOPWORDS]
    return tokens


def build_vocab(token_lists: list):
    vocab = sorted(set(word for tokens in token_lists for word in tokens))
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return vocab, word2idx


def compute_idf(token_lists, word2idx):
    df_counts = np.zeros(len(word2idx))
    N = len(token_lists)
    for token in token_lists:
        for word in set(token):
            if word in word2idx:
                df_counts[word2idx[word]] += 1
    idf = np.log((N + 1) / (df_counts + 1)) + 1
    return idf


def vectorize_tfidf(tokens, word2idx, idf):
    vec = np.zeros(len(word2idx))
    token_counts = {}

    for word in tokens:
        if word in word2idx:
            token_counts[word] = token_counts.get(word, 0) + 1

    total_terms = sum(token_counts.values())

    for word, count in token_counts.items():
        idx = word2idx[word]
        tf = count / total_terms
        vec[idx] = tf * idf[idx]

    return vec


def confusion_matrix_np(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


class LinearSVM:
    def __init__(self, num_features, lr=0.1, C=1, epochs=10):
        self.w = np.zeros(num_features)
        self.b = 0
        self.lr = lr
        self.C = C
        self.epochs =  epochs

    def fit(self, X, y):
        for a in range(self.epochs):
            print(f"Iter: {a} from {self.epochs}")
            for i in range(len(X)):
                xi, yi = X[i], y[i]
                if yi * (np.dot(self.w, xi) + self.b) >= 1:
                    self.w -= self.lr * self.w
                else:
                    self.w -= self.lr * (self.w - self.C * yi * xi)
                    self.b += self.lr * self.C * yi

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)
    

class OneVSRestSVM:
    def __init__(self, num_classes, num_features, lr=0.1, C=1, epochs=10):
        self.models = [LinearSVM(num_features, lr, C, epochs) for _ in range(num_classes)]

    def fit(self, X, y):
        for i, model in enumerate(self.models):
            binary_y = np.where(y == i, 1, -1)
            model.fit(X, binary_y)

    def predict(self, X):
        scores = np.array([np.dot(X, model.w) + model.b for model in self.models])
        return np.argmax(scores, axis=0)
    

def k_fold_split(X, y, k=5, seed=42):
    data = list(zip(X, y))
    random.Random(seed).shuffle(data)
    fold_size = len(data) // k
    folds = []

    for i in range(k):
        test_data = data[i*fold_size:(i+1)*fold_size]
        train_data = data[:i*fold_size] + data[(i+1)*fold_size:]
        X_train, y_train = zip(*train_data)
        X_test, y_test = zip(*test_data)
        folds.append((np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)))

    return folds


def cross_validate(X, y, num_classes, num_features, k=5, lr=0.01, C=1.0, epochs=50):
    folds = k_fold_split(X, y, k)
    acc_scores = []

    for fold_idx, (X_train, y_train, X_val, y_val) in enumerate(folds):
        print(f"Fold {fold_idx+1}/{k}")
        model = OneVSRestSVM(num_classes, num_features, lr=lr, C=C, epochs=epochs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        acc = np.mean(preds == y_val)
        acc_scores.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%")

    return np.mean(acc_scores)


def grid_search(X, y, num_classes, num_features, k=5):
    lr_values = [0.01, 0.05, 0.1]
    C_values = [0.1, 0.5, 1]
    epoch_values = [50]

    best_score = 0
    best_params = {}

    for lr in lr_values:
        for C in C_values:
            for epochs in epoch_values:
                print(f"\nTrying params: lr={lr}, C={C}, epochs={epochs}")
                score = cross_validate(X, y, num_classes, num_features, k, lr, C, epochs)
                print(f"Average Accuracy: {score * 100:.2f}%")

                if score > best_score:
                    best_score = score
                    best_params = {'lr': lr, 'C': C, 'epochs': epochs}

    print(f"\nBest Params: {best_params}, Best Accuracy: {best_score * 100:.2f}%")
    return best_params


def evaluate_model(y_true, y_pred, num_classes, label_names):
    cm = confusion_matrix_np(y_true, y_pred, num_classes)

    precision = []
    recall = []
    f1 = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

        print(f"{label_names[i]} -> Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1_score:.2f}")

    overall_acc = np.mean(np.array(y_pred) == np.array(y_true))
    print(f"\nOverall Accuracy: {overall_acc * 100:.2f}%")
    
    return cm, precision, recall, f1