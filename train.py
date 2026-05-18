import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib
import unicodedata
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def preprocess(text):
    text = text.lower()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df = pd.read_csv("data/emociones.csv")
df['texto'] = df['texto'].apply(preprocess)

y = pd.Series(df['emocion']).astype('category')
labels = list(y.cat.categories)
y_codes = y.cat.codes
id2label = dict(enumerate(labels))

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['texto']).toarray()

joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(id2label, "models/id2label.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_codes, test_size=0.2, random_state=42, stratify=y_codes
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train.values, dtype=torch.long))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class EmotionMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmotionMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc_out = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc_out(x)

input_dim = X.shape[1]
hidden_dim = 128
output_dim = len(labels)

model = EmotionMLP(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40
for epoch in range(epochs):
    total_loss = 0
    for features, labels_batch in train_loader:
        outputs = model(features)
        loss = criterion(outputs, labels_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# --- Guardar modelo ---
torch.save(model.state_dict(), "models/emociones_mlp.pth")
print("✅ Modelo MLP entrenado y guardado en models/emociones_mlp.pth")

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    outputs = model(X_test_tensor)
    preds = torch.argmax(outputs, dim=1).numpy()

cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - MLP")
plt.tight_layout()
plt.savefig("reports/confusion_matrix.png")
plt.show()

print(classification_report(y_test, preds, target_names=labels))
print("✅ Matriz de confusión guardada en reports/confusion_matrix.png")
