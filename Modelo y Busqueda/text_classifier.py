import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score, average_precision_score
from sklearn.preprocessing import label_binarize


class NewsClassifier:
    def __init__(self, model_path="Modelos_Entrenados/modelo_entrenado.pkl"):
        self.model_path = model_path
        self.model = None
        self.is_fitted = False

        if os.path.exists(self.model_path):
            self.load_model()

    def train(self, df: pd.DataFrame, force=False):
        if self.is_fitted and not force:
            print("✅ Modelo ya cargado desde disco.")
            return

        # Filtrar solo las entradas con categoría conocida y válida
        known_df = df[df['category'].notna() & (df['category'] != "unknown")]
        X = known_df['text']
        y = known_df['category']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(solver='lbfgs', max_iter=500))
        ])

        self.model.fit(X_train, y_train)
        self.is_fitted = True

        # Guardar modelo
        joblib.dump(self.model, self.model_path)
        print(f"✅ Modelo entrenado y guardado en '{self.model_path}'.")

        # Evaluación rápida
        y_pred = self.model.predict(X_test)
        print("🔍 Evaluación del modelo:")
        print(classification_report(y_test, y_pred))

    def load_model(self):
        self.model = joblib.load(self.model_path)
        self.is_fitted = True
        print(f"📦 Modelo cargado desde '{self.model_path}'.")

    def predict(self, text: str):
        if not self.is_fitted:
            raise Exception("⚠️ Modelo no entrenado ni cargado.")
        return self.model.predict([text])[0]

    def update_unknown_categories(self, df: pd.DataFrame):
        if not self.is_fitted:
            raise Exception("⚠️ Modelo no entrenado ni cargado.")
        
        mask = df['category'] == 'unknown'
        if mask.sum() == 0:
            print("🔎 No hay noticias con categoría 'unknown'.")
            return df

        print(f"📝 Actualizando {mask.sum()} noticias con categoría 'unknown'...")
        df.loc[mask, 'category'] = df.loc[mask, 'text'].apply(lambda x: self.predict(x))
        return df
    
    def evaluate(self, df: pd.DataFrame):
        """
        Retorna métricas de evaluación (Precisión, Recall, F1, MAP, nDCG) para mostrar en la interfaz.
        """
        known_df = df[df['category'].notna() & (df['category'] != "unknown")]
        X = known_df['text']
        y = known_df['category']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        y_pred = self.model.predict(X_test)
        y_score = self.model.predict_proba(X_test)

        labels = sorted(list(set(y_test)))  # etiquetas conocidas
        y_test_bin = label_binarize(y_test, classes=labels)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        map_score = average_precision_score(y_test_bin, y_score, average='macro')
        ndcg = ndcg_score(y_test_bin, y_score)

        return {
            "Precisión": round(precision, 4),
            "Recall": round(recall, 4),
            "F1 Score": round(f1, 4),
            "MAP": round(map_score, 4),
            "nDCG": round(ndcg, 4)
        }
