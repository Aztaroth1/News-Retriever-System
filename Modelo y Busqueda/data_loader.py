import pandas as pd
import ast

def load_dataset(csv_path: str):
    df = pd.read_csv(csv_path)
    
    # Asegurar que text_features es un diccionario válido
    df['text_features'] = df['text_features'].apply(ast.literal_eval)
    
    # Extraer el texto principal del análisis (puedes usar title + full_content si quieres)
    df['text'] = df['title'].fillna('') + " " + df['full_content'].fillna('')
    
    return df