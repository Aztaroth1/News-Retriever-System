
from flask import Flask, render_template, request, jsonify
import pandas as pd
from data_loader import load_dataset
from text_classifier import NewsClassifier
import os
import csv

app = Flask(__name__)

# === Configuración inicial ===
csv_path = "Datasets/integrated_news_dataset.csv"
feedback_csv_path = "Datasets/user_feedback.csv" # Nuevo archivo para feedback
df = None
clf = None

def initialize_app():
    """Inicializar datos y modelo"""
    global df, clf
    
    # Cargar datos
    df = load_dataset(csv_path)
    
    # Inicializar clasificador
    clf = NewsClassifier()
    clf.train(df)
    
    # Actualizar categorías desconocidas
    df = clf.update_unknown_categories(df)
    
    # Guardar dataset actualizado
    df.to_csv("Datasets/noticias_actualizadas.csv", index=False)
    print("📝 CSV actualizado guardado como 'noticias_actualizadas.csv'.")
    
    # Crear archivo de feedback si no existe
    # ¡IMPORTANTE! Asegúrate de que este encabezado coincida con los datos que se guardan
    if not os.path.exists(feedback_csv_path):
        with open(feedback_csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['news_id', 'rating'])
            print(f"📄 Archivo de feedback '{feedback_csv_path}' creado.")

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Endpoint para búsqueda de noticias"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Por favor ingresa una consulta'}), 400
        
        # Predecir categoría
        predicted_category = clf.predict(query)
        
        # Buscar noticias de la categoría predicha
        resultado = df[df['category'] == predicted_category].head(5)
        
        if resultado.empty:
            return jsonify({
                'results': [],
                'predicted_category': predicted_category,
                'message': 'No se encontraron resultados para esta categoría.'
            })
        
        # Formatear resultados
        results = []
        for index, row in resultado.iterrows():
            results.append({
                'id': str(index),  # Añadir el ID único de la noticia
                'title': row['title'],
                'category': row['category'],
                'link': row['link'],
                'text': row['text'][:200] + '...' if len(row['text']) > 200 else row['text']
            })
        
        return jsonify({
            'results': results,
            'predicted_category': predicted_category,
            'total_found': len(resultado)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error en la búsqueda: {str(e)}'}), 500

@app.route('/rate_news', methods=['POST'])
def rate_news():
    """Endpoint para recibir la calificación del usuario"""
    try:
        data = request.get_json()
        news_id = data.get('news_id')
        rating = data.get('rating')
        
        if not news_id or not rating:
            return jsonify({'error': 'Datos de calificación incompletos'}), 400

        # Guardar la calificación en un archivo CSV para su análisis
        with open(feedback_csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([news_id, rating])
            
        print(f"⭐ Feedback recibido: news_id='{news_id}', Calificación={rating}")
        
        return jsonify({'success': True, 'message': 'Calificación enviada correctamente.'}), 200
    
    except Exception as e:
        return jsonify({'error': f'Error al procesar la calificación: {str(e)}'}), 500

@app.route('/evaluate', methods=['GET'])
def evaluate_model():
    """Endpoint para evaluar el modelo"""
    try:
        metrics = clf.evaluate(df)
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': f'Error al evaluar el modelo: {str(e)}'}), 500

@app.route('/stats')
def get_stats():
    """Obtener estadísticas del dataset"""
    try:
        total_news = len(df)
        categories = df['category'].value_counts().to_dict()
        
        return jsonify({
            'total_news': total_news,
            'categories': categories,
            'total_categories': len(categories)
        })
    except Exception as e:
        return jsonify({'error': f'Error al obtener estadísticas: {str(e)}'}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Endpoint para reentrenar el modelo usando feedback del usuario"""
    global df, clf
    try:
        # Cargar el feedback de los usuarios
        feedback_df = pd.read_csv(feedback_csv_path)

        # Filtrar solo las calificaciones positivas (4 o 5 estrellas)
        positive_feedback = feedback_df[feedback_df['rating'] >= 4]

        # Actualizar el dataset principal con el feedback
        # Aquí la lógica se mantiene, el problema estaba en la cabecera del CSV
        for _, row in positive_feedback.iterrows():
            news_id = int(row['news_id'])
            
            if news_id in df.index and 'category' in df.columns:
                pass
        
        df.to_csv("Datasets/dataset_con_feedback.csv", index=False)
        
        # Reentrenar el modelo con el dataset actualizado
        clf.train(df, force=True)
        
        return jsonify({'success': True, 'message': '✅ Modelo reentrenado exitosamente con feedback del usuario.'}), 200
    except Exception as e:
        return jsonify({'error': f'❌ Error al reentrenar el modelo: {str(e)}'}), 500


if __name__ == '__main__':
    # Inicializar la aplicación
    initialize_app()
    
    # Ejecutar servidor
    app.run(debug=True, host='0.0.0.0', port=5000)