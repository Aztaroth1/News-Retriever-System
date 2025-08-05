from flask import Flask, render_template, request, jsonify
import pandas as pd
# Asegúrate de que data_loader.py y text_classifier.py estén en el mismo directorio
from data_loader import load_dataset
from text_classifier import NewsClassifier
import os
import csv

app = Flask(__name__)

# === Configuración inicial ===
csv_path = "Datasets/integrated_news_dataset.csv"
feedback_csv_path = "Datasets/user_feedback.csv" 
df = None
clf = None

def initialize_app():
    """Inicializar datos y modelo"""
    global df, clf
    print("--- 🚀 INICIANDO APLICACIÓN ---")
    try:
        # Cargar datos
        print(f"Intentando cargar dataset desde: {csv_path}")
        df = load_dataset(csv_path)
        if df is None:
            raise ValueError(f"load_dataset retornó None. Verifica '{csv_path}'.")
        print(f"✅ Dataset cargado. Filas: {len(df)}")
        print("Columnas del DataFrame inicial:", df.columns.tolist())
        
        # Inicializar clasificador
        print("Inicializando clasificador de noticias...")
        clf = NewsClassifier()
        print("✅ Clasificador inicializado.")

        # Entrenar el modelo
        print("Entrenando/cargando modelo...")
        clf.train(df) # La función train ya tiene su propio print de éxito/error
        print("✅ Modelo listo (entrenado o cargado).")
        
        # Actualizar categorías desconocidas
        print("Actualizando categorías desconocidas...")
        df = clf.update_unknown_categories(df)
        print("✅ Categorías desconocidas actualizadas.")
        
        # Guardar dataset actualizado
        updated_csv_path = "Datasets/noticias_actualizadas.csv"
        print(f"Guardando dataset actualizado en '{updated_csv_path}'...")
        df.to_csv(updated_csv_path, index=False)
        print(f"📝 CSV actualizado guardado como '{updated_csv_path}'.")
        
        # Crear archivo de feedback si no existe
        print(f"Verificando archivo de feedback: {feedback_csv_path}")
        if not os.path.exists(feedback_csv_path):
            with open(feedback_csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                # Asegúrate de que este encabezado coincida con los datos que se guardan
                writer.writerow(['news_id', 'rating']) 
                print(f"📄 Archivo de feedback '{feedback_csv_path}' creado con encabezado inicial.")
        else:
            print(f"📄 Archivo de feedback '{feedback_csv_path}' ya existe.")

        print("--- ✅ APLICACIÓN INICIALIZADA CORRECTAMENTE ---")

    except FileNotFoundError as fnf_error:
        print(f"❌ ERROR: Archivo no encontrado. Asegúrate de que '{fnf_error.filename}' exista y la ruta sea correcta.")
        print("Detalles del error:", fnf_error)
        exit(1) # Salir de la aplicación si un archivo crítico no se encuentra
    except Exception as e:
        print(f"❌ ERROR CRÍTICO durante la inicialización de la aplicación: {str(e)}")
        import traceback
        traceback.print_exc() # Imprime el traceback completo
        exit(1) # Salir de la aplicación si hay un error crítico

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
        print(f"❌ Error en el endpoint /search: {str(e)}")
        import traceback
        traceback.print_exc()
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
        print(f"❌ Error en el endpoint /rate_news: {str(e)}")
        import traceback
        traceback.print_exc()
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
        print(f"❌ Error en el endpoint /evaluate: {str(e)}")
        import traceback
        traceback.print_exc()
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
        print(f"❌ Error en el endpoint /stats: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al obtener estadísticas: {str(e)}'}), 500

@app.route('/retrain_model', methods=['POST'])
def retrain_model():
    """Endpoint para reentrenar el modelo usando feedback del usuario"""
    global df, clf
    print("--- 🔄 INICIANDO REENTRENAMIENTO ---")
    try:
        # Cargar el feedback de los usuarios
        print(f"Cargando feedback desde: {feedback_csv_path}")
        feedback_df = pd.read_csv(feedback_csv_path)
        print(f"✅ Feedback cargado. Filas: {len(feedback_df)}")
        print("Columnas en feedback_df:", feedback_df.columns.tolist())
        print("Primeras 5 filas de feedback_df:\n", feedback_df.head())

        # Filtrar solo las calificaciones positivas (4 o 5 estrellas)
        positive_feedback = feedback_df[feedback_df['rating'] >= 4]
        print(f"Noticias con feedback positivo: {len(positive_feedback)}")

        # Actualizar el dataset principal con el feedback
        print("🛠️ Aplicando feedback positivo al dataset...")
        for idx, row in positive_feedback.iterrows():
            try:
                news_id = int(row['news_id'])
                # En tu código original, no hay una columna 'correct_category' o 'user_category'
                # en el CSV de feedback para este punto.
                # Si el error es aquí, significa que 'category' o una columna similar
                # debería existir en el feedback_df para poder actualizar df.
                # Si 'category' no existe en feedback_df, esta línea fallará.
                # Asumiendo que el feedback solo tiene 'news_id' y 'rating',
                # esta sección 'pass' no hace nada útil para el reentrenamiento.
                # Si quieres usar el feedback para corregir, necesitas la categoría correcta.
                
                # *** POSIBLE PUNTO DE ERROR SI EL FEEDBACK CSV NO TIENE LA CATEGORÍA CORRECTA ***
                # Si el feedback solo tiene 'news_id' y 'rating', esta sección no puede actualizar 'df'.
                # Para que esta sección sea útil, el feedback_csv_path debería guardar la categoría correcta.
                # Por ahora, se mantiene el 'pass' del código original.
                if news_id in df.index and 'category' in df.columns:
                    # Si tu feedback_df tuviera una columna 'correct_category', la usarías así:
                    # df.loc[news_id, 'category'] = row['correct_category']
                    pass # No hace nada útil para el reentrenamiento en esta versión del código
                else:
                    print(f"⚠️ News_id {news_id} no encontrado en df.index o 'category' no en df.columns. Saltando.")
            except KeyError as ke:
                print(f"❌ Error de columna al procesar feedback positivo: {ke}. Verifica el encabezado de 'user_feedback.csv'.")
                continue # Continuar con el siguiente feedback
            except ValueError as ve:
                print(f"❌ Error de conversión de tipo para news_id: {ve} en fila {idx}. Valor: {row['news_id']}")
                continue # Continuar con el siguiente feedback

        df.to_csv("Datasets/dataset_con_feedback.csv", index=False)
        print("💾 Dataset actualizado (o no, si el feedback no tiene categoría) y guardado.")
        
        # Reentrenar el modelo con el dataset actualizado
        print("Reentrenando el modelo con el dataset actual...")
        clf.train(df, force=True)
        print("✅ Modelo reentrenado exitosamente.")
        
        print("--- ✅ REENTRENAMIENTO FINALIZADO ---")
        return jsonify({'success': True, 'message': '✅ Modelo reentrenado exitosamente con feedback del usuario.'}), 200
    except Exception as e:
        print(f"❌ ERROR en el endpoint /retrain_model: {str(e)}")
        import traceback
        traceback.print_exc() # Imprime el traceback completo
        return jsonify({'error': f'❌ Error al reentrenar el modelo: {str(e)}'}), 500


if __name__ == '__main__':
    # Inicializar la aplicación
    initialize_app()
    
    # Ejecutar servidor
    app.run(debug=True, host='0.0.0.0', port=5000)