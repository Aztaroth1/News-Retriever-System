import tkinter as tk
from tkinter import messagebox
import webbrowser
from data_loader import load_dataset
from text_classifier import NewsClassifier
import pandas as pd

# === Cargar datos desde CSV ===
csv_path = "Datasets/integrated_news_dataset.csv"
df = load_dataset(csv_path)

# === Inicializar y cargar/entrenar modelo ===
clf = NewsClassifier()

# Entrenar solo si el modelo no fue cargado (ya manejado dentro de la clase)
clf.train(df)

# Actualizar categorías "unknown"
df = clf.update_unknown_categories(df)

# === Guardar dataset actualizado con predicciones ===
df.to_csv("Datasets/noticias_actualizadas.csv", index=False)
print("📝 CSV actualizado guardado como 'noticias_actualizadas.csv'.")

# === Función para crear links clicables en la interfaz ===
def crear_link(frame, texto, url):
    link = tk.Label(frame, text=texto, fg="purple", cursor="hand2", font=('Arial', 10, 'underline'))
    link.bind("<Button-1>", lambda e: webbrowser.open_new(url))
    link.pack(anchor="w")

# === Función principal para búsqueda ===
def buscar():
    query = entry.get()
    if not query:
        messagebox.showwarning("Entrada Vacía", "Por favor ingresa una consulta")
        return

    pred = clf.predict(query)
    resultado = df[df['category'] == pred].head(5)

    # Limpiar resultados anteriores
    for widget in result_frame.winfo_children():
        widget.destroy()

    if resultado.empty:
        tk.Label(result_frame, text="No se encontraron resultados.", fg="red").pack()
        return

    for _, row in resultado.iterrows():
        tk.Label(result_frame, text=f"Título: {row['title']}", font=("Arial", 12, "bold")).pack(anchor="w")
        tk.Label(result_frame, text=f"Categoría: {row['category']}", font=("Arial", 10)).pack(anchor="w")
        crear_link(result_frame, row['link'], row['link'])
        tk.Label(result_frame, text="").pack()  # Espacio entre resultados

# === Configurar ventana principal ===
root = tk.Tk()
root.title("Buscador de Noticias")
root.geometry("800x600")

# Entrada
tk.Label(root, text="Escribe una consulta:", font=("Arial", 12)).pack(pady=5)
entry = tk.Entry(root, width=80, font=("Arial", 12))
entry.pack(pady=5)

# Botón buscar
tk.Button(root, text="Buscar", command=buscar, font=("Arial", 12)).pack(pady=10)

# Área de resultados
result_frame = tk.Frame(root)
result_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Ejecutar aplicación
root.mainloop()
