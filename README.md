# Pronóstico de Ventas (Streamlit)

App en Streamlit para:
- Cargar históricos (CSV/Excel) y auto-detectar si vienen transpuestos.
- Seleccionar **variable dependiente (Y)** y **variables independientes (X)** desde las columnas encontradas en el archivo.
- Calcular correlaciones con Y.
- Ejecutar regresiones **simples** y **múltiple**.
- Generar **pronóstico ponderado por R²** (combinando pronósticos de las regresiones simples).
- Exportar resultados a **Excel** desde el app.

## Archivos
- `app.py`
- `requirements.txt`

## Datos esperados
No exige nombres fijos: el usuario elige Y y X desde las columnas disponibles.
(Si existen columnas sugeridas como Ventas, PIB, Empleo, TipoCambioPct, Inflación, el app las propone por defecto).

## Deploy
Sube el repo a GitHub y en Streamlit Community Cloud selecciona `app.py` como entrypoint.
