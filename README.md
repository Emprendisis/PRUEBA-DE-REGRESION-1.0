# Pronóstico de Ventas (Streamlit)

App en Streamlit para:
- Cargar históricos (CSV/Excel) y auto-detectar si vienen transpuestos.
- Calcular correlaciones con **Ventas**.
- Ejecutar regresiones **simples** (Ventas vs PIB, Empleo, TipoCambioPct, Inflación).
- Ejecutar regresión **múltiple** (todas las X a la vez).
- Generar **pronóstico ponderado por R²** (combinando pronósticos de las regresiones simples).
- Exportar resultados a **Excel** desde el app.

## Archivos
- `app.py`
- `requirements.txt`

## Datos esperados
Columnas exactas esperadas:
- `Ventas`
- `PIB`
- `Empleo`
- `TipoCambioPct`
- `Inflación`

Pronósticos (X): puedes capturarlos manualmente o subir un archivo con esas columnas.

## Deploy
Sube el repo a GitHub y en Streamlit Community Cloud selecciona `app.py` como entrypoint.
