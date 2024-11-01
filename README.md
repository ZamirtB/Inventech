Entorno virtual de Python
python -m venv .venv

Activar entorno virtual de Python
.venv\Scripts\Activate

requirements con las dependencias que usas en tu proyecto
pip freeze > requirements.txt

Instalar las dependencias de Python de requirements.txt
pip install -r requirements.txt

Ejecutar pruebas.py
python -m streamlit run src/prueba.py
