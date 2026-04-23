###Python venv create 
laptop@DESKTOP-OFFP9S4 MINGW64 ~/git
$ mkdir langchain-curso

laptop@DESKTOP-OFFP9S4 MINGW64 ~/git
$ cd langchain-curso/

laptop@DESKTOP-OFFP9S4 MINGW64 ~/git/langchain-curso
$ python -m venv venv

laptop@DESKTOP-OFFP9S4 MINGW64 ~/git/langchain-curso
$ source venv/Scripts/activate
(venv)
laptop@DESKTOP-OFFP9S4 MINGW64 ~/git/langchain-curso
$ python --version
Python 3.14.0
(venv)


###pip install todo desde terminal que si esta conectado al env
python -m pip install --upgrade pip
pip install langchain langchain-openai langchain-google-genai

###Gemma 4
https://huggingface.co/collections/google/gemma-4
https://huggingface.co/google/gemma-4-E2B-it
pip install -U transformers torch accelerate
###para correr en local:
pip install pillow transformers torch accelerate langchain-core torchvision librosa streamlit PyPDF2 langchain-community pypdf beautifulsoup4

##BD vectorial local with all-MiniLM-L6-v
pip install chromadb langchain_chroma langchain_huggingface sentence-transformers 
pip install langchain-classic

##Procesamiento de audio/video local
pip install openai-whisper transformers torch accelerate
# Herramientas de sistema para el selector de archivos
# (tkinter suele venir con Python, pero por si acaso)
pip install tk
download ffmpeg-git-full.7z from
    https://www.gyan.dev/ffmpeg/builds/
    add to path bin folder

## para helpdesk_system
pip install -U langgraph-checkpoint-sqlite

##tools
pip install langchain-experimental wikipedia

###Agents
#Gmail agentes
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

#MultiAgente tipo supervisor (hay otros de tipo network o de jerarquía)
pip install langgraph-supervisor

###ejecutar chatBot_local.py con streamlit
c:/Users/laptop/git/langchain-curso/venv/Scripts/python.exe -m streamlit run chatBot_local.py

###ejecutar cv_analyzer con streamlit
c:/Users/laptop/git/langchain-curso/venv/Scripts/python.exe -m streamlit run cv_analyzer/app.py


###ejecutar asistente_legal_RAG con streamlit
c:/Users/laptop/git/langchain-curso/venv/Scripts/python.exe -m streamlit run asistente_legal_RAG/app.py
Ej:
Que personas participan en los contratos de arrendamiento de locales comerciales?



###ejecutar helpdesk_system con streamlit
c:/Users/laptop/git/langchain-curso/venv/Scripts/python.exe -m streamlit run helpdesk_system/app.py
Ej:
Tengo un error 500 en la aplicación
Mi carro está tirando aceite del motor
