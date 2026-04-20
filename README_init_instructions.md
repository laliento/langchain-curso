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


###ejecutar chatBot_local.py con streamlit
c:/Users/laptop/git/langchain-curso/venv/Scripts/python.exe -m streamlit run chatBot_local.py

###ejecutar chatBot_local.py con streamlit
c:/Users/laptop/git/langchain-curso/venv/Scripts/python.exe -m streamlit run cv_analyzer/app.py
