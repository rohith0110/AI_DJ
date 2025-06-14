This project needs Python 3.9
So, install python 3.9 64bit from python website.

Then in your project directory, run this command in your terminal:

```
py -3.9 -m venv venv
```

This will create a new virtual environment named `venv` in your project directory with python 3.9 as the default interpreter.

Then , activate the virtual environment by running:

```
venv\Scripts\activate
```

Then install the requirements using this code:

```
pip install -r requirements.txt
```

Then the project uses your youtube cookies to download videos from youtube.
Extract your youtube cookies as a txt file using cookies.txt extension and paste it in this folder

Then run

```
python app.py
```
