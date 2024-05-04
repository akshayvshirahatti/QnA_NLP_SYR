# Question Answering System Using Word2Vec and GloVe

This Flask application serves as a question-answering system that allows users to upload PDF documents and ask questions about the content of those documents. The application supports two types of models for answering questions: Word2Vec and GloVe.

## Installation

### Please make sure you are using Python 3.6 for this project

#### Follow the following steps to run the project

1. Clone the repository from github

```bash
git clone https://github.com/akshayvshirahatti/QuestionAnsweringSystemUsingWord2VecAndGloVe.git
```

2. Create Project on Firebase and change rule from 'read' to 'write' on Firebase Storage.
3. Add Firebase configuration in Flask/app.py file
4. Change the directory to Flask by doing cd Flask
5. Run the requirements.txt file to install all the dependencies for the project

```bash
pip install -r requirements.txt
```

6. Install cdQA using the following commands

```bash
git clone https://github.com/cdqa-suite/cdQA.git
```

```bash
cd cdQA
```

```bash
pip install -e .
```

7. Run app.py file

```bash
python app.py
```

8. If there is an error while running app.py do the following,

```bash
git clone -b legacy_py3.6 https://github.com/QUVA-Lab/e2cnn.git
```

```bash
cd e2cnn
```

```bash
python setup.py install
```

9. Open your browser write your IP Address and add :8080 (example- http://192.168.114.84:8080/).

Paste your firebase configurations in the following code in app.py

```bash
config = {
    "apiKey": "AIzaSyD-SAMPLEKEYxxxxxxxxx-yyyyyyyyyy",
    "authDomain": "your-project-id.firebaseapp.com",
    "projectId": "your-project-id",
    "databaseURL": "https://your-project-id.firebaseio.com",
    "storageBucket": "your-project-id.appspot.com",
    "messagingSenderId": "123456789012",
    "appId": "1:123456789012:web:33445566778899abcdef0",
    "measurementId": "G-XYZ123ABC456"
}
```

If you are facing any problem running project on localhost then, In app.py replace app.run(host="0.0.0.0", port=8080) with app.run(debug=True)
