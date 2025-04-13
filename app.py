from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard/dashboard.html', active_page='dashboard')

@app.route('/login')
def login():
    return render_template('auth/login.html')

@app.route('/register')
def register():
    return render_template('auth/register.html')

@app.route('/training/import')
def import_training():
    return render_template('training/import.html', active_page='import_training')

@app.route('/training/preprocessing')
def preprocessing_training():
    return render_template('training/preprocessing.html', active_page='preprocessing_training')

@app.route('/testing/import')
def import_testing():
    return render_template('testing/import.html', active_page='import_testing')

@app.route('/testing/preprocessing')
def preprocessing_testing():
    return render_template('testing/preprocessing.html', active_page='preprocessing_testing')

@app.route('/testing/classification')
def classification_testing():
    return render_template('testing/classification.html', active_page='classification_testing')

@app.route('/sentence')
def sentence():
    return render_template('sentence/sentence.html', active_page='sentence')


if __name__ == '__main__':
    app.run(debug=True)