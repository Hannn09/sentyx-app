from flask import Flask, render_template, request, redirect, url_for, flash
from models import db, TestingModel, TrainingModel, CrawlingTweet, PreprocessingTraining, ClassificationTraining, PreprocessingTesting, ClassificationTesting, UserModel
from preprocessing import casefolding, cleansing, tokenization, normalize, stopword_removal, stemming
from utils import get_pagination_range, generate_confusion_matrix
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text, extract
from sqlalchemy.sql import func
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix,  accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict, Counter
from wordcloud import WordCloud

import os
import subprocess
import emoji
import pandas as pd
import mysql.connector
import pickle
import matplotlib.pyplot as plt


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
DATASET_FOLDER = 'static/dataset'

app.config['DEBUG'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:@localhost/db_sentyx'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = '16177445beb012017229d7a17ae34440d5fc5299a82ab2e62ec79b8e9d04ec7a'
db.init_app(app)

# Database connection
db_user = 'root'
db_password = ''
db_host = 'localhost'
db_name = 'db_sentyx'


# Make Connection to MySQL
mydb = mysql.connector.connect(host= db_host, user= db_user, password= db_password, database=db_name)
mycursor = mydb.cursor()

engine = create_engine('mysql+mysqlconnector://', creator=lambda: mydb)
Session = sessionmaker(bind=engine)
session = Session()

# Initialize auth
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return UserModel.query.get(int(user_id))

# Routes
@app.route('/')
@login_required
def dashboard():
    # Count data Testing and Training
    trainingCount = TrainingModel.query.count()
    testingCount = TestingModel.query.count()

    # Count data per label in training
    positif_train = TrainingModel.query.filter_by(label='positif').count()
    negatif_train = TrainingModel.query.filter_by(label='negatif').count()
    netral_train = TrainingModel.query.filter_by(label='netral').count()

    # Count data per label in testing
    positif_test = ClassificationTesting.query.filter_by(result_classification='positif').count()
    negatif_test = ClassificationTesting.query.filter_by(result_classification='negatif').count()
    netral_test = ClassificationTesting.query.filter_by(result_classification='netral').count()

    # Count total data per label
    positif_total = positif_train + positif_test
    negatif_total = negatif_train + negatif_test
    netral_total = netral_train + netral_test

    # Count all data
    all_data = positif_total + negatif_total + netral_total

    # Count data positif and mixed
    positifCount = positif_train + positif_test
    mixedCount = negatif_train + netral_train + negatif_test + netral_test

    # Calculate percentage
    if all_data == 0:
        positif_percent = negatif_percent = netral_percent = 0
    else:
        positif_percent = round((positif_total / all_data) * 100, 1)
        negatif_percent = round((negatif_total / all_data) * 100, 1)
        netral_percent = round((netral_total / all_data) * 100, 1)
    
    # Get data training label and timestamp
    training_data =  db.session.query(TrainingModel.label, TrainingModel.timestamp).all()

    # Manual extract year to string
    sentiment_by_year = defaultdict(lambda: {'positif': 0, 'negatif': 0, 'netral': 0})

    # Get year timestamp in training data
    for label, timestamp in training_data:
       if timestamp and len(timestamp) >= 4:
            try:
                # Get last 4 digits in timestamp
                year = timestamp[-4:]
                if label in sentiment_by_year[year]:
                    sentiment_by_year[year][label] += 1
            except:
                continue
    
    years = sorted(sentiment_by_year.keys())
    positif_counts_year = [sentiment_by_year[year]['positif'] for year in years]
    negatif_counts_year = [sentiment_by_year[year]['negatif'] for year in years]
    netral_counts_year = [sentiment_by_year[year]['netral'] for year in years]

    # Get preprocessing training after stemming
    preprocessing_training = db.session.query(PreprocessingTraining.result).all()

    # group all result
    text_corpus = ' '.join([item[0] for item in preprocessing_training if item[0]])

    # word cloud
    wc = WordCloud(stopwords=None, background_color='white')
    wc.generate(text_corpus)

    # get frequency of words
    frequency = wc.words_

    # Get top 10 words
    top_words = Counter(frequency).most_common(3)

    return render_template('dashboard/dashboard.html', active_page='dashboard', trainingCount=trainingCount, testingCount=testingCount, positifCount=positifCount, mixedCount=mixedCount, positifPercent=positif_percent, negatifPercent=negatif_percent, netralPercent=netral_percent, positifCountsYear=positif_counts_year, negatifCountsYear=negatif_counts_year, netralCountsYear=netral_counts_year, years=years, topWords=top_words)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = UserModel.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for('login'))

    return render_template('auth/login.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        user = UserModel(name=name, email=email, password=password)
        session.add(user)
        session.commit()

        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('auth/register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logout successful!", "success")
    return redirect(url_for('login'))

@app.route('/training/import')
@login_required
def import_training():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = TrainingModel.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)
    
    return render_template('training/import.html', active_page='import_training', data=pagination.items, pagination=pagination, page_range=page_range)

# Route for import csv training
@app.route('/import-data-training', methods=['POST'])
@login_required
def upload_files_training():
    uploaded_file = request.files['file_tweet']

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        df = pd.read_csv(file_path)
        
        if 'tweet' in df.columns and 'label' in df.columns and 'created_at' in df.columns:  
            for _, row in df.iterrows():
                df.fillna('', inplace=True)
                tweet = row['tweet']
                label = row['label']
                created_at = row['created_at']
                slug = emoji.replace_emoji(tweet.strip().lower().replace(' ', '-'), replace='')

                # Find similar slug
                existing_slug = session.query(TrainingModel).filter_by(slug=slug).first()

                if existing_slug:
                    # Replace Slug
                    existing_slug.slug = tweet
                    existing_slug.label = label
                else:
                    # Insert New Data
                    new_data = TrainingModel(tweet=tweet, slug=slug, label=label, timestamp=created_at)
                    session.add(new_data)

            session.commit()
            flash("Upload Data Success!", "success")
            return redirect(url_for('import_training'))
        else:
            flash("The CSV File must contain 'tweet' and 'label' column.", "error")
            return redirect(url_for('import_training'))
    else:
        flash("No file selected for uploading.", "error")
        return redirect(url_for('import_testing'))

# Route for process preprocessing training
@app.route('/preprocessing-training')
@login_required
def proccess_preprocessing_training():
    data = TrainingModel.query.all()

    if not data:
        flash("No data available for preprocessing.", "error")
        return redirect(url_for('preprocessing_training'))

    for item in data:
        casefolding_text = casefolding(item.tweet)
        cleansing_text = cleansing(casefolding_text)
        tokenization_text = tokenization(cleansing_text)
        normalization_text = normalize(tokenization_text)
        stopword_removal_text = stopword_removal(normalization_text)
        stemming_text = stemming(stopword_removal_text)
        result_text = ' '.join(stemming_text)
        slug = item.tweet.strip().lower().replace(' ', '-')

        # find similar slug
        existing_slug = session.query(PreprocessingTraining).filter_by(slug=slug).first()

        if existing_slug:
            existing_slug.slug = slug
            existing_slug.result = result_text
            existing_slug.tweet = item.tweet
            existing_slug.label = item.label
        else:
            processed  = PreprocessingTraining(tweet=item.tweet, result=result_text, slug=slug, label=item.label)
            session.add(processed)

    session.commit()
    flash("Preprocessing Data Success!", "success")
    return redirect(url_for('preprocessing_training'))

@app.route('/training/preprocessing')
@login_required
def preprocessing_training():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = PreprocessingTraining.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)
    return render_template('training/preprocessing.html', active_page='preprocessing_training', data=pagination.items, pagination=pagination, page_range=page_range)

# Process classification training
@app.route('/classification-training')
@login_required
def process_classification_training():
    data = PreprocessingTraining.query.all()

    if not data:
        flash("No data available for classification.", "error")
        return redirect(url_for('classification_training'))
    
    # Load data from PreprocessingTraining
    result_preprocessing = [item.result for item in data]
    label = [item.label for item in data]
    tweet  = [item.tweet for item in data]
    slug = [item.slug for item in data]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(result_preprocessing)

    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(label)

    # Split data 80:20
    x_train, x_test, y_train, y_test, tweet_train, tweet_test, slug_train, slug_test = train_test_split(x, y, tweet, slug, test_size=0.2, random_state=42)  

    # SVM Classifier
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(x_train, y_train)

    # Save the model and vectorizer
    pickle.dump(classifier, open('static/model/svm_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('static/model/tfidf_vectorizer.pkl', 'wb'))
    pickle.dump(label_encoder, open('static/model/label_encoder.pkl', 'wb'))

    # Predict 
    prediction = classifier.predict(x_test)
    # Save accuracy to file
    accuracy = accuracy_score(y_test, prediction)
    with open("static/model/svm_accuracy.txt", "w") as f:
        f.write(f"{accuracy}")
    
    print(classification_report(y_test, prediction, target_names=['positif', 'negatif', 'netral']))
    print("Classification Accuracy: ", classifier.score(x_test, y_test))

    for t, prediction_label, s in zip(tweet_test, prediction, slug_test):
        label_name = str(label_encoder.inverse_transform([prediction_label])[0])
        # Find similar slug
        existing_slug = session.query(ClassificationTraining).filter_by(slug=s).first()

        if existing_slug:
            existing_slug.slug = s
            existing_slug.tweet = t
            existing_slug.label = label_name    
            existing_slug.result_classification = label_name    
        else:
            processed = ClassificationTraining(tweet=t, label=label_name, slug=s, result_classification=label_name)
            session.add(processed)

    session.commit()
    flash("Classification Data Success!", "success")
    return redirect(url_for('classification_training'))

@app.route('/training/classification')
@login_required
def classification_training():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = ClassificationTraining.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)
    return render_template('training/classification.html', active_page='classification_training', data=pagination.items, pagination=pagination, page_range=page_range)

@app.route('/testing/import')
@login_required
def import_testing():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = TestingModel.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)

    return render_template('testing/import.html', active_page='import_testing', data=pagination.items, pagination=pagination, page_range=page_range)

# Route for import csv testing
@app.route('/testing/import', methods=['POST'])
@login_required
def upload_files():
    uploaded_file = request.files['file_tweet']

    if uploaded_file.filename != '':
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)
        
        df = pd.read_csv(file_path)
        
        if 'tweet' in df.columns:  
            for _, row in df.iterrows():
                df.fillna('', inplace=True)
                tweet = row['tweet']
                slug = emoji.replace_emoji(tweet.strip().lower().replace(' ', '-'), replace='')

                # Find similar slug
                existing_slug = session.query(TestingModel).filter_by(slug=slug).first()

                if existing_slug:
                    # Replace Slug
                    existing_slug = session.query(TestingModel).filter_by(slug=slug).first()
                    existing_slug.slug = tweet
                else: 
                    # Insert New Data
                    new_data = TestingModel(tweet=tweet, slug=slug)
                    session.add(new_data)

            session.commit()
            flash("Upload Data Success!", "success")
            return redirect(url_for('import_testing'))
        else:
            flash("The CSV File must contain 'tweet' column.", "error")
            return redirect(url_for('import_testing'))
    else:
        flash("No file selected for uploading.", "error")
        return redirect(url_for('import_testing'))

        # 

# Route for process preprocessing testing
@app.route('/preprocessing-testing')
@login_required
def process_preprocessing_testing():
    data = TestingModel.query.all()

    if not data:
        flash("No data available for preprocessing.", "error")
        return redirect(url_for('preprocessing_training'))

    for item in data:
        casefolding_text = casefolding(item.tweet)
        cleansing_text = cleansing(casefolding_text)
        tokenization_text = tokenization(cleansing_text)
        normalization_text = normalize(tokenization_text)
        stopword_removal_text = stopword_removal(normalization_text)
        stemming_text = stemming(stopword_removal_text)
        result_text = ' '.join(stemming_text)
        slug = item.tweet.strip().lower().replace(' ', '-')

        # find similar slug
        existing_slug = session.query(PreprocessingTesting).filter_by(slug=slug).first()

        if existing_slug:
            existing_slug.slug = slug
            existing_slug.result = result_text
            existing_slug.tweet = item.tweet
        else:
            processed  = PreprocessingTesting(tweet=item.tweet, result=result_text, slug=slug)
            session.add(processed)

    session.commit()
    flash("Preprocessing Data Success!", "success")
    return redirect(url_for('preprocessing_testing'))

@app.route('/testing/preprocessing')
@login_required
def preprocessing_testing(): 
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = PreprocessingTesting.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)
    return render_template('testing/preprocessing.html', active_page='preprocessing_testing', data=pagination.items, pagination=pagination, page_range=page_range)

# Process classification testing
@app.route('/classification-testing')
@login_required
def process_classification_testing():
    data = PreprocessingTesting.query.all()

    if not data:
        flash("No data available for classification.", "error")
        return redirect(url_for('classification_testing'))
    
    # Load model 
    model = pickle.load(open('static/model/svm_model.pkl', 'rb'))
    vectorizer = pickle.load(open('static/model/tfidf_vectorizer.pkl', 'rb'))
    label_encoder = pickle.load(open('static/model/label_encoder.pkl', 'rb'))

    # Load data from PreprocessingTesting
    result_preprocessing = [item.result for item in data]
    slug = [item.slug for item in data]
    tweet  = [item.tweet for item in data]

    # TF-IDF Vectorization
    x_test = vectorizer.transform(result_preprocessing)

    # Predict
    prediction = model.predict(x_test)

    for t, prediction_label, s in zip(tweet, prediction, slug):
        label_name = str(label_encoder.inverse_transform([prediction_label])[0])

        # Find similar slug
        existing_slug = session.query(ClassificationTesting).filter_by(slug=s).first()

        if existing_slug:
            existing_slug.slug = s
            existing_slug.tweet = t
            existing_slug.result_classification = label_name    
        else:
            processed = ClassificationTesting(tweet=t, slug=s, result_classification=label_name)
            session.add(processed)

    session.commit()
    flash("Classification Data Success!", "success")
    return redirect(url_for('classification_testing'))

@app.route('/testing/classification')
@login_required
def classification_testing():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = ClassificationTesting.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)
    return render_template('testing/classification.html', active_page='classification_testing', data=pagination.items, pagination=pagination, page_range=page_range)

@app.route('/sentence', methods=['GET', 'POST'])
@login_required
def sentence():

    if request.method == 'POST':
        text = request.form.get('sentence')

        if text is None:
            flash("Sentence can't be null", "error")
            return redirect(url_for('sentence'))
        
        # Preprocessing
        casefolding_text = casefolding(text)
        cleansing_text = cleansing(casefolding_text)
        tokenization_text = tokenization(cleansing_text)
        normalization_text = normalize(tokenization_text)
        stopword_removal_text = stopword_removal(normalization_text)
        stemming_text = stemming(stopword_removal_text)

        result_text = ' '.join(stemming_text)

        # Load model
        model = pickle.load(open('static/model/svm_model.pkl', 'rb'))
        vectorizer = pickle.load(open('static/model/tfidf_vectorizer.pkl', 'rb'))
        label_encoder = pickle.load(open('static/model/label_encoder.pkl', 'rb'))

        # Predict
        final_result = model.predict(vectorizer.transform([result_text]))
        label_name = str(label_encoder.inverse_transform([final_result[0]])[0])

        # Ambil index label prediksi
        label_index = final_result[0]

        # Ambil bobot koefisien dari model linear
        weights = model.coef_.toarray()[label_index]
        feature_names = vectorizer.get_feature_names_out()

        # Vectorize kalimat input
        x_input = vectorizer.transform([result_text])
        non_zero_indices = x_input.nonzero()[1]

        # Cari kata-kata penting berdasarkan bobot
        keyword_weights = [(feature_names[i], weights[i]) for i in non_zero_indices]
        keyword_weights.sort(key=lambda x: abs(x[1]), reverse=True)
        top_keywords = [kw[0] for kw in keyword_weights[:3]]

        # Load accuracy
        # with open("static/model/svm_accuracy.txt", "r") as f:
        #     accuracy = f.read()
        # accuracy = float(accuracy) * 100


        return render_template('sentence/sentence.html', active_page='sentence', text=text, result=label_name, keywords=top_keywords, preprocessing=result_text)

    return render_template('sentence/sentence.html', active_page='sentence')

@app.route('/crawling')
@login_required
def crawl():
    page = request.args.get('page', 1, type=int)
    per_page = 10
    pagination = CrawlingTweet.query.paginate(page=page, per_page=per_page)
    page_range = get_pagination_range(pagination.page, pagination.pages)

    return render_template('crawl/crawl.html', active_page='crawl', data=pagination.items, pagination=pagination, page_range=page_range)

@app.route('/crawl-tweet', methods=['POST'])
@login_required
def crawl_tweet():
    keywords = request.form['keyword']
    twitter_auth_token = '062107cbeb3429f7db34bd3aed99d78dc8cd9d31'

    search_keyword = f'{keywords} -filter:links -filter:replies -gestun -giveaway -wts -galbay lang:in since:2021-01-01 until:2025-04-01'
    
    if ' ' in keywords:
        keywords = keywords.replace(' ', '_')   
    
    filename = f'{keywords}.csv'

    # Run tweet-harvest 
    limit = 50
    command = f'npx --yes tweet-harvest@latest -o "{filename}" -s "{search_keyword}" -l {limit} --token {twitter_auth_token}'
    subprocess.run(command, shell=True, capture_output=True, text=True)
    file_path = f'tweets-data/{filename}'
   
    # read csv 
    df = pd.read_csv(file_path)
    
    if 'full_text' in df.columns:
        df['full_text'] = df['full_text'].fillna('').astype(str)
        for _, row in df.iterrows():
            tweet = row['full_text']

            # Preprocessing
            casefolding_text = casefolding(tweet)
            cleansing_text = cleansing(casefolding_text)
            tokenization_text = tokenization(cleansing_text)
            normalization_text = normalize(tokenization_text)
            stopword_removal_text = stopword_removal(normalization_text)
            stemming_text = stemming(stopword_removal_text)
            result_text = ' '.join(stemming_text)

            # Load model
            model = pickle.load(open('static/model/svm_model.pkl', 'rb'))
            vectorizer = pickle.load(open('static/model/tfidf_vectorizer.pkl', 'rb'))
            label_encoder = pickle.load(open('static/model/label_encoder.pkl', 'rb'))

            # TF-IDF Vectorization & Predict
            prediction = model.predict(vectorizer.transform([result_text]))
            label_name = str(label_encoder.inverse_transform([prediction[0]])[0])

            
            data = CrawlingTweet(tweet=tweet, label=label_name, preprocessing_result=result_text)
            session.add(data)

        session.commit()
        flash("Success Crawling Data!", "success")
        return redirect(url_for('crawl'))
    else:
        flash("The CSV File must contain 'full_text' column.", "error")
        return redirect(url_for('crawl'))

@app.route('/evaluation')
@login_required
def evaluation():
    return render_template('evaluation/evaluation.html', active_page='evaluation')

@app.route('/evaluation-process', methods=['POST'])
@login_required
def evaluation_process():
    ratio = request.form['ratio']
    # Split and covert to int ratio 
    train_ratio = int (ratio.split(':')[0]) / 100
    test_ratio = 1 - train_ratio

    # Load data from PreprocessingTraining
    data = PreprocessingTraining.query.all()
    result_preprocessing = [item.result for item in data]
    label = [item.label for item in data]

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(result_preprocessing)

    # Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(label)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=42)

    # SVM Classifier
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = generate_confusion_matrix(y_test, y_pred, labels=label_encoder.classes_)

    return render_template('evaluation/evaluation.html', active_page='evaluation', accuracy=accuracy, precision=precision, recall=recall, confusion_matrix=cm, selected_ratio=ratio)


if __name__ == '__main__':
    app.run(port=5000)