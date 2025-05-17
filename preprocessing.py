import string
import re
import emoji
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Casefolding
def casefolding(text):
    return text.lower()

# Cleasing
def cleansing(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove mention
    text = re.sub(r"@\w+", '', text)
    # Remove hashtags
    text = re.sub(r"#\w+", '', text)
    # Remove URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove emoji
    text = emoji.replace_emoji(text, replace='')
    # Remove character non-alphabet
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spacing or new line and trimming
    text = re.sub(r"\s+", ' ', text).strip()
    # Remove repetition words
    text = re.sub(r'\b(\w+)-\1\b', r'\1', text)
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    return text

# Tokenization
def tokenization(text):
    return word_tokenize(text)

# Normalization
normalisasi_dict = {
    "gak": "tidak", "udah": "sudah", "belanja": "berbelanja", "beli": "membeli", "coba": "mencoba", "pas": "saat", "nanti": "kemudian",
    "kurang": "tidak cukup", "super": "sangat", "terbaik": "bagus", "kecewa": "tidak puas", "kali": "mungkin", "nyoba": "mencoba",
    "sempat": "pernah", "makin": "semakin", "error": "kesalahan", "padahal": "namun", "fitur": "fungsi", "limit": "batas", "fast": "cepat",
    "barang": "produk", "bayar": "membayar", "impian": "keinginan", "kasih": "terima kasih", "lama": "waktu lama", "takut": "cemas",
    "tanpa": "tidak dengan", "terima": "menerima", "karena": "sebab", "gue" : "saya", "kalo": "kalau", "yg": "yang", "pake": "menggunakan",
    "knp" : "kenapa", "tiap": "setiap", "klik": "tekan", "bahaya": "berisiko", "buat": "untuk", "bisa": "dapat", "bisa aja": "mungkin",
    "emang": "memang", "beda" : "berbeda", "ngebantu": "membantu",  "lumayan": "cukup",  "nyari": "mencari", "sampai": "hingga", "inget": "ingat",
    "tetep": "tetap", "cuma": "hanya", "biar" : "agar", "ngotak" : "masuk akal"
}
def normalize(text):
    return [normalisasi_dict.get(word, word) for word in text]

# Stopword
def stopword_removal(text):
    factory = StopWordRemoverFactory()
    custom_stopwords = ["btw", "nih", "Nih", "anjirr", "anjir", "anjirrrrrrrr", "Btw", "ya", "Anjingggggg", "ngentot", "kontol", "bangsatnya", "tai", "deh", "&amp;", "hahaha", "wkwkwk"]
    stop_word = factory.get_stop_words() + custom_stopwords
    return [word for word in text if word not in stop_word]


# Stemming
def stemming(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return [stemmer.stem(word) for word in text]