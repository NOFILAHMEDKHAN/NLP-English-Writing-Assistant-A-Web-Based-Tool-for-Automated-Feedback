import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob
import random
import PyPDF2
import os
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64
from werkzeug.utils import secure_filename
import pandas as pd

# Download required NLTK data (uncomment if running first time)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'pdf', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class EnglishLearningAssistant:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.feedback_history = []
        
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
        except Exception as e:
            return f"Error reading PDF: {e}"
        
    def extract_text_from_excel(self, excel_path):
        """Extract text from an Excel file"""
        try:
            df = pd.read_excel(excel_path)
            text = " ".join(df.astype(str).fillna('').values.flatten())
            return text
        except Exception as e:
            return f"Error reading Excel: {e}"
        
    def analyze_text(self, text):
        """Comprehensive text analysis with educational feedback"""
        analysis = {}
        
        # Tokenization and basic stats
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        content_words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        analysis['sentence_count'] = len(sentences)
        analysis['word_count'] = len(words)
        analysis['unique_words'] = len(set(words))
        analysis['vocab_richness'] = round(analysis['unique_words'] / len(words) * 100, 1) if words else 0
        
        # POS Analysis
        pos_tags = pos_tag(words)
        analysis['pos_tags'] = pos_tags
        
        # Common grammar patterns
        verb_forms = [word for word, tag in pos_tags if tag.startswith('V')]
        nouns = [word for word, tag in pos_tags if tag.startswith('N')]
        adjectives = [word for word, tag in pos_tags if tag.startswith('J')]
        
        analysis['verbs'] = verb_forms
        analysis['nouns'] = nouns
        analysis['adjectives'] = adjectives
        
        # Sentiment and complexity
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.3:
            sentiment_label = "Positive"
        elif polarity < -0.3:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"
        analysis['sentiment'] = sentiment_label
        analysis['avg_sentence_length'] = round(len(words) / len(sentences), 1) if sentences else 0
        
        return analysis
    
    def generate_feedback(self, analysis):
        """Generate personalized learning feedback"""
        feedback = []
        
        # Vocabulary feedback
        if analysis['vocab_richness'] < 30:
            feedback.append("Try using more varied vocabulary. You're repeating words quite a bit.")
        elif analysis['vocab_richness'] > 60:
            feedback.append("Great vocabulary diversity! You're using a wide range of words.")
        
        # Sentence structure feedback
        if analysis['avg_sentence_length'] > 20:
            feedback.append("Your sentences are quite long. Try breaking them up for better clarity.")
        elif analysis['avg_sentence_length'] < 8:
            feedback.append("Your sentences are very short. Try combining some ideas for better flow.")
        
        # Parts of speech feedback
        if len(analysis['adjectives']) < 2:
            feedback.append("Add more descriptive words (adjectives) to make your writing more vivid.")
        
        if len(analysis['verbs']) < 3:
            feedback.append("Try using more action words to make your writing more dynamic.")
        
        # Sentiment feedback
        if analysis['sentiment'] == "Positive":
            feedback.append("Your writing has a positive tone! 😊")
        elif analysis['sentiment'] == "Negative":
            feedback.append("Your writing has a negative tone. 😟")
        
        return feedback
    
    def practice_suggestion(self):
        """Suggest practice based on user's needs"""
        practices = [
            "Try writing about your day using at least 5 different verbs.",
            "Describe your favorite place using plenty of adjectives.",
            "Write a short story with sentences of varying lengths.",
            "Practice using compound sentences with 'and', 'but', and 'because'.",
            "Try using at least three new words you learned recently."
        ]
        return random.choice(practices)
    
    def create_pie_chart(self, analysis):
        """Create a pie chart of parts of speech distribution"""
        nouns_count = len(analysis['nouns'])
        verbs_count = len(analysis['verbs'])
        adjectives_count = len(analysis['adjectives'])
        others_count = analysis['word_count'] - (nouns_count + verbs_count + adjectives_count)
        
        # Ensure we don't have negative values
        others_count = max(0, others_count)
        
        labels = ['Nouns', 'Verbs', 'Adjectives', 'Others']
        sizes = [nouns_count, verbs_count, adjectives_count, others_count]
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Filter out zero values
        non_zero_sizes = []
        non_zero_labels = []
        non_zero_colors = []
        for i, size in enumerate(sizes):
            if size > 0:
                non_zero_sizes.append(size)
                non_zero_labels.append(labels[i])
                non_zero_colors.append(colors[i])
        
        if not non_zero_sizes:
            return None
            
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title('Parts of Speech Distribution')
        
        # Convert plot to base64 for HTML embedding
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        
        return plot_url

assistant = EnglishLearningAssistant()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'})
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    text = ""
    try:
        file.save(filename)
        if ext == 'pdf':
            text = assistant.extract_text_from_pdf(filename)
        elif ext in ['xlsx', 'xls']:
            text = assistant.extract_text_from_excel(filename)
        os.remove(filename)
    except Exception as e:
        return jsonify({'error': f'Error processing file: {e}'})

    if not text or len(text.strip()) < 10 or (isinstance(text, str) and text.startswith("Error")):
        return jsonify({'error': f'Could not extract enough text from {filename}. Error: {text}'})

    # Analyze the text
    analysis = assistant.analyze_text(text)
    feedback = assistant.generate_feedback(analysis)
    practice = assistant.practice_suggestion()
    plot_url = assistant.create_pie_chart(analysis)

    response = {
        'filename': filename,
        'analysis': analysis,
        'feedback': feedback,
        'practice': practice,
        'plot_url': plot_url
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
