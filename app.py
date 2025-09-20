# app.py

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob
import random
import PyPDF2
import os
import glob
from flask import Flask, render_template, request, jsonify
import matplotlib.pyplot as plt
import io
import base64

# Download required NLTK data (uncomment if running first time)
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

app = Flask(__name__)

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
        analysis['sentiment'] = TextBlob(text).sentiment.polarity
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
        if analysis['sentiment'] > 0.3:
            feedback.append("Your writing has a positive tone! 😊")
        elif analysis['sentiment'] < -0.3:
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
    # Get PDF files in current directory
    pdf_files = glob.glob("*.pdf")
    return render_template('index.html', pdf_files=pdf_files)

@app.route('/analyze', methods=['POST'])
def analyze():
    pdf_file = request.form.get('pdf_file')
    
    if not pdf_file:
        return jsonify({'error': 'No PDF file selected'})
    
    # Extract text from PDF
    text = assistant.extract_text_from_pdf(pdf_file)
    
    if not text or len(text.strip()) < 10 or text.startswith("Error"):
        return jsonify({'error': f'Could not extract enough text from {pdf_file}. Error: {text}'})
    
    # Analyze the text
    analysis = assistant.analyze_text(text)
    
    # Generate feedback
    feedback = assistant.generate_feedback(analysis)
    
    # Get practice suggestion
    practice = assistant.practice_suggestion()
    
    # Create visualization
    plot_url = assistant.create_pie_chart(analysis)
    
    # Prepare response
    response = {
        'pdf_file': pdf_file,
        'analysis': analysis,
        'feedback': feedback,
        'practice': practice,
        'plot_url': plot_url
    }
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
