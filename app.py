import os
import boto3
import csv
import pandas as pd
from flask import Flask, request, render_template
from flask_cors import CORS
import uuid
import tempfile
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# AWS Configuration from Environment Variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
region_name = os.getenv('AWS_REGION', 'ap-south-1')
bucket_name = os.getenv('BUCKET_NAME', 'cccpbucket1')

# Initialize AWS clients
s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key, region_name=region_name)
comprehend_client = boto3.client('comprehend', aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key, region_name=region_name)

# Helper function to upload a local file to S3
def upload_local_file_to_s3(local_file_path, bucket_name, s3_file_name):
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_file_name)
        print(f"Local file {local_file_path} uploaded to {bucket_name} as {s3_file_name}")
        return True
    except Exception as e:
        print(f"Error uploading local file: {str(e)}")
        return False

# Helper function to perform Comprehend analysis
def analyze_with_comprehend(text):
    sentiment_response = comprehend_client.detect_sentiment(Text=text, LanguageCode='en')
    key_phrases_response = comprehend_client.detect_key_phrases(Text=text, LanguageCode='en')
    return {'Sentiment': sentiment_response, 'KeyPhrases': key_phrases_response}

# Helper function to process CSV file and generate a results CSV
def process_csv(file):
    results = []
    file.seek(0)  # Reset file pointer to the beginning

    try:
        csv_reader = csv.reader(file.read().decode('utf-8').splitlines())
    except UnicodeDecodeError:
        file.seek(0)
        csv_reader = csv.reader(file.read().decode('ISO-8859-1').splitlines())

    for row in csv_reader:
        if len(row) > 0:
            text = row[0]  # Assuming the first column contains the text for analysis
            analysis_result = analyze_with_comprehend(text)
            sentiment = analysis_result['Sentiment']['Sentiment']
            results.append({'Review': text, 'Sentiment': sentiment})

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(results)
    csv_file_path = os.path.join(tempfile.gettempdir(), 'sentiment_analysis_results.csv')
    df.to_csv(csv_file_path, index=False)

    # Upload results CSV to S3
    upload_local_file_to_s3(csv_file_path, bucket_name, os.path.basename(csv_file_path))

    return csv_file_path

# Function to visualize sentiment trends
def visualize_sentiment_trends(csv_file_path):
    # Load the CSV into a DataFrame
    df = pd.read_csv(csv_file_path)

    if 'Sentiment' not in df.columns:
        raise ValueError("CSV must contain 'Sentiment' column.")

    # Sentiment Distribution Visualization (Bar Graph)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=df['Sentiment'].value_counts().index, y=df['Sentiment'].value_counts().values, palette='viridis')
    plt.title('Overall Sentiment Distribution (Bar Graph)')
    plt.xlabel('Sentiment')
    plt.ylabel('Number of Reviews')
    plt.xticks(rotation=45)

    bar_image_stream = BytesIO()
    plt.savefig(bar_image_stream, format='png')
    plt.close()
    bar_image_stream.seek(0)

    # Pie Chart for Sentiment Distribution
    plt.figure(figsize=(8, 8))
    plt.pie(df['Sentiment'].value_counts(), labels=df['Sentiment'].value_counts().index, autopct='%1.1f%%', startangle=140)
    plt.title('Sentiment Distribution (Pie Chart)')
    
    pie_image_stream = BytesIO()
    plt.savefig(pie_image_stream, format='png')
    plt.close()
    pie_image_stream.seek(0)

    # Word Frequency Graph
    all_reviews = ' '.join(df['Review'].tolist())
    word_counts = Counter(all_reviews.split())
    most_common_words = word_counts.most_common(10)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=[word[0] for word in most_common_words], y=[word[1] for word in most_common_words], palette='magma')
    plt.title('Most Common Words in Reviews')
    plt.xlabel('Words')
    plt.ylabel('Frequency')

    word_frequency_image_stream = BytesIO()
    plt.savefig(word_frequency_image_stream, format='png')
    plt.close()
    word_frequency_image_stream.seek(0)

    return bar_image_stream, pie_image_stream, word_frequency_image_stream

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    bar_image_url = None
    pie_image_url = None
    word_frequency_image_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part', 400
        
        file = request.files['file']
        if file.filename == '':
            return 'No selected file', 400
        
        if file:
            file_name = f"{uuid.uuid4()}_{file.filename}"
            
            if file.filename.endswith('.csv'):
                result_csv_path = process_csv(file)

                # Visualize sentiment trends
                bar_image_stream, pie_image_stream, word_frequency_image_stream = visualize_sentiment_trends(result_csv_path)

                # Save the images to a static directory
                os.makedirs('static', exist_ok=True)
                bar_image_path = os.path.join('static', 'sentiment_bar.png')
                pie_image_path = os.path.join('static', 'sentiment_pie.png')
                word_frequency_image_path = os.path.join('static', 'word_frequency.png')

                with open(bar_image_path, 'wb') as f:
                    f.write(bar_image_stream.getvalue())
                
                with open(pie_image_path, 'wb') as f:
                    f.write(pie_image_stream.getvalue())

                with open(word_frequency_image_path, 'wb') as f:
                    f.write(word_frequency_image_stream.getvalue())

                # Set the URLs for the images
                bar_image_url = f"/static/{os.path.basename(bar_image_path)}"
                pie_image_url = f"/static/{os.path.basename(pie_image_path)}"
                word_frequency_image_url = f"/static/{os.path.basename(word_frequency_image_path)}"

                # Render the template with image URLs
                return render_template('index.html', 
                                       bar_image=bar_image_url, 
                                       pie_image=pie_image_url,
                                       word_frequency_image=word_frequency_image_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
