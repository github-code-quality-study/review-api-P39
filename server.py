import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        if environ["REQUEST_METHOD"] == "GET":
            # Parse query parameters
            query_string = environ['QUERY_STRING']
            query_params = parse_qs(query_string)

            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews

            # Filter by location
            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            # Filter by date range
            if start_date and end_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [
                    review for review in filtered_reviews
                    if start_date <= datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date
                ]

            # Analyze sentiment for each review
            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = sentiment

            # Sort by compound sentiment score in descending order
            filtered_reviews.sort(key=lambda x: x['sentiment']['compound'], reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

        if environ["REQUEST_METHOD"] == "POST":
            try:
                request_body_size = int(environ.get('CONTENT_LENGTH', 0))
            except (ValueError):
                request_body_size = 0

            request_body = environ['wsgi.input'].read(request_body_size)
            post_data = parse_qs(request_body.decode('utf-8'))

            review_body = post_data.get('ReviewBody', [None])[0]
            location = post_data.get('Location', [None])[0]

            if not review_body or not location:
                response_body = json.dumps({'error': 'Missing ReviewBody or Location'}).encode('utf-8')
                start_response("400 Bad Request", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])
                return [response_body]

            new_review = {
                'ReviewId': str(uuid.uuid4()),
                'ReviewBody': review_body,
                'Location': location,
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            reviews.append(new_review)
            response_body = json.dumps(new_review, indent=2).encode('utf-8')

            start_response("201 Created", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])
            return [response_body]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8080))
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()