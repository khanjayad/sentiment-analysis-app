import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import urllib.parse
from flask import Flask, render_template, request
import feedparser
from transformers import pipeline
import plotly.express as px
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Load sentiment analysis model
classifier = pipeline("sentiment-analysis")

# Fetch headlines and links from Google News RSS
def fetch_headlines(keyword):
    rss_url = f"https://news.google.com/rss/search?q={keyword}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(rss_url)
    headlines = [(entry.title, entry.link) for entry in feed.entries[:20]]
    return headlines

# Analyze sentiment of each headline
def analyze_sentiment(headlines):
    texts = [title for title, link in headlines]
    results = classifier(texts)
    return [
        {"text": text, "url": url, "sentiment": r['label'], "score": round(r['score'], 2)}
        for (text, url), r in zip(headlines, results)
    ]

# Generate bar and pie charts
def generate_charts(sentiment_data):
    df = pd.DataFrame(sentiment_data)
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    bar_chart = px.bar(sentiment_counts, x='Sentiment', y='Count', title='Sentiment Analysis Results')
    pie_chart = px.pie(sentiment_counts, names='Sentiment', values='Count', title='Sentiment Distribution')
    return bar_chart.to_html(full_html=False), pie_chart.to_html(full_html=False)

# Generate wordcloud image
def generate_wordcloud(texts):
    wc = WordCloud(width=800, height=400, background_color='white', stopwords=STOPWORDS)
    wc.generate(" ".join(texts))
    buffer = BytesIO()
    wc.to_image().save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return img_str

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    headlines = []
    keyword = ""
    bar_chart = pie_chart = ""
    positive_wc = negative_wc = ""
    error = None

    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            error = "Please enter a keyword."
        else:
            raw_headlines = fetch_headlines(keyword)
            if not raw_headlines:
                error = "No headlines found."
            else:
                sentiment_data = analyze_sentiment(raw_headlines)
                headlines = sentiment_data

                positive = [s['text'] for s in sentiment_data if s['sentiment'] == "POSITIVE"]
                negative = [s['text'] for s in sentiment_data if s['sentiment'] == "NEGATIVE"]
                neutral = [s['text'] for s in sentiment_data if s['sentiment'] == "NEUTRAL"]

                bar_chart, pie_chart = generate_charts(sentiment_data)
                if positive:
                    positive_wc = f"data:image/png;base64,{generate_wordcloud(positive)}"
                if negative:
                    negative_wc = f"data:image/png;base64,{generate_wordcloud(negative)}"

                return render_template(
                    "index.html",
                    keyword=keyword,
                    headlines=headlines,
                    positive_count=len(positive),
                    negative_count=len(negative),
                    neutral_count=len(neutral),
                    bar_chart=bar_chart,
                    pie_chart=pie_chart,
                    positive_wc=positive_wc,
                    negative_wc=negative_wc,
                    error=None
                )

    return render_template("index.html", keyword=keyword, headlines=[], error=error)

if __name__ == '__main__':
    app.run(debug=True)
