import os
import gzip
import json
import re
from collections import defaultdict

def extract_date_from_url(url):
    # Regular expression to find yyyy/mm/dd in the URL
    match = re.search(r'\d{4}/\d{2}/\d{2}', url)
    return match.group(0) if match else None

def process_article_data(author_articles_dir):
    # A dictionary to hold articles text by date
    articles_by_date = defaultdict(list)

    # Iterate through each author's folder
    for author_id in os.listdir(author_articles_dir):
        author_folder = os.path.join(author_articles_dir, author_id)
        articles_file = os.path.join(author_folder, 'articles_text.json.gz')

        # Check if the compressed articles file exists
        if os.path.exists(articles_file):
            with gzip.open(articles_file, 'rt', encoding='utf-8') as file:
                articles_data = json.load(file)
                for url, text in articles_data[0].items():
                    date = extract_date_from_url(url)
                    if date:
                        articles_by_date[date].append(text)

    return articles_by_date

def save_articles_by_date(articles_by_date, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save articles in folders named by date
    for date, articles in articles_by_date.items():
        date_folder = os.path.join(output_dir, date.replace('/', '-'))
        os.makedirs(date_folder, exist_ok=True)

        # Save each article in a separate file
        for i, article in enumerate(articles):
            with open(os.path.join(date_folder, f'article_{i}.txt'), 'w', encoding='utf-8') as file:
                file.write(article)

# Main process
author_articles_dir = 'data/author_articles/'
output_dir = 'data/fool_articles/dated_articles/'
articles_by_date = process_article_data(author_articles_dir)
save_articles_by_date(articles_by_date, output_dir)
