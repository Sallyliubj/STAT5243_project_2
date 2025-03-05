### Note: this script is migrated from https://github.com/johnfeng2023/GR5243_project_1/blob/main/data.py

import yfinance as yf
import pandas as pd
from datetime import datetime
import requests
import time

sp500 = yf.Ticker("^GSPC")
sp500_df = sp500.history(start="2024-01-01", end="2024-12-31")
# Remove the timezone and extract only the date part
sp500_df.index = sp500_df.index.tz_localize(None).date
sp500_df["Date"] = sp500_df.index

guardian_api_key = "api_key_here" # Insert your own guardian api key here
base_url = "https://content.guardianapis.com/search"


# query parameters for S&P 500 news in 2024 from guardian api
params = {
    "q": "S&P 500",
    "from-date": "2024-01-01",
    "to-date": "2024-12-31",
    "api-key": guardian_api_key,
    "show-fields": "headline,body",
    "page-size": 50,
    "page": 1,
}

articles = []

while True:
    response = requests.get(base_url, params=params)
    data = response.json()

    # Extract articles from the current page
    results = data["response"]["results"]
    if not results:
        break
    articles.extend(results)

    # Check if we've retrieved all pages
    current_page = data["response"]["currentPage"]
    total_pages = data["response"]["pages"]
    if current_page >= total_pages:
        break
    params["page"] += 1
    time.sleep(1)


news_df = pd.DataFrame(
    [
        {
            "id": article.get("id"),
            "type": article.get("type"),
            "sectionName": article.get("sectionName"),
            "publicationDate": article.get("webPublicationDate"),
            "webTitle": article.get("webTitle"),
            "webUrl": article.get("webUrl"),
            "headline": article.get("fields", {}).get("headline"),
            "body": article.get("fields", {}).get("body"),
        }
        for article in articles
    ]
)

news_df["publicationDate"] = pd.to_datetime(news_df["publicationDate"]).dt.date
news_df = news_df.sort_values(by="publicationDate", ascending=True)

merged_df = pd.merge(
    sp500_df, news_df, how="left", left_on="Date", right_on="publicationDate"
)

sp500_df.to_csv("./dataset/sp500_df")
news_df.to_csv("./dataset/news_df")
merged_df.to_csv("./dataset/merged_df")
