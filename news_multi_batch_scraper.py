import asyncio
import aiohttp
import aiofiles
import os
import time
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

BASE_DIR = "text"
os.makedirs(BASE_DIR, exist_ok=True)

CATEGORIES = ["domestic", "world", "it", "economy", "entertainment", "sports", "local"]
BASE_URL = "https://news.yahoo.co.jp/categories/{}"
MAX_PAGE = 20
PAST_DAYS = 7  # 過去N日分の記事だけ収集
URLS_SEEN_FILE = "urls_seen.txt"

# HTMLから本文抽出
def extract_article_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    paragraphs = soup.select("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    return text

# 既存URL読み込み
def load_seen_urls():
    if not os.path.exists(URLS_SEEN_FILE):
        return set()
    with open(URLS_SEEN_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

# URL保存
def save_seen_urls(urls):
    with open(URLS_SEEN_FILE, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")

# 記事HTML取得&保存
async def fetch_article(session, url, category, cutoff_time):
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            html = await resp.text()
            # 日付チェック: 記事内の<meta>や時刻情報から判定
            soup = BeautifulSoup(html, "html.parser")
            time_tag = soup.find("time")
            if time_tag and time_tag.has_attr("datetime"):
                article_time = datetime.fromisoformat(time_tag["datetime"])
                if article_time < cutoff_time:
                    return None  # 過去指定日より古い
            article_text = extract_article_text(html)
            safe_name = url.replace("https://", "").replace("/", "_").replace("?", "_").replace("=", "_")
            filename = os.path.join(BASE_DIR, category, f"{safe_name}_{int(time.time()*1000)}.txt")
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            async with aiofiles.open(filename, mode='w', encoding='utf-8') as f:
                await f.write(article_text if article_text else html)
            return url
    except:
        return None

# カテゴリページから記事リンク抽出
async def fetch_category_page(session, category, page):
    url = f"{BASE_URL.format(category)}?page={page}"
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return []
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            links = [a.get("href") for a in soup.select("a") if a.get("href") and "/articles/" in a.get("href")]
            return list(set(links))
    except:
        return []

# カテゴリ収集
async def collect_category(session, category, seen_urls, cutoff_time):
    all_new_urls = []
    for page in range(1, MAX_PAGE+1):
        urls = await fetch_category_page(session, category, page)
        urls = [u for u in urls if u not in seen_urls]
        if not urls:
            continue
        tasks = [fetch_article(session, url, category, cutoff_time) for url in urls]
        results = await asyncio.gather(*tasks)
        new_urls = [r for r in results if r]
        all_new_urls.extend(new_urls)
        seen_urls.update(new_urls)
    return all_new_urls

async def main():
    cutoff_time = datetime.now() - timedelta(days=PAST_DAYS)
    seen_urls = load_seen_urls()
    async with aiohttp.ClientSession() as session:
        total_new = 0
        for category in CATEGORIES:
            new_urls = await collect_category(session, category, seen_urls, cutoff_time)
            total_new += len(new_urls)
            print(f"カテゴリ {category} 新規取得記事数: {len(new_urls)}")
        save_seen_urls(seen_urls)
        print(f"総新規取得記事数: {total_new}")

if __name__ == "__main__":
    asyncio.run(main())