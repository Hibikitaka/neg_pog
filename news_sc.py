# multi_news_scraper_large.py
import os
import time
import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

SAVE_DIR = "text/multi_news/"
os.makedirs(SAVE_DIR, exist_ok=True)

# 対象サイトとページネーションURLのテンプレート
SITES = {
    "yahoo_top": {
        "base_url": "https://news.yahoo.co.jp/pickup/past?page={page}",
        "max_page": 500  # 過去記事ページ数
    },
    "itmedia": {
        "base_url": "https://www.itmedia.co.jp/news/articles/{year}/{month}/{page}.html",
        "max_page": 300
    },
    "cnet": {
        "base_url": "https://japan.cnet.com/archive/{page}/",
        "max_page": 300
    }
}

def init_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def get_article_urls(driver, url):
    try:
        driver.get(url)
        time.sleep(0.5)
        links = driver.find_elements("css selector", "a")
        urls = []
        for a in links:
            href = a.get_attribute("href")
            if href and "article" in href:  # 各サイトごとに条件調整
                urls.append(href)
        return list(set(urls))
    except WebDriverException:
        return []

def get_article_text(driver, url):
    try:
        driver.get(url)
        time.sleep(0.5)
        selectors = ["div.article_body p", "article p"]
        for sel in selectors:
            elems = driver.find_elements("css selector", sel)
            if elems:
                text = "\n".join([e.text for e in elems if e.text.strip()])
                if text:
                    return text
        return None
    except WebDriverException:
        return None

def save_article(category, url, text):
    timestamp = int(time.time() * 1000)
    filename = f"{category}_{timestamp}.txt"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\nCategory: {category}\n\n{text}")
    return filepath

def process_url(driver, category, url):
    text = get_article_text(driver, url)
    if text:
        return save_article(category, url, text)
    return None

def main():
    print("=== ニュース記事収集開始 ===")
    driver = init_driver()
    all_urls = set()

    try:
        # サイトごとにページネーションを回してURL取得
        for site, info in SITES.items():
            print(f"サイト {site} の記事URL取得中...")
            base_url = info["base_url"]
            max_page = info["max_page"]
            for page in range(1, max_page + 1):
                url = base_url.format(page=page, year=2025, month="12")
                urls = get_article_urls(driver, url)
                all_urls.update([(site, u) for u in urls])
                if page % 50 == 0:
                    print(f" → {len(all_urls)} 件取得中 (page {page})")
        all_urls = list(all_urls)
        print(f"総取得記事数（重複除去後）: {len(all_urls)} 件")

        # 並列で本文取得 & 保存
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_url, driver, site, url) for site, url in all_urls]
            for idx, f in enumerate(concurrent.futures.as_completed(futures), 1):
                result = f.result()
                if result:
                    print(f"[{idx}/{len(futures)}] 保存: {result}")

    finally:
        driver.quit()
        print("=== 収集終了 ===")

if __name__ == "__main__":
    main()
