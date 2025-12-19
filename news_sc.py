# news_sc_txt.py
import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import WebDriverException

# 保存先ディレクトリ
SAVE_DIR = "text/yahoo/"
os.makedirs(SAVE_DIR, exist_ok=True)

# Yahooニュースカテゴリ
CATEGORIES = {
    "top": "https://news.yahoo.co.jp/",
    "domestic": "https://news.yahoo.co.jp/categories/domestic",
    "world": "https://news.yahoo.co.jp/categories/world",
    "business": "https://news.yahoo.co.jp/categories/business",
    "it": "https://news.yahoo.co.jp/categories/it"
}

# Selenium初期化
def init_driver():
    options = Options()
    options.add_argument("--headless")  # ヘッドレスモード
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    return driver

# 記事本文取得
def get_article_text(driver, url):
    try:
        driver.get(url)
        time.sleep(1)
        selectors = [
            "div.article_body p",
            "div.article_body.highLightSearchTarget p",
            "article p"
        ]
        for sel in selectors:
            elems = driver.find_elements("css selector", sel)
            if elems:
                text = "\n".join([e.text for e in elems if e.text.strip()])
                if text:
                    return text
        print(f"本文抽出失敗（本文セレクタ不一致）: {url}")
        return None
    except WebDriverException as e:
        print(f"本文取得失敗: {url} ({e})")
        return None

# カテゴリ記事URL取得
def get_article_urls(driver, category_url):
    try:
        driver.get(category_url)
        time.sleep(1)
        links = driver.find_elements("css selector", "a")
        urls = []
        for a in links:
            href = a.get_attribute("href")
            if href and "/articles/" in href:
                urls.append(href)
        return list(set(urls))  # 重複除去
    except WebDriverException as e:
        print(f"記事URL取得失敗: {category_url} ({e})")
        return []

# メイン処理
def main():
    print("=== Yahooニューススクレイピング開始 ===")
    driver = init_driver()
    all_urls = []
    try:
        # カテゴリ別にURL取得
        for cat, url in CATEGORIES.items():
            print(f"カテゴリ {cat}: {url}")
            urls = get_article_urls(driver, url)
            print(f" → {len(urls)}件取得")
            all_urls.extend([(cat, u) for u in urls])
        all_urls = list(set(all_urls))  # 重複除去
        print(f"総取得記事数: {len(all_urls)} 件")

        # 記事ごとに本文取得 & TXT保存
        for idx, (cat, url) in enumerate(all_urls, 1):
            print(f"[{idx}/{len(all_urls)}] {url}")
            text = get_article_text(driver, url)
            if not text:
                print(" → 本文取得失敗、スキップ")
                continue
            filename = f"news_{int(time.time()*1000)}.txt"
            filepath = os.path.join(SAVE_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"URL: {url}\n")
                f.write(f"Category: {cat}\n\n")
                f.write(text)
            print(f"保存: {filepath}")

    finally:
        driver.quit()
        print("=== 終了 ===")

if __name__ == "__main__":
    main()
