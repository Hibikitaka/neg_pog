import os

def count_articles(root="text"):
    """
    pos / neg フォルダの記事数をカウント
    """
    for label, folder in [("ポジ", "pos"), ("ネガ", "neg")]:
        path = os.path.join(root, folder)
        if not os.path.exists(path):
            print(f"{folder} フォルダが存在しません")
            continue

        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        print(f"{label}記事数: {len(files)}")

if __name__ == "__main__":
    count_articles()
