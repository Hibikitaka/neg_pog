from collections import Counter

def check_file(path, n=10):
    """
    ファイル内容の確認
    - 最初の n 行を表示
    - ラベル分布を表示
    """
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    print(f"--- {path} の最初の {n} 行 ---")
    for line in lines[:n]:
        print(line.strip())

    # ラベル分布を確認
    labels = [line.split()[0] for line in lines if line.strip()]
    counter = Counter(labels)
    print(f"\n--- {path} のラベル分布 ---")
    for label, count in counter.items():
        print(f"{label}: {count}")

if __name__ == "__main__":
    check_file("train.txt")
    check_file("test.txt")

def load_news_balanced(root, limit=None, seed=42):
    random.seed(seed)
    data = []

    for label, folder in [("__label__ポジ", "pos"), ("__label__ネガ", "neg")]:
        path = os.path.join(root, folder)
        print(f"Checking folder: {os.path.abspath(path)}")  # フルパス確認
        count = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                if not os.path.isfile(full):
                    continue
                print(f"Found file: {full}")  # 実際に見つかったファイル
                try:
                    with open(full, encoding="utf-8") as f:
                        text = f.read().strip()
                        if len(text) > 10:
                            data.append((text, label))
                            count += 1
                except UnicodeDecodeError:
                    continue
        print(f"{folder} 件数={count}")

    pos = [d for d in data if d[1] == "__label__ポジ"]
    neg = [d for d in data if d[1] == "__label__ネガ"]
    n = min(len(pos), len(neg))
    print(f"Total pos={len(pos)}, neg={len(neg)}")  # 件数確認
    if n == 0:
        raise ValueError("ポジまたはネガの記事が読み込まれていません。")

    data = pos[:n] + neg[:n]
    random.shuffle(data)

    if limit:
        data = data[:limit]

    texts, labels = zip(*data)
    return list(texts), list(labels)