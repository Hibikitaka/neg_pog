import os
import random
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import MeCab
import fasttext

# =========================================================
# 1. 感情辞書読み込み
# =========================================================
def load_polarity_dict(path):
    dic = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) >= 4:
                try:
                    dic[parts[0]] = float(parts[3])
                except ValueError:
                    pass
    return dic

# =========================================================
# 2. MeCab
# =========================================================
tagger = MeCab.Tagger("-Ochasen")

def get_words(text):
    node = tagger.parseToNode(text)
    words = []
    while node:
        features = node.feature.split(",")
        pos = features[0]
        base = features[6] if len(features) > 6 else "*"
        if pos in ["名詞", "形容詞", "動詞"]:
            words.append(base if base != "*" else node.surface)
        node = node.next
    return words if words else ["未解析"]

# =========================================================
# 3. ニュース読み込み（pos / neg 同数）
# =========================================================
def load_news_balanced(root, limit=None, seed=42):
    random.seed(seed)
    data = []

    for label, folder in [("__label__ポジ", "pos"), ("__label__ネガ", "neg")]:
        path = os.path.join(root, folder)
        for filename in sorted(os.listdir(path)):
            full = os.path.join(path, filename)
            if not os.path.isfile(full):
                continue
            with open(full, encoding="utf-8") as f:
                text = f.read().strip()
                if len(text) > 10:
                    data.append((text, label))

    random.shuffle(data)

    pos = [d for d in data if d[1] == "__label__ポジ"]
    neg = [d for d in data if d[1] == "__label__ネガ"]
    n = min(len(pos), len(neg))

    data = pos[:n] + neg[:n]
    random.shuffle(data)

    if limit:
        data = data[:limit]

    texts, labels = zip(*data)
    return list(texts), list(labels)

# =========================================================
# 4. train / test 分割
# =========================================================
def train_test_split(texts, labels, test_ratio=0.2):
    split = int(len(texts) * (1 - test_ratio))
    return (
        texts[:split], labels[:split],
        texts[split:], labels[split:]
    )

# =========================================================
# 5. 辞書方式予測
# =========================================================
def predict_dict(texts, dic, threshold):
    preds = []
    for text in texts:
        words = get_words(text)
        hit = [dic[w] for w in words if w in dic]
        score = sum(hit) / len(hit) if hit else 0
        preds.append("__label__ポジ" if score >= threshold else "__label__ネガ")
    return preds

# =========================================================
# 6. fastText用データ作成（改行除去）
# =========================================================
def make_fasttext_file(texts, labels, path):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(texts, labels):
            t = t.replace("\n", " ")
            f.write(f"{l} {t}\n")

# =========================================================
# 7. 実験本体
# =========================================================
def run_full_experiment(dic, news_root, counts_list):
    dict_f1 = []
    ft_f1 = []

    for count in counts_list:
        print(f"\n=== {count} 件で評価 ===")

        texts, labels = load_news_balanced(news_root, limit=count)
        tr_x, tr_y, te_x, te_y = train_test_split(texts, labels)

        # ---- 辞書方式 ----
        scores = []
        for t in tr_x:
            w = get_words(t)
            hit = [dic[x] for x in w if x in dic]
            scores.append(sum(hit) / len(hit) if hit else 0)

        threshold = statistics.median(scores)
        preds_dict = predict_dict(te_x, dic, threshold)
        f1_d = f1_score(te_y, preds_dict, average="macro")
        dict_f1.append(f1_d)

        # ---- fastText ----
        make_fasttext_file(tr_x, tr_y, "train.txt")
        make_fasttext_file(te_x, te_y, "test.txt")

        model = fasttext.train_supervised(
            "train.txt",
            epoch=20,
            lr=0.5,
            wordNgrams=2,
            verbose=0
        )

        preds_ft = [
            model.predict(t.replace("\n", " "))[0][0]
            for t in te_x
        ]
        f1_f = f1_score(te_y, preds_ft, average="macro")
        ft_f1.append(f1_f)

        print(f"辞書方式 F1 = {f1_d:.3f}")
        print(f"fastText F1 = {f1_f:.3f}")

    return dict_f1, ft_f1

# =========================================================
# 8. グラフ描画
# =========================================================
def plot_compare(counts, dict_f1, ft_f1):
    plt.figure(figsize=(10, 5))
    plt.plot(counts, dict_f1, marker="o", label="辞書方式")
    plt.plot(counts, ft_f1, marker="o", label="fastText")
    plt.xlabel("ニュース件数")
    plt.ylabel("F1スコア")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.title("データ件数とネガポジ分類精度の比較")
    plt.show()

# =========================================================
# 9. main
# =========================================================
def main():
    POLARITY_FILE = "sentiment_dict.txt"
    NEWS_ROOT = "text"

    dic = load_polarity_dict(POLARITY_FILE)
    counts_list = [500, 1000, 2000, 4000, 8000, 10000]

    dict_f1, ft_f1 = run_full_experiment(dic, NEWS_ROOT, counts_list)
    plot_compare(counts_list, dict_f1, ft_f1)

if __name__ == "__main__":
    main()  

