import os
import random
from collections import defaultdict
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
        if pos in ["名詞", "形容詞", "動詞", "副詞"]:
            words.append(base if base != "*" else node.surface)
        node = node.next
    return words

# =========================================================
# 3. 辞書語優先抽出
# =========================================================


def extract_dict_words(text, dic):
    words = get_words(text)
    hits = [w for w in words if w in dic]
    return hits if hits else words

# =========================================================
# 4. ニュース読み込み
# =========================================================


def load_news_from_files(root, seed=42):
    random.seed(seed)
    data = []
    for label, folder in [("__label__ポジ", "pos"), ("__label__ネガ", "neg")]:
        path = os.path.join(root, folder)
        count = 0
        for dirpath, _, filenames in os.walk(path):
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                if not os.path.isfile(full):
                    continue
                try:
                    with open(full, encoding="utf-8") as f:
                        articles = [line.strip() for line in f if line.strip()]
                        for article in articles:
                            if len(article) > 10:
                                data.append((article, label))
                                count += 1
                except UnicodeDecodeError:
                    continue
        print(f"{folder} 件数={count}")
    pos = [d for d in data if d[1] == "__label__ポジ"]
    neg = [d for d in data if d[1] == "__label__ネガ"]
    n = min(len(pos), len(neg))
    data = pos[:n] + neg[:n]
    random.shuffle(data)
    texts, labels = zip(*data)
    return list(texts), list(labels)

# =========================================================
# 5. stratified train/test split
# =========================================================


def stratified_split(texts, labels, test_ratio=0.2, seed=42):
    random.seed(seed)
    buckets = defaultdict(list)
    for t, l in zip(texts, labels):
        buckets[l].append(t)

    tr_x, tr_y, te_x, te_y = [], [], [], []
    for label, items in buckets.items():
        random.shuffle(items)
        split = int(len(items) * (1 - test_ratio))
        tr_x.extend(items[:split])
        tr_y.extend([label] * split)
        te_x.extend(items[split:])
        te_y.extend([label] * (len(items) - split))
    return tr_x, tr_y, te_x, te_y

# =========================================================
# 6. 辞書方式でスコア取得
# =========================================================


def get_sentiment_scores(texts, dic):
    """文章ごとのポジネガ度合いスコアを返す"""
    scores = []
    for text in texts:
        words = get_words(text)
        if not words:
            scores.append(0.0)
            continue
        dict_scores = [dic.get(w, 0.0) for w in words]
        score = sum(dict_scores) / len(words)
        scores.append(score)
    return scores


def predict_dict_label(scores, threshold=0.0):
    """スコアに基づきラベルに変換"""
    return ["__label__ポジ" if s > threshold else "__label__ネガ" for s in scores]

# =========================================================
# 7. fastText 用ファイル作成
# =========================================================


def make_fasttext_file(texts, labels, dic, path):
    with open(path, "w", encoding="utf-8") as f:
        for t, l in zip(texts, labels):
            toks = extract_dict_words(t, dic)
            if not toks:
                toks = get_words(t)
            f.write(f"{l} {' '.join(toks)}\n")

# =========================================================
# 8. 件数ごとの実験（度合いスコアも取得）
# =========================================================


def run_counts_experiment(dic, news_root, counts_list):
    dict_f1_list = []
    ft_f1_list = []

    texts, labels = load_news_from_files(news_root)

    for count in counts_list:
        print(f"\n=== {count} 件で評価 ===")
        if count < len(texts):
            texts_count, labels_count = texts[:count], labels[:count]
        else:
            texts_count, labels_count = texts, labels
        tr_x, tr_y, te_x, te_y = stratified_split(texts_count, labels_count)
        print(
            f"train: ポジ={tr_y.count('__label__ポジ')}, ネガ={tr_y.count('__label__ネガ')}")
        print(
            f"test:  ポジ={te_y.count('__label__ポジ')}, ネガ={te_y.count('__label__ネガ')}")

        # 辞書方式スコア
        scores_dict = get_sentiment_scores(te_x, dic)
        preds_dict = predict_dict_label(scores_dict)
        f1_d = f1_score(te_y, preds_dict, average="macro")
        dict_f1_list.append(f1_d)

        # fastText
        make_fasttext_file(tr_x, tr_y, dic, "train.txt")
        make_fasttext_file(te_x, te_y, dic, "test.txt")
        print("Training fastText model...")
        model = fasttext.train_supervised(
            input="train.txt",
            epoch=50,
            lr=1.0,
            wordNgrams=2,
            minCount=1,
            loss="softmax",
            verbose=0
        )
        # 確率を取得してポジ度合い
        scores_ft = [model.predict(" ".join(extract_dict_words(t, dic)))[1][0]
                     if model.predict(" ".join(extract_dict_words(t, dic)))[0][0] == "__label__ポジ"
                     else 1 - model.predict(" ".join(extract_dict_words(t, dic)))[1][0]
                     for t in te_x]
        preds_ft = ["__label__ポジ" if s >
                    0.5 else "__label__ネガ" for s in scores_ft]
        f1_f = f1_score(te_y, preds_ft, average="macro")
        ft_f1_list.append(f1_f)

        print(f"辞書方式 F1 = {f1_d:.3f}")
        print(f"fastText F1 = {f1_f:.3f}")

        # ヒストグラムで分布可視化
        plt.figure(figsize=(10, 4))
        plt.hist(scores_dict, bins=20, alpha=0.5, label="辞書方式")
        plt.hist(scores_ft, bins=20, alpha=0.5, label="fastText")
        plt.xlabel("ポジティブ度")
        plt.ylabel("件数")
        plt.title(f"{count} 件でのポジネガ度分布")
        plt.legend()
        plt.show()

    return dict_f1_list, ft_f1_list

# =========================================================
# 9. F1比較グラフ
# =========================================================


def plot_results(counts, dict_f1, ft_f1):
    plt.figure(figsize=(10, 5))
    plt.plot(counts, dict_f1, marker="o", label="辞書方式改善版")
    plt.plot(counts, ft_f1, marker="o", label="fastText")
    plt.xlabel("使用件数")
    plt.ylabel("F1スコア")
    plt.ylim(0, 1.05)
    plt.title("ニュース件数増加によるネガポジ分類精度の比較")
    plt.grid(True)
    plt.legend()
    plt.show()

# =========================================================
# 10. main
# =========================================================


def main():
    POLARITY_FILE = "sentiment_dict.txt"
    NEWS_ROOT = "text"
    counts_list = [10, 50, 100, 200, 500, 1000, 2000]

    dic = load_polarity_dict(POLARITY_FILE)
    dict_f1, ft_f1 = run_counts_experiment(dic, NEWS_ROOT, counts_list)
    plot_results(counts_list, dict_f1, ft_f1)


if __name__ == "__main__":
    main()
