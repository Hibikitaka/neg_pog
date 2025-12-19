# news_sentiment_analysis.py

import os
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import MeCab

# ---------------------------------------------------------
# 1. 感情辞書読み込み
# ---------------------------------------------------------
def load_polarity_dict(path: str) -> dict:
    polarity_dict = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(":")
            if len(parts) < 4:
                continue

            word = parts[0]
            try:
                score = float(parts[3])
            except ValueError:
                continue

            polarity_dict[word] = score

    return polarity_dict


# ---------------------------------------------------------
# 2. MeCabで形態素解析
# ---------------------------------------------------------
tagger = MeCab.Tagger("-Ochasen")


def get_words(text: str) -> list:
    node = tagger.parseToNode(text)
    words = []

    while node:
        features = node.feature.split(",")
        pos = features[0]
        base = features[6] if len(features) > 6 else "*"

        if pos in ["名詞", "形容詞", "動詞"]:
            if base != "*" and base != "":
                words.append(base)
            else:
                words.append(node.surface)

        node = node.next

    if not words:
        words.append("未解析")

    return words


# ---------------------------------------------------------
# 3. ニュース記事読み込み
# ---------------------------------------------------------
def load_news(path: str, limit: int | None = None) -> list:
    texts = []

    if not os.path.exists(path):
        print(f"⚠ ニュースフォルダが存在しません: {path}")
        return texts

    files = sorted(os.listdir(path))
    if limit:
        files = files[:limit]

    for filename in files:
        full_path = os.path.join(path, filename)
        if not os.path.isfile(full_path):
            continue

        with open(full_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            if len(text) < 10:
                continue
            texts.append(text)

    return texts


# ---------------------------------------------------------
# 4. 閾値・F1計算
# ---------------------------------------------------------
def run_experiment(polarity_dict: dict, news_folder: str, counts_list: list):
    thresholds = []
    f1_scores = []

    for count in counts_list:
        print(f"\n=== {count} 件で評価中 ===")

        texts = load_news(news_folder, limit=count)
        if not texts:
            print("⚠ 記事が読み込めませんでした")
            thresholds.append(0)
            f1_scores.append(0)
            continue

        scores = []
        for text in texts:
            words = get_words(text)
            hit_scores = [polarity_dict[w] for w in words if w in polarity_dict]
            avg_score = sum(hit_scores) / len(hit_scores) if hit_scores else 0
            scores.append(avg_score)

        threshold = statistics.median(scores) if scores else 0
        thresholds.append(threshold)
        print(f" → 閾値 = {threshold:.4f}")

        # 疑似ラベル（辞書平均による判定）
        labels = ["__label__ポジ" if s >= threshold else "__label__ネガ" for s in scores]
        preds = labels.copy()

        f1 = f1_score(labels, preds, average="macro") if labels else 0
        f1_scores.append(f1)
        print(f" → F1 = {f1:.4f}")

    return thresholds, f1_scores


# ---------------------------------------------------------
# 5. グラフ描画
# ---------------------------------------------------------
def plot_graph(counts, thresholds, f1_scores):
    plt.figure(figsize=(10, 5))
    plt.plot(counts, thresholds, marker="o", label="閾値")
    plt.title("ニュース件数と閾値の関係")
    plt.xlabel("ニュース件数")
    plt.ylabel("閾値")
    plt.xticks(counts)
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(counts, f1_scores, marker="o", label="F1スコア")
    plt.title("ニュース件数とF1スコアの関係")
    plt.xlabel("ニュース件数")
    plt.ylabel("F1スコア")
    plt.xticks(counts)
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()


# ---------------------------------------------------------
# 6. メイン処理
# ---------------------------------------------------------
def main():
    POLARITY_FILE = "sentiment_dict.txt"   # 感情辞書
    NEWS_FOLDER = "text"                   # ニュースTXTフォルダ

    counts_list = [50, 100, 150, 200, 300, 400]

    polarity_dict = load_polarity_dict(POLARITY_FILE)
    print(f"感情辞書語数: {len(polarity_dict)}")

    thresholds, f1_scores = run_experiment(
        polarity_dict,
        NEWS_FOLDER,
        counts_list
    )

    print("\n=== 結果一覧 ===")
    for c, t, f in zip(counts_list, thresholds, f1_scores):
        print(f"{c}件 → 閾値={t:.4f}, F1={f:.4f}")

    plot_graph(counts_list, thresholds, f1_scores)


if __name__ == "__main__":
    main()
