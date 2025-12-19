import os
import glob
import fasttext
import MeCab
import statistics
from sklearn.metrics import classification_report

# ==============================
# 設定
# ==============================
NEWS_DIR = r"textf/news_data"
POLARITY_DICT = "sentiment_dict.txt"
TRAIN_FILE = "train.txt"
MODEL_FILE = "sentiment.bin"

# ==============================
# 辞書読み込み
# ==============================
def load_polarity_dict(path):
    dic = {}
    with open(path, encoding="utf-8") as f:
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
                dic[word] = score
            except:
                continue
    return dic


# ==============================
# MeCab (原形取得)
# ==============================
tagger = MeCab.Tagger()

def tokenize(text):
    node = tagger.parseToNode(text)
    words = []
    while node:
        features = node.feature.split(",")
        pos = features[0]
        if pos in ["名詞", "動詞", "形容詞"]:
            base = features[6]
            words.append(base if base != "*" else node.surface)
        node = node.next
    return words


# ==============================
# 記事を読み込み
# ==============================
def load_news():
    texts = []
    for f in sorted(glob.glob(os.path.join(NEWS_DIR, "*.txt"))):
        with open(f, encoding="utf-8") as fp:
            text = fp.read().replace("\n", " ")
            texts.append(text)
    return texts


# ==============================
# 擬似ラベル付け + train.txt 生成
# ==============================
def make_train_file(dic):
    texts = load_news()
    score_avgs = []
    items = []

    for t in texts:
        words = tokenize(t)
        scores = [dic[w] for w in words if w in dic]
        avg = sum(scores)/len(scores) if scores else 0
        score_avgs.append(avg)
        items.append((words, avg))

    # ⬅ 閾値（中央値）を自動決定
    th = statistics.median(score_avgs)
    print(f"\n★ 自動閾値（中央値） = {th:.5f}")

    pos = []
    neg = []

    for words, avg in items:
        label = "__label__ポジ" if avg >= th else "__label__ネガ"
        line = f"{label} {' '.join(words)}"
        if label == "__label__ポジ":
            pos.append(line)
        else:
            neg.append(line)

    print(f"ポジ：{len(pos)}  ネガ：{len(neg)}")

    # train.txt（偏り補正）
    size = min(len(pos), len(neg))
    pos = pos[:size]
    neg = neg[:size]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for line in pos + neg:
            f.write(line + "\n")

    print(f"train.txt 作成 → {len(pos)+len(neg)} 件")
    return size, th


# ==============================
# fastText モデル学習
# ==============================
def train_fasttext():
    model = fasttext.train_supervised(
        input=TRAIN_FILE,
        lr=0.5,
        epoch=10,
        wordNgrams=2,
        dim=100,
    )
    model.save_model(MODEL_FILE)
    return model


# ==============================
# モデル精度評価（擬似ラベルで検証）
# ==============================
def evaluate(model, dic, th):
    texts = load_news()
    y_true = []
    y_pred = []

    for t in texts:
        words = tokenize(t)
        scores = [dic[w] for w in words if w in dic]
        avg = sum(scores)/len(scores) if scores else 0

        t_label = "__label__ポジ" if avg >= th else "__label__ネガ"
        y_true.append(t_label)

        p_label = model.predict(" ".join(words))[0][0]
        y_pred.append(p_label)

    print("\n=== 精度レポート（擬似ラベル基準） ===")
    print(classification_report(y_true, y_pred))


# ==============================
# メイン処理
# ==============================
def main():
    print("=== 自動学習パイプライン開始 ===")

    dic = load_polarity_dict(POLARITY_DICT)

    size, th = make_train_file(dic)
    model = train_fasttext()
    evaluate(model, dic, th)

    print("\n=== 完了！ ===")


if __name__ == "__main__":
    main()
