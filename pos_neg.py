import MeCab
from collections import defaultdict

# ------------------------------
# 1. MeCab（原形取得）
# ------------------------------
tagger = MeCab.Tagger("-Ochasen")

def tokenize(text):
    words = []
    for line in tagger.parse(text).split("\n"):
        cols = line.split("\t")
        if len(cols) >= 3:
            base = cols[2]
            if base:
                words.append(base)
    return words


# ------------------------------
# 2. 単語ごとのポジ/ネガ出現回数を集計
# ------------------------------
pos_file = "train_pos.txt"
neg_file = "train_neg.txt"

pos_count = defaultdict(int)
neg_count = defaultdict(int)

# ポジ
with open(pos_file, encoding="utf-8") as f:
    for line in f:
        for w in tokenize(line.strip()):
            pos_count[w] += 1

# ネガ
with open(neg_file, encoding="utf-8") as f:
    for line in f:
        for w in tokenize(line.strip()):
            neg_count[w] += 1


# ------------------------------
# 3. -1〜1 に収まる安定単語スコア辞書を作成
# ------------------------------
sentiment_dict = {}

for w in set(list(pos_count.keys()) + list(neg_count.keys())):
    p = pos_count[w]
    n = neg_count[w]
    total = p + n
    if total == 0:
        continue

    # スコア：出現比率
    score = (p - n) / total     # -1〜1 に収まる
    sentiment_dict[w] = score

print("辞書語彙数:", len(sentiment_dict))


# ------------------------------
# 4. 判定関数
# ------------------------------
def classify(text):
    words = tokenize(text)
    if not words:
        return 0, "中立"

    total = sum(sentiment_dict.get(w, 0) for w in words)
    score = total / len(words)

    # 閾値の調整（中立を広げる）
    if score > -0.09:
        label = "ポジ"
    else:
        label = "ネガ"

    return score, label


# ------------------------------
# 5. テスト実行
# ------------------------------
while True:
    text = input("文章：")
    if not text:
        break

    score, label = classify(text)
    print(f"判定: {label}（スコア: {score:.3f}）")