import MeCab
from collections import defaultdict

# ------------------------------
# 1. MeCabで形態素解析
# ------------------------------
tagger = MeCab.Tagger()

def tokenize(text):
    words = []
    node = tagger.parseToNode(text)
    while node:
        features = node.feature.split(",")
        if len(features) >= 7:
            base = features[6]
            if base != "*" and base:
                words.append(base)
        node = node.next
    return words

# ------------------------------
# 2. 学習データから単語スコアを作成
# ------------------------------
pos_file = "text/pos/train_pos.txt"
neg_file = "text/neg/train_neg.txt"

pos_count = defaultdict(int)
neg_count = defaultdict(int)

# ポジ文章
with open(pos_file, encoding="utf-8") as f:
    for line in f:
        for w in tokenize(line.strip()):
            pos_count[w] += 1

# ネガ文章
with open(neg_file, encoding="utf-8") as f:
    for line in f:
        for w in tokenize(line.strip()):
            neg_count[w] += 1

# 単語スコア辞書作成
MIN_COUNT = 1
sentiment_dict = {}
for w in set(list(pos_count.keys()) + list(neg_count.keys())):
    p = pos_count[w]
    n = neg_count[w]
    total = p + n
    if total < MIN_COUNT:
        continue
    sentiment_dict[w] = (p - n) / total  # -1〜1

# ------------------------------
# 3. sentiment_dict.txt を読み込む
# ------------------------------
# すでに学習データから作った辞書
# sentiment_dict = {単語: スコア, ...}

# sentiment_dict.txt も読み込む
semantic_dict = {}
try:
    with open("text/sentiment_dict.txt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"無効な行をスキップ -> {line}")
                continue
            word, score = parts
            try:
                semantic_dict[word] = float(score)
            except ValueError:
                print(f"スコアが数値でない行をスキップ -> {line}")
except FileNotFoundError:
    print("警告: sentint_dict.txt が見つかりません")
# sentiment_dict に semantic_dict を統合（学習辞書優先）
for w, score in semantic_dict.items():
    if w not in sentiment_dict:
        sentiment_dict[w] = score

print("統合辞書語彙数:", len(sentiment_dict))

# ------------------------------
# 5. 文章判定関数（中立なし版）
# ------------------------------
def classify(text):
    words = tokenize(text)
    if not words:
        return 0, "中立"

    known_words = [w for w in words if w in sentiment_dict]
    if not known_words:
        return 0, "中立"

    score = sum(sentiment_dict[w] for w in known_words) / len(known_words)

    # 閾値調整（中立なし）
    if score < -0.04:
        label = "ネガ"
    else:
        label = "ポジ"

    return score, label


