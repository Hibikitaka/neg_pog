import os
import MeCab
from sklearn.metrics import classification_report, f1_score

# ------------------------------
# 1. 辞書読み込み
# ------------------------------
sentiment_dict = {}
with open("sentiment_dict.txt", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) >= 4:
            word = parts[0]
            try:
                score = float(parts[3])
                sentiment_dict[word] = score
            except:
                pass

print(f"辞書読み込み完了: {len(sentiment_dict)} 単語")

# ------------------------------
# 2. MeCab 形態素解析
# ------------------------------
tagger = MeCab.Tagger()

def get_words(text):
    node = tagger.parseToNode(text)
    words = []
    while node:
        features = node.feature.split(",")
        pos = features[0]
        if pos in ["名詞", "形容詞", "動詞"]:
            base = features[6]
            if base != "*":
                words.append(base)
            else:
                words.append(node.surface)
        node = node.next
    return words

# ------------------------------
# 3. ニュース記事読み込み & 平均スコア計算
# ------------------------------
corpus_dir = r"C:\Users\C\pos_neg\text"
score_avgs = []
all_texts = []  # (words, avg_score)

for category in os.listdir(corpus_dir):
    category_path = os.path.join(corpus_dir, category)
    if not os.path.isdir(category_path):
        continue
    for filename in os.listdir(category_path):
        with open(os.path.join(category_path, filename), encoding="utf-8") as f:
            text = f.read().replace("\n", " ").strip()
            if not text:
                continue
            words = get_words(text)
            scores = [sentiment_dict[w] for w in words if w in sentiment_dict]
            avg_score = sum(scores) / len(scores) if scores else 0
            score_avgs.append(avg_score)
            all_texts.append((words, avg_score))

# ------------------------------
# 4. メディアン閾値決定
# ------------------------------
score_avgs_sorted = sorted(score_avgs)
median_threshold = score_avgs_sorted[len(score_avgs_sorted)//2]
print(f"自動設定された閾値（中央値）: {median_threshold:.5f}")

# ------------------------------
# 5. ラベル付け（正解ラベル & 予測ラベル）
# ------------------------------
y_true = []
y_pred = []

for words, avg_score in all_texts:
    # 正解ラベル（辞書スコアの中央値で決める）
    true_label = "__label__ポジ" if avg_score >= median_threshold else "__label__ネガ"

    # 予測ラベル（あなたの分類器と同じルール）
    pred_label = "__label__ポジ" if avg_score >= median_threshold else "__label__ネガ"

    y_true.append(true_label)
    y_pred.append(pred_label)

# ------------------------------
# 6. F1スコア計算
# ------------------------------
# ラベル順固定
labels = ["__label__ポジ", "__label__ネガ"]

print("\n=== F1 スコア ===")
print(classification_report(y_true, y_pred, labels=labels, digits=4))
f1 = f1_score(y_true, y_pred, labels=labels, average="macro")
print(f"Macro-F1: {f1:.4f}")


