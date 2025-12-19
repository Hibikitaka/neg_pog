import os
import MeCab

# ------------------------------
# 1. 辞書読み込み
# ------------------------------
sentiment_dict = {}
with open("sentiment_dict.txt", encoding="utf-8") as f:
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
        sentiment_dict[word] = score

print(f"辞書読み込み完了: {len(sentiment_dict)} 単語")

# ------------------------------
# 2. MeCabで形態素解析（原形を取得）
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
# 3. 文章ごとの平均スコア計算（閾値決定用）
# ------------------------------
corpus_dir = r"C:\Users\C\pos_neg\text"
score_avgs = []
all_texts = []

for category in os.listdir(corpus_dir):
    category_path = os.path.join(corpus_dir, category)
    if not os.path.isdir(category_path):
        continue
    for filename in os.listdir(category_path):
        file_path = os.path.join(category_path, filename)
        with open(file_path, encoding="utf-8") as f:
            text = f.read().replace("\n", " ").replace("\u3000", " ").strip()
            if not text:
                continue
            words = get_words(text)
            hit_scores = [sentiment_dict[w]
                          for w in words if w in sentiment_dict]
            avg_score = sum(hit_scores)/len(hit_scores) if hit_scores else 0
            score_avgs.append(avg_score)
            all_texts.append((text, words, avg_score))

# 中央値を閾値として設定
score_avgs_sorted = sorted(score_avgs)
median_threshold = score_avgs_sorted[len(score_avgs_sorted)//2]
print(f"自動設定された閾値（中央値）: {median_threshold:.5f}")

# ------------------------------
# 4. ポジ・ネガ分類
# ------------------------------
pos_lines = []
neg_lines = []

for text, words, avg_score in all_texts:
    label = "__label__ポジ" if avg_score >= median_threshold else "__label__ネガ"
    line = f"{label} {' '.join(words)}"
    if label == "__label__ポジ":
        pos_lines.append(line)
    else:
        neg_lines.append(line)

# ------------------------------
# 5. ファイル出力（スコア付き）
# ------------------------------
with open("train_pos.txt", "w", encoding="utf-8") as f:
    for text, words, avg_score in all_texts:
        if avg_score >= median_threshold:
            line = f"__label__ポジ {avg_score:.5f} {' '.join(words)}"
            f.write(line + "\n")

with open("train_neg.txt", "w", encoding="utf-8") as f:
    for text, words, avg_score in all_texts:
        if avg_score < median_threshold:
            line = f"__label__ネガ {avg_score:.5f} {' '.join(words)}"
            f.write(line + "\n")

# 確認用出力
pos_count = sum(
    1 for _, _, avg_score in all_texts if avg_score >= median_threshold)
neg_count = sum(
    1 for _, _, avg_score in all_texts if avg_score < median_threshold)
print(f"学習データ作成完了: ポジ {pos_count} 件, ネガ {neg_count} 件")
print(f"使用した閾値: {median_threshold:.5f}")


