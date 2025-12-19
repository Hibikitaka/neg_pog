import os
import MeCab
import matplotlib.pyplot as plt

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
# 3. 平均スコア収集
# ------------------------------
corpus_dir = r"C:\Users\C\pos_neg\text"
score_avgs = []

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
            hit_scores = [sentiment_dict[w] for w in words if w in sentiment_dict]
            if hit_scores:
                avg = sum(hit_scores) / len(hit_scores)
                score_avgs.append(avg)
            else:
                # 辞書にヒットする単語なしは0として扱う
                score_avgs.append(0)

# ------------------------------
# 4. ヒストグラム表示
# ------------------------------
plt.figure(figsize=(10,5))
plt.hist(score_avgs, bins=50, color='skyblue', edgecolor='black')
plt.title("文章ごとの平均スコア分布")
plt.xlabel("平均スコア")
plt.ylabel("文章数")
plt.grid(axis='y')
plt.show()

# ------------------------------
# 5. 閾値候補の自動計算（中央値）
# ------------------------------
score_avgs_sorted = sorted(score_avgs)
median_score = score_avgs_sorted[len(score_avgs_sorted)//2]
print(f"推奨閾値（中央値）: {median_score:.5f}")
