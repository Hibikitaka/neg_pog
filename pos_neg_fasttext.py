# pos_neg_fasttext.py
import os
import fasttext

# =========================
# 設定
# =========================
corpus_dir = "neg_pog" \
    "/text"  # livedoor ニュースコーパスの展開先
train_file = "train.txt"
model_file = "sentiment.bin"

# =========================
# ポジ・ネガ分類の簡易ルール
# =========================
# 以下の辞書に含まれるキーワードでラベルを自動付与
pos_keywords = ["良い", "好き", "おすすめ", "最高", "楽しい", "便利", "満足"]
neg_keywords = ["悪い", "嫌い", "ひどい", "最悪", "不便", "不満", "問題"]

# =========================
# コーパス読み込み
# =========================
data = []
for category in os.listdir(corpus_dir):
    cat_path = os.path.join(corpus_dir, category)
    if not os.path.isdir(cat_path):
        continue
    for filename in os.listdir(cat_path):
        file_path = os.path.join(cat_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().replace("\n", " ")
            label = None
            for kw in pos_keywords:
                if kw in text:
                    label = "__label__pos"
                    break
            for kw in neg_keywords:
                if kw in text:
                    label = "__label__neg"
                    break
            if label:
                data.append(f"{label} {text}")

# =========================
# 学習データ保存
# =========================
with open(train_file, "w", encoding="utf-8") as f:
    for line in data:
        f.write(line + "\n")

print(f"学習データ件数: {len(data)}")
print(f"train.txt を作成しました。")

# =========================
# fastText 学習
# =========================
model = fasttext.train_supervised(
    input=train_file,
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    verbose=2,
    minCount=1
)

model.save_model(model_file)
print(f"モデルを {model_file} に保存しました。")

# =========================
# 簡単な判定テスト
# =========================
while True:
    text = input("判定したい文章を入力してください（Enterだけで終了）：")
    if not text:
        break
    label, prob = model.predict(text)
    print(f"判定: {label[0]}    確率: {prob[0]:.3f}")
