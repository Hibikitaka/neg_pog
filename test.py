import pickle

# 1. テキストファイルから感情辞書を読み込む
# 辞書の形式: { "単語": (スコア, 品詞) }
sentiment_dict = {}
with open("sentiment_dict.txt", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue  # 空行はスキップ
        parts = line.split(":")
        if len(parts) != 4:
            print(f"スキップ: 形式が不正な行 -> {line}")
            continue
        word, reading, pos, score = parts
        try:
            sentiment_dict[word] = (float(score), pos)
        except ValueError:
            print(f"スキップ: スコア変換できない行 -> {line}")

# 2. pickle で保存
with open("sentiment_dict.pkl", "wb") as f:
    pickle.dump(sentiment_dict, f)

print(f"{len(sentiment_dict)} 件の単語を pickle に保存しました。")
