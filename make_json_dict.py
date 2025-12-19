import json

INPUT_FILE = "sentiment_dict.txt"
OUTPUT_FILE = "dict.json"

score_dict = {}

with open(INPUT_FILE, encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split(":")
        if len(parts) != 4:
            continue
        word = parts[0]
        score = float(parts[3])
        score_dict[word] = score

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(score_dict, f, ensure_ascii=False, indent=2)

print(f"dict.json を生成しました（語彙数: {len(score_dict)}）")
