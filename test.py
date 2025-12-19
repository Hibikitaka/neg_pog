words_in_dict = [w for w in words if w in sentiment_dict]
print("辞書に存在する単語数:", len(words_in_dict), "総単語数:", len(words))
