import fasttext
import os

def main():
    TRAIN_FILE = "train.txt"
    MODEL_FILE = "sentiment.bin"

    # 学習元ファイル確認
    if not os.path.exists(TRAIN_FILE):
        print(f"[エラー] {TRAIN_FILE} が見つかりません。先に学習データを生成してください。")
        return

    print("=== fastText モデル学習を開始します ===")

    # モデル学習
    model = fasttext.train_supervised(
        input=TRAIN_FILE,
        lr=0.5,          # 学習率（基本これでOK）
        epoch=25,        # 学習エポック数
        wordNgrams=2,    # N-gram（精度アップ）
        dim=100,         # ベクトル次元（デフォルト100）
        loss="softmax"   # 損失関数
    )

    # モデル保存
    model.save_model(MODEL_FILE)

    print("=== 学習完了！ ===")
    print(f"モデルを保存しました → {MODEL_FILE}")

    # 手動テスト（任意）
    while True:
        text = input("\n判定したい文章を入力してください（Enterだけで終了）： ")
        if text.strip() == "":
            break

        label, prob = model.predict(text)
        print(f"判定: {label[0]}    確率: {prob[0]:.3f}")


if __name__ == "__main__":
    main()
