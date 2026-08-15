"""
ارزیابی رتبه‌بندی به پروتکل NCF (He et al., 2017):
هر کاربر تست = ۱ آیتم مثبت (آخرین تعامل، leave-one-out) در برابر ۹۹ آیتم منفی نمونه‌برداری‌شده.

    HR@10   = نسبت کاربرانی که آیتم واقعی‌شان در ۱۰ رتبهٔ برتر آمده
    NDCG@10 = میانگین 1/log2(rank+1) برای همان کاربران (و ۰ برای بقیه)

این اسکریپت مستقل از آموزش اجرا می‌شود و مدل ذخیره‌شده را بارگذاری می‌کند،
بنابراین برای گرفتن متریک‌ها نیازی به آموزش مجدد نیست.

    python evaluate_ranking.py movie
    python evaluate_ranking.py book
"""
import numpy as np
import os
import sys
import json

# روی ویندوز اگر خروجی به فایل/pipe هدایت شود، پایتون از cp1252 استفاده می‌کند
# و چاپ متن فارسی با UnicodeEncodeError کرش می‌کند. اجباراً UTF-8 می‌کنیم.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

from tensorflow.keras.models import load_model

NUM_NEG = 99
K = 10
CHUNK = 1024
SEED = 42


def evaluate(domain):
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    model_file = os.path.join(data_dir, 'final_hybrid_model.keras')

    for path in (processed_file, model_file):
        if not os.path.exists(path):
            print(f"خطا: فایل '{path}' پیدا نشد.")
            return

    print(f"--- بارگذاری دادهٔ تست: {processed_file} ---")
    data = np.load(processed_file)
    X_user_test = data['X_user_test']
    X_item_test = data['X_item_test']
    X_time_test = data['X_time_test']
    y_test = data['y_test']
    n_items = int(data['n_items'])
    n_test = len(y_test)

    print(f"تعداد کاربران تست (leave-one-out): {n_test:,}")
    print(f"تعداد آیتم‌ها (با پدینگ): {n_items:,}")

    print(f"--- بارگذاری مدل: {model_file} ---")
    model = load_model(model_file)

    rng = np.random.default_rng(SEED)
    ranks = np.empty(n_test, dtype=np.int32)

    print(f"\n--- ارزیابی: ۱ مثبت در برابر {NUM_NEG} منفی ---")
    for start in range(0, n_test, CHUNK):
        end = min(start + CHUNK, n_test)
        B = end - start

        preds = model.predict(
            [X_user_test[start:end], X_item_test[start:end], X_time_test[start:end]],
            batch_size=256, verbose=0
        )

        targets = y_test[start:end].astype(np.int64)
        rows = np.arange(B)

        # نمونه‌برداری ۹۹ منفی یکتا در هر سطر، با حذف توکن پدینگ (۰) و خود آیتم هدف.
        # کلید تصادفی می‌دهیم و ۹۹ کوچک‌ترین را برمی‌داریم => نمونه‌برداری بدون جایگزینی.
        keys = rng.random((B, n_items), dtype=np.float32)
        keys[:, 0] = np.inf
        keys[rows, targets] = np.inf
        neg_idx = np.argpartition(keys, NUM_NEG, axis=1)[:, :NUM_NEG]

        target_scores = preds[rows, targets][:, None]
        neg_scores = np.take_along_axis(preds, neg_idx, axis=1)

        # رتبهٔ آیتم هدف = ۱ + تعداد منفی‌هایی که امتیاز بالاتری گرفته‌اند
        ranks[start:end] = 1 + (neg_scores > target_scores).sum(axis=1)

        print(f"  پیشرفت: {end:,}/{n_test:,}")

    hit = ranks <= K
    hr = float(hit.mean())
    ndcg = float(np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean())

    print("\n" + "=" * 58)
    print(f"  دامنه: {domain.upper()}   (پروتکل NCF: ۱ مثبت در برابر {NUM_NEG} منفی)")
    print(f"  کاربران ارزیابی‌شده : {n_test:,}")
    print(f"  HR@{K}   : {hr * 100:.2f}%")
    print(f"  NDCG@{K} : {ndcg * 100:.2f}%")
    print(f"  میانگین رتبهٔ آیتم هدف : {ranks.mean():.2f}  (از ۱۰۰)")
    print("=" * 58)

    out = os.path.join(data_dir, 'ranking_metrics.json')
    with open(out, 'w') as f:
        json.dump({
            'domain': domain,
            'protocol': f'leave-one-out, 1 positive vs {NUM_NEG} sampled negatives',
            'n_test_users': int(n_test),
            'seed': SEED,
            f'HR@{K}': hr,
            f'NDCG@{K}': ndcg,
            'mean_rank': float(ranks.mean()),
        }, f, indent=2)
    print(f"متریک‌ها در '{out}' ذخیره شد.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("مثال: python evaluate_ranking.py movie")
        print("   یا: python evaluate_ranking.py book")
    else:
        evaluate(sys.argv[1])
