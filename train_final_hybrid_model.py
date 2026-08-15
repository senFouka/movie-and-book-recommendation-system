import numpy as np
import random as py_random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate, 
    Flatten, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
import os
import json

# روی ویندوز اگر خروجی به فایل/pipe هدایت شود، پایتون از cp1252 استفاده می‌کند
# و چاپ متن فارسی با UnicodeEncodeError کرش می‌کند. اجباراً UTF-8 می‌کنیم.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(domain):
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    # --- نام مدل نهایی ---
    model_file = os.path.join(data_dir, 'final_hybrid_model.keras')

    print(f"--- ۱. بارگذاری داده‌های پردازش شده از: {processed_file} ---")

    try:
        data = np.load(processed_file)
    except FileNotFoundError:
        print(f"فایل '{processed_file}' پیدا نشد.")
        return

    # --- بارگذاری آرایه‌های از پیش تقسیم‌شده (leave-one-out) ---
    X_user_train = data['X_user_train']
    X_item_train = data['X_item_train']
    X_time_train = data['X_time_train']
    y_train      = data['y_train']

    X_user_test  = data['X_user_test']
    X_item_test  = data['X_item_test']
    X_time_test  = data['X_time_test']
    y_test       = data['y_test']

    n_users = int(data['n_users'])
    n_items = int(data['n_items'])
    n_time_features = int(data['n_time_features'])
    MAX_SEQUENCE_LENGTH = X_item_train.shape[1]

    print(f"تعداد کاربران: {n_users}")
    print(f"تعداد آیتم‌ها: {n_items}")
    print(f"تعداد دسته‌های زمانی: {n_time_features}")

    print("\n--- ۲. تقسیم داده‌ها (leave-one-out) ---")
    X_train = [X_user_train, X_item_train, X_time_train]
    X_test  = [X_user_test,  X_item_test,  X_time_test]

    print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
    print(f"تعداد نمونه‌های تست (یک به‌ازای هر کاربر): {len(y_test)}")


    print("\n--- ۳. ساخت معماری نهایی (NCF + Dual-LSTM + Masked-Attention) ---")
    
    USER_EMBEDDING_SIZE = 64
    ITEM_EMBEDDING_SIZE = 64
    TIME_EMBEDDING_SIZE = 16
    LSTM_UNITS_ITEM = 64
    LSTM_UNITS_TIME = 16

    # --- تعریف ۳ ورودی ---
    user_input = Input(shape=(1,), name='UserInput')
    item_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='ItemSequenceInput')
    time_seq_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='TimeSequenceInput')

    # --- شاخه ۱: NCF (سلیقه کلی کاربر) ---
    user_embedding_layer = Embedding(input_dim=n_users, output_dim=USER_EMBEDDING_SIZE, name='UserEmbedding')
    user_vec = Flatten(name='FlattenUser')(user_embedding_layer(user_input))

    # --- شاخه ۲: رفتار ترتیبی (LSTM + Time + Attention) ---
    # بخش آیتم
    item_embedding_layer = Embedding(input_dim=n_items, output_dim=ITEM_EMBEDDING_SIZE, name='ItemEmbedding', mask_zero=True)
    item_seq_embedding = item_embedding_layer(item_seq_input)
    lstm_item_out = LSTM(LSTM_UNITS_ITEM, return_sequences=True, name='LSTM_Item')(item_seq_embedding)

    # بخش زمان
    time_embedding_layer = Embedding(input_dim=n_time_features, output_dim=TIME_EMBEDDING_SIZE, name='TimeEmbedding', mask_zero=True)
    time_seq_embedding = time_embedding_layer(time_seq_input)
    lstm_time_out = LSTM(LSTM_UNITS_TIME, return_sequences=True, name='LSTM_Time')(time_seq_embedding)

    # ادغام خروجی‌های LSTM (ماسک‌ها حفظ می‌شوند)
    combined_lstm_out = Concatenate(axis=2, name='CombineLSTMs')([lstm_item_out, lstm_time_out])

    # اعمال Attention روی داده‌های تمیز
    attention_out = Attention(name='Attention')([combined_lstm_out, combined_lstm_out])
    # خلاصه کردن خروجی Attention به یک بردار
    context_vec = GlobalAveragePooling1D(name='AttentionPooling')(attention_out)
    
    # --- ادغام نهایی (کلیدی‌ترین بخش) ---
    # ترکیب سلیقه کلی (NCF) + رفتار اخیر (LSTM+Attention)
    final_combined_vec = Concatenate(name='Combine_NCF_LSTM')([user_vec, context_vec])

    # --- شبکه MLP نهایی ---
    dense_1 = Dense(128, activation='relu', name='Dense_1')(final_combined_vec)
    dense_2 = Dense(64, activation='relu', name='Dense_2')(dense_1)
    output = Dense(n_items, activation='softmax', name='Output')(dense_2)

    # --- ساخت مدل نهایی ---
    model = Model(
        inputs=[user_input, item_seq_input, time_seq_input], 
        outputs=output,
        name='Final_Hybrid_Model'
    )
    model.summary()

    print("\n--- ۴. کامپایل و آموزش کامل مدل ---")
    metrics = ['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc')]
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=metrics)
    early_stopper = EarlyStopping(monitor='val_top_10_acc', patience=3, verbose=1, mode='max', restore_best_weights=True)

    print(f"شروع آموزش مدل نهایی {domain}...")
    history = model.fit(X_train, y_train, batch_size=256, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopper])

    print("\n--- ۵. ذخیره تاریخچه آموزش و رسم نمودارها ---")
    history_keys = list(history.history.keys())
    print(f"کلیدهای history.history: {history_keys}")

    history_file = os.path.join(data_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    print(f"تاریخچه آموزش در '{history_file}' ذخیره شد.")

    epochs_range = range(1, len(history.history['loss']) + 1)
    COLOR_TRAIN = '#1B6CA8'
    COLOR_VAL   = '#E76F51'

    # Plot A — Loss convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, history.history['loss'], color=COLOR_TRAIN, linewidth=2, label='Training Loss')
    ax.plot(epochs_range, history.history['val_loss'], color=COLOR_VAL, linewidth=2, label='Validation Loss')
    ax.set_title('Model Convergence (Loss)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    loss_fig_path = os.path.join(data_dir, 'fig_loss.png')
    fig.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"نمودار Loss در '{loss_fig_path}' ذخیره شد.")

    # Resolve the real top-10 accuracy key (name='top_10_acc' → key 'top_10_acc')
    top10_train_key = next((k for k in history_keys if 'top_10' in k and not k.startswith('val_')), None)
    top10_val_key   = next((k for k in history_keys if 'top_10' in k and k.startswith('val_')), None)

    if top10_train_key and top10_val_key:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, history.history[top10_train_key], color=COLOR_TRAIN, linewidth=2, label='Training HR@10')
        ax.plot(epochs_range, history.history[top10_val_key],   color=COLOR_VAL,   linewidth=2, label='Validation HR@10')
        ax.set_title('Top-10 Accuracy over Epochs', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('HR@10 (Top-10 Accuracy)', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        acc_fig_path = os.path.join(data_dir, 'fig_top10_acc.png')
        fig.savefig(acc_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"نمودار Top-10 Accuracy در '{acc_fig_path}' ذخیره شد.")
    else:
        print(f"هشدار: کلید top_10_acc در history پیدا نشد. کلیدهای موجود: {history_keys}")

    print("\n--- ۶. ارزیابی نهایی مدل (Full-Vocab Keras Metrics) ---")
    results = model.evaluate(X_test, y_test)
    print(f"✅ مدل {domain} (Final Hybrid) با موفقیت آموزش دید.")
    print(f"Loss (خطا) روی داده‌های تست: {results[0]:.4f}")
    print(f"Accuracy (دقت) روی داده‌های تست: {results[1] * 100:.2f}%")
    print(f"Top 10 Accuracy (full-vocab) روی داده‌های تست: {results[2] * 100:.2f}%")

    model.save(model_file)
    print(f"مدل نهایی در '{model_file}' ذخیره شد.")

    print("\n--- ۷. ارزیابی NCF-Protocol: HR@10 و NDCG@10 (1 مثبت در برابر 99 منفی) ---")
    NUM_NEG = 99
    EVAL_CHUNK = 512
    rng = np.random.default_rng(seed=42)
    hits = 0
    ndcg_sum = 0.0
    n_test = len(y_test)

    print(f"پیش‌بینی دسته‌ای برای {n_test} کاربر تست...")
    for start in range(0, n_test, EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, n_test)
        preds = model.predict(
            [X_user_test[start:end], X_item_test[start:end], X_time_test[start:end]],
            batch_size=256, verbose=0
        )
        for local_i in range(end - start):
            target = int(y_test[start + local_i])
            pred_vec = preds[local_i]

            # نمونه‌برداری 99 آیتم منفی (بدون جایگزینی، با حذف آیتم هدف و پدینگ)
            neg_pool = np.arange(1, n_items)
            neg_pool = neg_pool[neg_pool != target]
            negatives = rng.choice(neg_pool, size=NUM_NEG, replace=False)

            candidates = np.concatenate([[target], negatives])
            scores = pred_vec[candidates]

            order = np.argsort(scores)[::-1]
            ranked = candidates[order]
            top10 = ranked[:10]

            if target in top10:
                hits += 1
                rank = int(np.where(top10 == target)[0][0]) + 1  # 1-indexed
                ndcg_sum += 1.0 / np.log2(rank + 1)

        if (start // EVAL_CHUNK) % 10 == 0:
            print(f"  پیشرفت: {end}/{n_test}")

    hr10   = hits / n_test
    ndcg10 = ndcg_sum / n_test

    print("\n" + "=" * 55)
    print(f"  دامنه: {domain.upper()}")
    print(f"  HR@10  (NCF Protocol, 1 vs {NUM_NEG}): {hr10  * 100:.2f}%")
    print(f"  NDCG@10(NCF Protocol, 1 vs {NUM_NEG}): {ndcg10 * 100:.2f}%")
    print("=" * 55)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python train_final_hybrid_model.py movie")
        print("   یا: python train_final_hybrid_model.py book")
    else:
        domain = sys.argv[1]
        main(domain)