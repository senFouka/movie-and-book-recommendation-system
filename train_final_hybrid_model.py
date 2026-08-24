import numpy as np
import random as py_random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, LSTM, Dense, Concatenate,
    Flatten, Attention, GlobalAveragePooling1D, Reshape
)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
import os
import json
from datetime import datetime

# روی ویندوز اگر خروجی به فایل/pipe هدایت شود، پایتون از cp1252 استفاده می‌کند
# و چاپ متن فارسی با UnicodeEncodeError کرش می‌کند. اجباراً UTF-8 می‌کنیم.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===========================================================================
# کلید شاخه محتوا (Content branch)
# ---------------------------------------------------------------------------
# True  → شاخه سوم (محتوای آیتم) به مدل اضافه می‌شود و خروجی‌ها با پسوند
#         «_content» ذخیره می‌شوند تا مدل پایه بازنویسی نشود.
# False → دقیقاً همان معماری و همان نام فایل‌های قبلی (مدل پایه).
# سایر ابرپارامترها، تقسیم leave-two-out، EarlyStopping و ارزیابی ۱ در برابر ۹۹
# در هر دو حالت یکسان هستند.
# ===========================================================================
USE_CONTENT = True

CONTENT_PROJECTION_SIZE = 16   # movie: ابعاد Dense روی بردار multi-hot ژانر
AUTHOR_EMBEDDING_SIZE = 16     # book: ابعاد Embedding نویسنده
CONTENT_LSTM_UNITS = 16        # انکودر سبک دنباله محتوا (هر دو دامنه)

SEED = 42

_TRUTHY = {'1', 'true', 'yes', 'on'}
_FALSY = {'0', 'false', 'no', 'off'}

# نوع ویژگی محتوایی از روی کلید موجود در فایل داده تشخیص داده می‌شود،
# نه از روی نام دامنه. ترتیب = اولویت تشخیص.
CONTENT_FEATURE_KEYS = ('item_genre_matrix', 'item_author_ids')
_KIND_BY_KEY = {'item_genre_matrix': 'genre', 'item_author_ids': 'author'}
# لایه‌ای که خروجی «جست‌وجوی خام محتوا» را می‌دهد (برای بررسی هم‌ترازی)
_LOOKUP_LAYER_BY_KIND = {'genre': 'GenreLookup', 'author': 'SqueezeAuthorIDs'}


def detect_content_kind(data):
    """نوع ویژگی محتوایی موجود در فایل داده را برمی‌گرداند ('genre' | 'author' | None)."""
    for key in CONTENT_FEATURE_KEYS:
        if key in data.files:
            return _KIND_BY_KEY[key]
    return None


def resolve_use_content(override=None):
    """
    تعیین وضعیت شاخه محتوا بدون ویرایش سورس، تا هر دو بازو با فایل یکسان اجرا شوند.
    اولویت: آرگومان خط فرمان (--content=on/off) > متغیر محیطی USE_CONTENT > ثابت ماژول.
    """
    if override is not None:
        value = str(override).strip().lower()
        source = 'آرگومان خط فرمان'
    elif os.environ.get('USE_CONTENT') is not None:
        value = os.environ['USE_CONTENT'].strip().lower()
        source = 'متغیر محیطی USE_CONTENT'
    else:
        print(f"وضعیت شاخه محتوا از ثابت ماژول USE_CONTENT خوانده شد: {USE_CONTENT}")
        return bool(USE_CONTENT)

    if value in _TRUTHY:
        resolved = True
    elif value in _FALSY:
        resolved = False
    else:
        raise ValueError(
            f"مقدار نامعتبر برای شاخه محتوا: '{override if override is not None else os.environ.get('USE_CONTENT')}'. "
            f"مقادیر مجاز: {sorted(_TRUTHY)} یا {sorted(_FALSY)}"
        )
    print(f"وضعیت شاخه محتوا از {source} خوانده شد: {resolved}")
    return resolved


def build_content_branch(domain, data, item_seq_input, n_items, max_sequence_length):
    """
    شاخه محتوا: برای هر آیتمِ دنباله ورودی، ویژگی محتوایی‌اش جست‌وجو می‌شود و
    سپس کل دنباله به یک بردار واحد فشرده می‌شود.

    جست‌وجو داخل خود گراف انجام می‌شود (لایه Embedding با وزن ثابت و
    trainable=False). به این ترتیب ورودی‌های مدل همان ۳ ورودی قبلی می‌مانند و
    اسکریپت‌های ارزیابی/استنتاج موجود بدون تغییر کار می‌کنند.

    نوع ویژگی از روی کلیدهای موجود در خود فایل داده تشخیص داده می‌شود، نه از روی
    نام دامنه؛ بنابراین افزودن دامنه جدید نیازی به تغییر این تابع ندارد.

    خروجی: (content_vec | None, توضیح متنی)
    """
    kind = detect_content_kind(data)
    if kind is None:
        print(f"هشدار: هیچ کلید محتوایی برای دامنه '{domain}' در فایل داده نیست "
              f"(انتظار یکی از: {', '.join(CONTENT_FEATURE_KEYS)}).")
        print(f"ابتدا «python build_dataset.py {domain} --content-only» را اجرا کنید.")
        print("شاخه محتوا غیرفعال شد.")
        return None, None

    if kind == 'genre':
        item_genre_matrix = np.asarray(data['item_genre_matrix'], dtype='float32')
        n_genres = int(data['n_genres'])
        assert item_genre_matrix.shape == (n_items, n_genres), \
            f"شکل ماتریس ژانر {item_genre_matrix.shape} با اندیس آیتم ({n_items}, {n_genres}) هم‌تراز نیست!"
        assert not item_genre_matrix[0].any(), "سطر پدینگ ماتریس ژانر باید تماماً صفر باشد!"
        print(f"شاخه محتوا [{domain}]: بردار multi-hot ژانر با {n_genres} بعد "
              f"→ Dense({CONTENT_PROJECTION_SIZE}) → LSTM({CONTENT_LSTM_UNITS})")

        # جست‌وجوی ژانر: آیتم → بردار multi-hot (وزن ثابت، آموزش‌ناپذیر)
        genre_seq = Embedding(
            input_dim=n_items, output_dim=n_genres,
            embeddings_initializer=Constant(item_genre_matrix),
            trainable=False, mask_zero=True, name='GenreLookup'
        )(item_seq_input)

        # نمایش محتوایی هر آیتم (ماسک از لایه Embedding عبور می‌کند)
        content_seq = Dense(
            CONTENT_PROJECTION_SIZE, activation='relu', name='GenreProjection'
        )(genre_seq)

        content_vec = LSTM(CONTENT_LSTM_UNITS, name='LSTM_Content')(content_seq)
        return content_vec, f"genre multi-hot ({n_genres}) → Dense({CONTENT_PROJECTION_SIZE}) → LSTM({CONTENT_LSTM_UNITS})"

    if kind == 'author':
        item_author_ids = np.asarray(data['item_author_ids'])
        n_authors = int(data['n_authors'])
        assert item_author_ids.shape == (n_items,), \
            f"شکل جدول نویسنده {item_author_ids.shape} با اندیس آیتم ({n_items},) هم‌تراز نیست!"
        assert item_author_ids[0] == 0, "خانه پدینگ جدول نویسنده باید ۰ باشد!"
        assert item_author_ids.max() < n_authors, "شناسه نویسنده خارج از محدوده واژگان!"
        print(f"شاخه محتوا [{domain}]: شناسه نویسنده ({n_authors} واژه) "
              f"→ Embedding({AUTHOR_EMBEDDING_SIZE}, mask_zero=True) → LSTM({CONTENT_LSTM_UNITS})")

        # جست‌وجوی نویسنده: آیتم → شناسه نویسنده (وزن ثابت، آموزش‌ناپذیر).
        # خروجی float است اما لایه Embedding بعدی خودش به int32 تبدیل می‌کند و
        # شناسه‌ها بسیار کوچک‌تر از ۲^۲۴ هستند، پس تبدیل دقیق است.
        author_id_lookup = Embedding(
            input_dim=n_items, output_dim=1,
            embeddings_initializer=Constant(item_author_ids.reshape(-1, 1).astype('float32')),
            trainable=False, mask_zero=False, name='ItemToAuthorLookup'
        )(item_seq_input)
        author_id_seq = Reshape((max_sequence_length,), name='SqueezeAuthorIDs')(author_id_lookup)

        # شناسه ۰ = پدینگ/ناشناخته → mask_zero=True دقیقاً همان پدینگ آیتم را ماسک می‌کند
        author_seq_embedding = Embedding(
            input_dim=n_authors, output_dim=AUTHOR_EMBEDDING_SIZE,
            mask_zero=True, name='AuthorEmbedding'
        )(author_id_seq)

        content_vec = LSTM(CONTENT_LSTM_UNITS, name='LSTM_Content')(author_seq_embedding)
        return content_vec, f"author id ({n_authors}) → Embedding({AUTHOR_EMBEDDING_SIZE}) → LSTM({CONTENT_LSTM_UNITS})"

    raise RuntimeError(f"نوع محتوای ناشناخته: {kind}")


def verify_content_alignment(domain, data, model, item_seq_input, X_item_sample):
    """
    تأیید نهایی هم‌ترازی: خروجی جست‌وجوی داخل گراف با جست‌وجوی مستقیم numpy
    روی همان دنباله‌های واقعی مقایسه می‌شود.
    """
    print("\n--- ۳.۲ بررسی هم‌ترازی شاخه محتوا با اندیس آیتم (forward pass) ---")
    kind = detect_content_kind(data)
    lookup_layer = _LOOKUP_LAYER_BY_KIND[kind]
    if kind == 'genre':
        table = np.asarray(data['item_genre_matrix'], dtype='float32')
        vocab = data['genre_vocab']
    else:
        table = np.asarray(data['item_author_ids'])
        vocab = data['author_vocab']

    probe_model = Model(inputs=item_seq_input, outputs=model.get_layer(lookup_layer).output)
    from_graph = probe_model.predict(X_item_sample, verbose=0)
    from_numpy = table[X_item_sample]

    max_diff = float(np.abs(from_graph - from_numpy).max())
    print(f"بیشینه اختلاف گراف در برابر numpy روی {len(X_item_sample)} دنباله: {max_diff:g}")
    assert max_diff == 0.0, "جست‌وجوی محتوا در گراف با جدول اصلی هم‌تراز نیست!"
    print("[check] جست‌وجوی داخل گراف دقیقاً با item_genre_matrix/item_author_ids یکی است: OK")

    # نمایش چند آیتم واقعی؛ ترجیحاً دنباله‌ای که پدینگ هم داشته باشد
    padded_rows = np.nonzero((X_item_sample == 0).any(axis=1))[0]
    row = int(padded_rows[0]) if len(padded_rows) else 0
    seq = X_item_sample[row]
    print(f"نمونه دنباله (آخرین ۵ آیتم غیرصفر) از X_item_test[{row}]:")
    non_pad = [int(i) for i in seq if i != 0][-5:]
    for item_id in non_pad:
        if kind == 'genre':
            names = [vocab[j] for j in np.nonzero(table[item_id])[0]]
            print(f"  item={item_id:<6} → ژانرها: {'|'.join(names)}")
        else:
            print(f"  item={item_id:<6} → author_id={table[item_id]} → '{vocab[table[item_id]]}'")
    n_pad = int((seq == 0).sum())
    print(f"موقعیت‌های پدینگ در این دنباله: {n_pad} (ویژگی محتوایی صفر و ماسک‌شده)")


def main(domain, summary_only=False, content_override=None):
    # تکرارپذیری: بذر یکسان برای python/numpy/tensorflow
    py_random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    print(f"بذر تصادفی (seed) روی {SEED} تنظیم شد.")

    use_content = resolve_use_content(content_override)

    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    # --- نام مدل نهایی (نسخه محتوایی جدا ذخیره می‌شود تا مدل پایه حفظ شود) ---
    variant_suffix = '_content' if use_content else ''
    model_file = os.path.join(data_dir, f'final_hybrid{variant_suffix}_model.keras')

    print(f"--- ۱. بارگذاری داده‌های پردازش شده از: {processed_file} ---")

    try:
        data = np.load(processed_file)
    except FileNotFoundError:
        print(f"فایل '{processed_file}' پیدا نشد.")
        return

    # --- بارگذاری آرایه‌های از پیش تقسیم‌شده (leave-TWO-out) ---
    missing = [k for k in ('X_user_val', 'X_item_val', 'X_time_val', 'y_val') if k not in data.files]
    if missing:
        print(f"خطا: کلیدهای اعتبارسنجی در فایل داده نیستند: {missing}")
        print("لطفاً ابتدا build_dataset.py را دوباره اجرا کنید.")
        return

    X_user_train = data['X_user_train']
    X_item_train = data['X_item_train']
    X_time_train = data['X_time_train']
    y_train      = data['y_train']

    X_user_val   = data['X_user_val']
    X_item_val   = data['X_item_val']
    X_time_val   = data['X_time_val']
    y_val        = data['y_val']

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

    print("\n--- ۲. تقسیم داده‌ها (leave-TWO-out) ---")
    X_train = [X_user_train, X_item_train, X_time_train]
    X_val   = [X_user_val,   X_item_val,   X_time_val]
    X_test  = [X_user_test,  X_item_test,  X_time_test]

    print(f"تعداد نمونه‌های آموزشی: {len(y_train)}")
    print(f"تعداد نمونه‌های اعتبارسنجی (یک به‌ازای هر کاربر): {len(y_val)}")
    print(f"تعداد نمونه‌های تست (یک به‌ازای هر کاربر): {len(y_test)}")
    print("مجموعه تست فقط برای ارزیابی نهایی استفاده می‌شود و به fit() داده نمی‌شود.")


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

    # --- شاخه ۳: محتوای آیتم (ژانر برای فیلم / نویسنده برای کتاب) ---
    content_vec, content_desc = (None, None)
    if use_content:
        content_vec, content_desc = build_content_branch(
            domain, data, item_seq_input, n_items, MAX_SEQUENCE_LENGTH
        )
    else:
        print("شاخه محتوا غیرفعال است (USE_CONTENT=False) — مدل پایه.")

    # --- ادغام نهایی (کلیدی‌ترین بخش) ---
    # ترکیب سلیقه کلی (NCF) + رفتار اخیر (LSTM+Attention) [+ محتوای آیتم]
    branches = [user_vec, context_vec]
    if content_vec is not None:
        branches.append(content_vec)
    final_combined_vec = Concatenate(name='Combine_NCF_LSTM')(branches)

    # --- شبکه MLP نهایی ---
    dense_1 = Dense(128, activation='relu', name='Dense_1')(final_combined_vec)
    dense_2 = Dense(64, activation='relu', name='Dense_2')(dense_1)
    output = Dense(n_items, activation='softmax', name='Output')(dense_2)

    # --- ساخت مدل نهایی ---
    model = Model(
        inputs=[user_input, item_seq_input, time_seq_input],
        outputs=output,
        name='Final_Hybrid_Content_Model' if content_vec is not None else 'Final_Hybrid_Model'
    )
    model.summary()

    total_params = int(model.count_params())
    trainable_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    non_trainable_params = total_params - trainable_params
    print("\n--- ۳.۱ خلاصه پارامترها ---")
    print(f"شاخه محتوا: {'فعال — ' + content_desc if content_vec is not None else 'غیرفعال'}")
    print(f"ورودی‌های مدل: {[t.name for t in model.inputs]}")
    print(f"بردار ادغام‌شده نهایی: {final_combined_vec.shape[-1]} بعد "
          f"({' + '.join(str(b.shape[-1]) for b in branches)})")
    print(f"مجموع پارامترها: {total_params:,}")
    print(f"  پارامترهای آموزش‌پذیر: {trainable_params:,}")
    print(f"  پارامترهای ثابت (جدول جست‌وجوی محتوا): {non_trainable_params:,}")

    if content_vec is not None:
        verify_content_alignment(domain, data, model, item_seq_input, X_item_test[:64])

    if summary_only:
        print("\nحالت --summary-only: مدل ساخته و بررسی شد؛ هیچ آموزشی انجام نشد.")
        return

    print("\n--- ۴. کامپایل و آموزش کامل مدل ---")
    metrics = ['sparse_categorical_accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top_10_acc')]
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=metrics)
    early_stopper = EarlyStopping(monitor='val_top_10_acc', patience=3, verbose=1, mode='max', restore_best_weights=True)

    print(f"شروع آموزش مدل نهایی {domain}...")
    # EarlyStopping فقط روی مجموعه اعتبارسنجی؛ مجموعه تست دست‌نخورده می‌ماند.
    history = model.fit(X_train, y_train, batch_size=256, epochs=100, validation_data=(X_val, y_val), callbacks=[early_stopper])

    print("\n--- ۵. ذخیره تاریخچه آموزش و رسم نمودارها ---")
    history_keys = list(history.history.keys())
    print(f"کلیدهای history.history: {history_keys}")

    history_file = os.path.join(data_dir, f'training_history{variant_suffix}.json')
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
    loss_fig_path = os.path.join(data_dir, f'fig_loss{variant_suffix}.png')
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
        acc_fig_path = os.path.join(data_dir, f'fig_top10_acc{variant_suffix}.png')
        fig.savefig(acc_fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"نمودار Top-10 Accuracy در '{acc_fig_path}' ذخیره شد.")
    else:
        print(f"هشدار: کلید top_10_acc در history پیدا نشد. کلیدهای موجود: {history_keys}")

    print("\n--- ۶. ارزیابی نهایی مدل روی مجموعه تست (Full-Vocab Keras Metrics) ---")
    print("توجه: این مجموعه در طول آموزش و EarlyStopping اصلاً دیده نشده است.")
    results = model.evaluate(X_test, y_test)
    print(f"✅ مدل {domain} (Final Hybrid) با موفقیت آموزش دید.")
    print(f"Loss (خطا) روی داده‌های تست: {results[0]:.4f}")
    print(f"Accuracy (دقت) روی داده‌های تست: {results[1] * 100:.2f}%")
    print(f"Top 10 Accuracy (full-vocab) روی داده‌های تست: {results[2] * 100:.2f}%")

    model.save(model_file)
    print(f"مدل نهایی در '{model_file}' ذخیره شد.")

    print("\n--- ۷. ارزیابی NCF-Protocol روی مجموعه تست: HR@10، NDCG@10 و میانگین رتبه (1 مثبت در برابر 99 منفی) ---")
    NUM_NEG = 99
    EVAL_CHUNK = 512
    rng = np.random.default_rng(seed=42)
    hits = 0
    ndcg_sum = 0.0
    rank_sum = 0
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

            # رتبه هدف در میان کل ۱۰۰ کاندید (۱-ایندکس)
            rank = int(np.where(ranked == target)[0][0]) + 1
            rank_sum += rank
            if rank <= 10:
                hits += 1
                ndcg_sum += 1.0 / np.log2(rank + 1)

        if (start // EVAL_CHUNK) % 10 == 0:
            print(f"  پیشرفت: {end}/{n_test}")

    hr10      = hits / n_test
    ndcg10    = ndcg_sum / n_test
    mean_rank = rank_sum / n_test

    print("\n" + "=" * 55)
    print(f"  دامنه: {domain.upper()}  (ارزیابی فقط روی مجموعه تست)")
    print(f"  شاخه محتوا: {'فعال (' + content_desc + ')' if content_vec is not None else 'غیرفعال'}")
    print(f"  اندازه‌ها → train: {len(y_train)} | val: {len(y_val)} | test: {n_test}")
    print(f"  HR@10     (NCF Protocol, 1 vs {NUM_NEG}): {hr10   * 100:.2f}%")
    print(f"  NDCG@10   (NCF Protocol, 1 vs {NUM_NEG}): {ndcg10 * 100:.2f}%")
    print(f"  Mean Rank (NCF Protocol, 1 vs {NUM_NEG}): {mean_rank:.2f} از {NUM_NEG + 1}")
    print("=" * 55)

    # --- ۸. مانیفست اجرا: ثبت اینکه این خروجی‌ها از کدام بازو و کدام بذر آمده‌اند ---
    # در try قرار دارد تا هیچ خطایی در این مرحله، نتیجه یک آموزش طولانی را از بین نبرد.
    try:
        manifest_file = os.path.join(data_dir, f'run_manifest{variant_suffix}.json')
        manifest = {
            'domain': domain,
            'use_content': bool(use_content),
            'arm': 'content' if use_content else 'baseline',
            'content_branch': content_desc,
            'training_seed': SEED,
            'model_file': model_file,
            'history_file': history_file,
            'params': {
                'total': total_params,
                'trainable': trainable_params,
                'non_trainable': non_trainable_params,
                'fused_vector_dims': int(final_combined_vec.shape[-1]),
            },
            'epochs_run': len(history.history['loss']),
            'best_val_top_10_acc': (max(history.history[top10_val_key])
                                    if top10_val_key else None),
            'split_sizes': {'train': int(len(y_train)), 'val': int(len(y_val)),
                            'test': int(n_test)},
            'test_full_vocab': {'loss': float(results[0]),
                                'accuracy': float(results[1]),
                                'top_10_acc': float(results[2])},
            'test_ncf_protocol': {'num_negatives': NUM_NEG, 'HR@10': float(hr10),
                                  'NDCG@10': float(ndcg10), 'MeanRank': float(mean_rank)},
            'finished_at': datetime.now().isoformat(timespec='seconds'),
        }
        with open(manifest_file, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"مانیفست اجرا در '{manifest_file}' ذخیره شد.")
    except Exception as exc:  # noqa: BLE001 - نباید نتیجه آموزش را از بین ببرد
        print(f"هشدار: نوشتن مانیفست اجرا ناموفق بود ({exc}). مدل و متریک‌ها سالم ذخیره شده‌اند.")

def usage():
    print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
    print("مثال: python train_final_hybrid_model.py movie")
    print("   یا: python train_final_hybrid_model.py book")
    print("انتخاب صریح بازو (بدون ویرایش سورس):")
    print("       python train_final_hybrid_model.py movie --content=off   # مدل پایه")
    print("       python train_final_hybrid_model.py movie --content=on    # مدل محتوایی")
    print("       USE_CONTENT=0 python train_final_hybrid_model.py movie   # معادل با متغیر محیطی")
    print("فقط خلاصه مدل بدون آموزش:")
    print("       python train_final_hybrid_model.py movie --summary-only")


if __name__ == "__main__":
    argv = sys.argv[1:]
    positional = [a for a in argv if not a.startswith('--')]
    flags = [a for a in argv if a.startswith('--')]

    summary_only_flag = False
    content_override_flag = None
    unknown = []
    for flag in flags:
        if flag == '--summary-only':
            summary_only_flag = True
        elif flag.startswith('--content='):
            content_override_flag = flag.split('=', 1)[1]
        else:
            unknown.append(flag)

    if len(positional) != 1 or unknown:
        if unknown:
            print(f"آرگومان ناشناخته: {' '.join(unknown)}")
        usage()
    else:
        main(positional[0], summary_only=summary_only_flag,
             content_override=content_override_flag)