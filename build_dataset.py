import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import sys
import os

# روی ویندوز اگر خروجی به فایل/pipe هدایت شود، پایتون از cp1252 استفاده می‌کند
# و چاپ متن فارسی با UnicodeEncodeError کرش می‌کند. اجباراً UTF-8 می‌کنیم.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Night'

def pad(seqs, maxlen):
    """پدینگ امن: اگر لیست خالی باشد آرایه‌ای با شکل درست و صفر سطر برمی‌گرداند."""
    if len(seqs) == 0:
        return np.zeros((0, maxlen), dtype='int32')
    return pad_sequences(seqs, maxlen=maxlen, padding='pre')

def main(domain):
    print(f"--- شروع پردازش برای دامنه: {domain} ---")
    
    data_dir = f"{domain}_data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if domain == 'movie':
        print("در حال خواندن داده‌های فیلم...")
        ratings_file = os.path.join(data_dir, 'ratings.dat')
        column_names = ['UserID', 'ItemID', 'Rating', 'Timestamp']
        data = pd.read_csv(ratings_file, sep='::', names=column_names, engine='python', encoding='latin-1')
        
        data['DateTime'] = pd.to_datetime(data['Timestamp'], unit='s')
        data['HourOfDay'] = data['DateTime'].dt.hour
        data['TimeOfDay'] = data['HourOfDay'].apply(get_time_of_day)

    elif domain == 'book':
        print("در حال خواندن داده‌های کتاب...")
        ratings_file_1 = os.path.join(data_dir, 'book_ratings.csv')
        ratings_file_2 = os.path.join(data_dir, 'ratings.csv')
        
        if os.path.exists(ratings_file_1):
            ratings_file = ratings_file_1
        elif os.path.exists(ratings_file_2):
            ratings_file = ratings_file_2
        else:
            print(f"خطا: فایل book_ratings.csv یا ratings.csv در پوشه {data_dir} پیدا نشد.")
            return

        data = pd.read_csv(ratings_file)
        data.rename(columns={'user_id': 'UserID', 'book_id': 'ItemID', 'rating': 'Rating'}, inplace=True)
        
        data['Timestamp'] = data.reset_index().index
        data['TimeOfDay'] = 'AllDay' # ما اطلاعات زمان روز نداریم
    
    else:
        print(f"خطا: دامنه '{domain}' ناشناخته است.")
        return

    print("--- ۲. آماده‌سازی داده‌ها ---")
    
    user_encoder = LabelEncoder()
    data['UserID_encoded'] = user_encoder.fit_transform(data['UserID'])
    n_users = data['UserID_encoded'].nunique()
    print(f"تعداد کاربران یکتا: {n_users}")

    item_encoder = LabelEncoder()
    data['ItemID_encoded'] = item_encoder.fit_transform(data['ItemID']) + 1
    n_items = data['ItemID_encoded'].nunique() + 1
    print(f"تعداد آیتم‌های یکتا (با احتساب پدینگ): {n_items}")

    time_encoder = LabelEncoder()
    data['TimeOfDay_encoded'] = time_encoder.fit_transform(data['TimeOfDay']) + 1
    n_time_features = data['TimeOfDay_encoded'].nunique() + 1
    print(f"تعداد دسته‌های زمانی (با احتساب پدینگ): {n_time_features}")

    print("\n--- ۳. ساخت دنباله‌ها (تقسیم leave-TWO-out) ---")
    # مرتب‌سازی پایدار (mergesort): برای فیلم بر اساس Timestamp واقعی؛ برای کتاب،
    # Timestamp همان اندیس سطر اصلی است، پس ترتیب سطرها دقیقاً مثل قبل حفظ می‌شود.
    data = data.sort_values(['UserID_encoded', 'Timestamp'], kind='mergesort')
    grouped = data.groupby('UserID_encoded')
    MIN_SEQUENCE_LENGTH = 3
    MAX_SEQUENCE_LENGTH = 20

    tr_users, tr_item_seqs, tr_time_seqs, tr_labels = [], [], [], []
    va_users, va_item_seqs, va_time_seqs, va_labels = [], [], [], []
    te_users, te_item_seqs, te_time_seqs, te_labels = [], [], [], []

    n_users_seen = 0
    n_users_dropped_short = 0      # کمتر از MIN_SEQUENCE_LENGTH → کلاً حذف
    n_users_no_val = 0             # کوتاه‌تر از آن‌که هر سه بخش بدون هم‌پوشانی ساخته شود
    n_users_no_train = 0           # هیچ جفت آموزشی نداشت
    n_train_dropped_leak = 0       # جفت آموزشی که برچسبش با برچسب val/test یکی بود
    n_val_inputs_cleaned = 0       # ورودی val که آیتم تست در آن تکرار شده بود

    for user_id, group in grouped:
        n_users_seen += 1
        item_list = group['ItemID_encoded'].tolist()
        time_list = group['TimeOfDay_encoded'].tolist()
        L = len(item_list)
        if L < MIN_SEQUENCE_LENGTH:
            n_users_dropped_short += 1
            continue

        # --- نمونه تست: تمام تاریخچه به‌جز آخری → آیتم آخر ---
        test_label = item_list[-1]
        te_users.append(user_id)
        te_item_seqs.append(item_list[:-1])
        te_time_seqs.append(time_list[:-1])
        te_labels.append(test_label)

        # --- نمونه اعتبارسنجی: هرچه قبل از یکی‌مانده‌به‌آخر → آیتم یکی‌مانده‌به‌آخر ---
        # برای داشتن هم‌زمانِ train + val + test بدون هم‌پوشانی، حداقل ۴ تعامل لازم است.
        val_label = None
        if L >= 4:
            val_items = item_list[:-2]
            val_times = time_list[:-2]
            # تضمین: آیتم تست نباید در ورودی اعتبارسنجی باشد
            # (فقط وقتی رخ می‌دهد که همان آیتم قبلاً هم برای این کاربر ثبت شده باشد)
            if test_label in val_items:
                keep = [j for j, it in enumerate(val_items) if it != test_label]
                val_items = [val_items[j] for j in keep]
                val_times = [val_times[j] for j in keep]
                n_val_inputs_cleaned += 1
            if val_items:
                val_label = item_list[-2]
                va_users.append(user_id)
                va_item_seqs.append(val_items)
                va_time_seqs.append(val_times)
                va_labels.append(val_label)
            else:
                n_users_no_val += 1
        else:
            n_users_no_val += 1

        # --- نمونه‌های آموزشی: پیشوندهای رشدیابنده، تا قبل از نمونه اعتبارسنجی ---
        train_end = L - 2 if val_label is not None else L - 1
        forbidden = {test_label}
        if val_label is not None:
            forbidden.add(val_label)

        n_tr_before = len(tr_labels)
        for i in range(1, train_end):
            if item_list[i] in forbidden:
                # برچسب آموزشی هرگز نباید با آیتم val یا test این کاربر یکی باشد
                n_train_dropped_leak += 1
                continue
            tr_users.append(user_id)
            tr_item_seqs.append(item_list[:i])
            tr_time_seqs.append(time_list[:i])
            tr_labels.append(item_list[i])
        if len(tr_labels) == n_tr_before:
            n_users_no_train += 1

    n_users_kept = len(te_users)
    print(f"کاربران بررسی‌شده: {n_users_seen}")
    print(f"کاربران حذف‌شده (کمتر از {MIN_SEQUENCE_LENGTH} تعامل): {n_users_dropped_short}")
    print(f"کاربران نگه‌داشته‌شده: {n_users_kept}")
    print(f"کاربرانی که نمونه اعتبارسنجی نگرفتند (دنباله خیلی کوتاه): {n_users_no_val}")
    print(f"کاربرانی که هیچ نمونه آموزشی نگرفتند: {n_users_no_train}")
    print(f"جفت‌های آموزشی حذف‌شده به دلیل تکرار آیتم val/test: {n_train_dropped_leak}")
    print(f"ورودی‌های اعتبارسنجی پاک‌سازی‌شده (آیتم تست تکراری حذف شد): {n_val_inputs_cleaned}")
    print(f"دنباله‌های آموزشی: {len(tr_labels)}")
    print(f"نمونه‌های اعتبارسنجی (یک به‌ازای هر کاربر): {len(va_labels)}")
    print(f"نمونه‌های تست (یک به‌ازای هر کاربر): {len(te_labels)}")

    print("\n--- ۳.۱ بررسی نشتی (leakage checks) ---")
    val_label_by_user  = dict(zip(va_users, va_labels))
    test_label_by_user = dict(zip(te_users, te_labels))

    leak_train_vs_val  = sum(1 for u, lab in zip(tr_users, tr_labels)
                             if u in val_label_by_user and val_label_by_user[u] == lab)
    leak_train_vs_test = sum(1 for u, lab in zip(tr_users, tr_labels)
                             if u in test_label_by_user and test_label_by_user[u] == lab)
    leak_test_in_val_input = sum(1 for u, seq in zip(va_users, va_item_seqs)
                                 if u in test_label_by_user and test_label_by_user[u] in seq)
    val_one_per_user  = len(set(va_users)) == len(va_users)
    test_one_per_user = len(set(te_users)) == len(te_users)
    val_users_subset  = set(va_users).issubset(set(te_users))

    print(f"[check] برچسب آموزشی برابر با برچسب اعتبارسنجی همان کاربر: {leak_train_vs_val} (باید ۰ باشد)")
    print(f"[check] برچسب آموزشی برابر با برچسب تست همان کاربر: {leak_train_vs_test} (باید ۰ باشد)")
    print(f"[check] آیتم تست موجود در ورودی اعتبارسنجی: {leak_test_in_val_input} (باید ۰ باشد)")
    print(f"[check] اعتبارسنجی دقیقاً یک نمونه به‌ازای هر کاربر: {val_one_per_user}")
    print(f"[check] تست دقیقاً یک نمونه به‌ازای هر کاربر: {test_one_per_user}")
    print(f"[check] کاربران اعتبارسنجی زیرمجموعه کاربران تست: {val_users_subset}")

    assert leak_train_vs_val == 0 and leak_train_vs_test == 0 and leak_test_in_val_input == 0, \
        "نشتی داده تشخیص داده شد!"
    assert val_one_per_user and test_one_per_user, "نمونه‌های val/test باید یکتا به‌ازای هر کاربر باشند!"

    print("\n--- ۴. پدینگ دنباله‌ها ---")
    X_user_train = np.array(tr_users)
    X_item_train = pad(tr_item_seqs, MAX_SEQUENCE_LENGTH)
    X_time_train = pad(tr_time_seqs, MAX_SEQUENCE_LENGTH)
    y_train      = np.array(tr_labels)

    X_user_val   = np.array(va_users)
    X_item_val   = pad(va_item_seqs, MAX_SEQUENCE_LENGTH)
    X_time_val   = pad(va_time_seqs, MAX_SEQUENCE_LENGTH)
    y_val        = np.array(va_labels)

    X_user_test  = np.array(te_users)
    X_item_test  = pad(te_item_seqs, MAX_SEQUENCE_LENGTH)
    X_time_test  = pad(te_time_seqs, MAX_SEQUENCE_LENGTH)
    y_test       = np.array(te_labels)

    print("\n--- ۵. ذخیره فایل‌ها ---")
    output_path = os.path.join(data_dir, 'processed_data.npz')
    np.savez_compressed(
        output_path,
        X_user_train=X_user_train, X_item_train=X_item_train,
        X_time_train=X_time_train, y_train=y_train,
        X_user_val=X_user_val,     X_item_val=X_item_val,
        X_time_val=X_time_val,     y_val=y_val,
        X_user_test=X_user_test,   X_item_test=X_item_test,
        X_time_test=X_time_test,   y_test=y_test,
        n_users=n_users, n_items=n_items, n_time_features=n_time_features
    )
    print(f"داده‌های پردازش شده در '{output_path}' ذخیره شدند.")
    print(f"اندازه‌های نهایی [{domain}] → train: {len(y_train)} | val: {len(y_val)} | test: {len(y_test)}")

    joblib.dump(user_encoder, os.path.join(data_dir, 'user_encoder.joblib'))
    joblib.dump(item_encoder, os.path.join(data_dir, 'item_encoder.joblib'))
    joblib.dump(time_encoder, os.path.join(data_dir, 'time_encoder.joblib'))
    print("انکودرها با موفقیت ذخیره شدند.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python build_dataset.py movie")
        print("   یا: python build_dataset.py book")
    else:
        domain = sys.argv[1]
        main(domain)