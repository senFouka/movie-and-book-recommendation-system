import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import sys 
import os 

def get_time_of_day(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Night'

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

    print("\n--- ۳. ساخت دنباله‌ها (تقسیم leave-one-out) ---")
    data = data.sort_values(['UserID_encoded', 'Timestamp'])
    grouped = data.groupby('UserID_encoded')
    MIN_SEQUENCE_LENGTH = 3
    MAX_SEQUENCE_LENGTH = 20

    tr_users, tr_item_seqs, tr_time_seqs, tr_labels = [], [], [], []
    te_users, te_item_seqs, te_time_seqs, te_labels = [], [], [], []

    for user_id, group in grouped:
        item_list = group['ItemID_encoded'].tolist()
        time_list = group['TimeOfDay_encoded'].tolist()
        if len(item_list) < MIN_SEQUENCE_LENGTH:
            continue

        # یک نمونه تست: تمام تاریخچه به‌جز آخری → آیتم آخر
        te_users.append(user_id)
        te_item_seqs.append(item_list[:-1])
        te_time_seqs.append(time_list[:-1])
        te_labels.append(item_list[-1])

        # نمونه‌های آموزشی: پیشوندهای رشد‌یابنده، بدون لمس آیتم آخر
        for i in range(1, len(item_list) - 1):
            tr_users.append(user_id)
            tr_item_seqs.append(item_list[:i])
            tr_time_seqs.append(time_list[:i])
            tr_labels.append(item_list[i])

    print(f"دنباله‌های آموزشی: {len(tr_labels)}")
    print(f"نمونه‌های تست (leave-one-out، یک به‌ازای هر کاربر): {len(te_labels)}")

    print("\n--- ۴. پدینگ دنباله‌ها ---")
    X_user_train = np.array(tr_users)
    X_item_train = pad_sequences(tr_item_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    X_time_train = pad_sequences(tr_time_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    y_train      = np.array(tr_labels)

    X_user_test  = np.array(te_users)
    X_item_test  = pad_sequences(te_item_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    X_time_test  = pad_sequences(te_time_seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
    y_test       = np.array(te_labels)

    print("\n--- ۵. ذخیره فایل‌ها ---")
    output_path = os.path.join(data_dir, 'processed_data.npz')
    np.savez_compressed(
        output_path,
        X_user_train=X_user_train, X_item_train=X_item_train,
        X_time_train=X_time_train, y_train=y_train,
        X_user_test=X_user_test,   X_item_test=X_item_test,
        X_time_test=X_time_test,   y_test=y_test,
        n_users=n_users, n_items=n_items, n_time_features=n_time_features
    )
    print(f"داده‌های پردازش شده در '{output_path}' ذخیره شدند.")

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