import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import os

MAX_SEQUENCE_LENGTH = 20

def load_titles(domain, data_dir):
    titles = {}
    try:
        if domain == 'movie':
            movies_file = os.path.join(data_dir, 'movies.dat')
            if os.path.exists(movies_file):
                df = pd.read_csv(movies_file, sep='::', engine='python', encoding='latin-1', header=None, names=['ID', 'Title', 'Genre'])
                titles = dict(zip(df['ID'], df['Title']))
        
        elif domain == 'book':
            books_file = os.path.join(data_dir, 'books.csv')
            if os.path.exists(books_file):
                df = pd.read_csv(books_file)
                titles = dict(zip(df['book_id'], df['original_title']))
    except:
        pass
    return titles

def main():
    if len(sys.argv) < 3:
        print("خطا: لطفاً دامنه و آیدی کاربر را وارد کنید.")
        print("مثال: python run_hybrid_interactive.py movie 1")
        return

    domain = sys.argv[1]
    user_id_input = int(sys.argv[2])
    
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    
    model_file = os.path.join(data_dir, 'final_hybrid_model.keras')

    print(f"--- سیستم پیشنهادگر پیشرفته (NCF + LSTM + Attention) ---")
    print(f"User ID: {user_id_input}")

    if not os.path.exists(model_file):
        print(f"❌ خطا: فایل مدل '{model_file}' پیدا نشد.")
        print("لطفاً ابتدا 'python train_final_hybrid_model.py' را برای این دامنه اجرا کنید.")
        return

    print("⏳ بارگذاری داده‌ها...")
    data = np.load(processed_file)
    X_item = data['X_item']
    X_time = data['X_time']

    user_idx = user_id_input if user_id_input < len(X_item) else 0

    current_item_seq = list(X_item[user_idx][X_item[user_idx] > 0])
    current_time_seq = list(X_time[user_idx][X_time[user_idx] > 0])
    
    print(f"⏳ بارگذاری مدل {os.path.basename(model_file)}...")
    model = load_model(model_file)
    titles_map = load_titles(domain, data_dir)
    
    print("✅ سیستم آماده است.")
    print("-" * 50)

    last_recommended_ids = set()

    while True:
        print("\n📜 تاریخچه (5 تای آخر):")
        recent = current_item_seq[-5:]
        for item_id in recent:
            title = titles_map.get(item_id, f"Item {item_id}")
            print(f"   - {title}")

        user_input = np.array([user_idx]) 
        item_seq_input = pad_sequences([current_item_seq], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
        time_seq_input = pad_sequences([current_time_seq], maxlen=MAX_SEQUENCE_LENGTH, padding='pre')
        
        preds = model.predict([user_input, item_seq_input, time_seq_input], verbose=0)[0]

        for seen_item in current_item_seq:
            if seen_item < len(preds):
                preds[seen_item] = -1.0

        top_indices = preds.argsort()[::-1][:10]

        print("\n🎁 پیشنهادات جدید (Hybrid AI):")
        current_recommendations = []
        
        for rank, item_id in enumerate(top_indices, 1):
            title = titles_map.get(item_id, f"Item {item_id}")
            prob = preds[item_id] * 100
            if prob < 0: prob = 0
            
            if (len(last_recommended_ids) > 0) and (item_id not in last_recommended_ids):
                new_tag = " 🆕"
            else:
                new_tag = ""

            print(f"   [{rank}] {title} ({prob:.1f}%){new_tag}")
            current_recommendations.append(item_id)

        last_recommended_ids = set(current_recommendations)

        print("\n👇 انتخاب (1-10) یا 'q':")
        choice = input("> ")

        if choice.lower() == 'q':
            break
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < 10:
                selected_item = current_recommendations[choice_idx]
                
                if preds[selected_item] == -1.0:
                    print("❌ تکراری.")
                    continue

                title = titles_map.get(selected_item, f"Item {selected_item}")
                print(f"✅ انتخاب شد: {title}")
                
                current_item_seq.append(selected_item)
                last_time = current_time_seq[-1] if current_time_seq else 1
                current_time_seq.append(last_time)
                
                if len(current_item_seq) > MAX_SEQUENCE_LENGTH * 2:
                     current_item_seq = current_item_seq[-MAX_SEQUENCE_LENGTH:]
                     current_time_seq = current_time_seq[-MAX_SEQUENCE_LENGTH:]
            else:
                print("❌ عدد اشتباه.")
        except ValueError:
            print("❌ ورودی نامعتبر.")

if __name__ == "__main__":
    main()