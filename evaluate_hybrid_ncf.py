import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import random
import sys
from tqdm import tqdm

def evaluate_hr_at_10_universal(model, X_user, X_item, X_time, y, n_items, num_negatives=99):
    hits = 0
    total = len(y)
    
    input_count = len(model.inputs)
    
    sample_indices = random.sample(range(total), min(1000, total)) 
    
    print(f"--- ارزیابی مدل (Input Count: {input_count}) به روش NCF (1 real vs {num_negatives} negatives) ---")
    
    for idx in tqdm(sample_indices):
        target_user = X_user[idx]
        user_item_seq = X_item[idx]
        user_time_seq = X_time[idx]
        target_item = y[idx] 
        
        negatives = []
        while len(negatives) < num_negatives:
            neg_item = random.randint(1, n_items - 1)
            if neg_item != target_item:
                negatives.append(neg_item)
        
        items_to_rank = [target_item] + negatives
        
        if input_count == 3:
            pred_vector = model.predict([
                np.array([target_user]), 
                np.array([user_item_seq]), 
                np.array([user_time_seq])
            ], verbose=0)[0]
        else:
            pred_vector = model.predict([
                np.array([user_item_seq]), 
                np.array([user_time_seq])
            ], verbose=0)[0]
        
        scores = {}
        for item_id in items_to_rank:
            if item_id < len(pred_vector):
                scores[item_id] = pred_vector[item_id]
            else:
                scores[item_id] = -1.0
            
        ranked_items = sorted(scores, key=scores.get, reverse=True)
        top_10 = ranked_items[:10]
        
        if target_item in top_10:
            hits += 1
            
    return hits / len(sample_indices)

def get_best_model_path(data_dir, domain):
    """
    به طور خودکار بهترین مدل موجود در پوشه را پیدا می‌کند.
    """
    priority_list = [
        'final_hybrid_model.keras',              
        'lstm_dual_masked_attention_model.keras',
        'lstm_simple_model.keras'                
    ]
    
    for model_name in priority_list:
        path = os.path.join(data_dir, model_name)
        if os.path.exists(path):
            return path
            
    return None

def main():
    if len(sys.argv) < 2:
        print("❌ خطا: لطفاً دامنه را مشخص کنید.")
        print("الگو: python evaluate_hybrid_ncf.py [movie/book]")
        return

    domain = sys.argv[1]
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    
    model_file = get_best_model_path(data_dir, domain)

    if not os.path.exists(processed_file):
        print(f"❌ فایل داده {processed_file} پیدا نشد. ابتدا build_dataset.py را اجرا کنید.")
        return

    if not model_file:
        print(f"❌ هیچ مدل آموزش‌دیده‌ای در {data_dir} پیدا نشد.")
        return

    print(f"📊 دامنه انتخاب شده: {domain}")
    print(f"📂 بارگذاری داده‌ها از: {processed_file}")
    data = np.load(processed_file)
    
    limit = 20000
    X_user_test = data['X_user'][-limit:]
    X_item_test = data['X_item'][-limit:]
    X_time_test = data['X_time'][-limit:]
    y_test = data['y'][-limit:]
    n_items = int(data['n_items'])
    
    print(f"🧠 بارگذاری مدل هوشمند از: {os.path.basename(model_file)}")
    try:
        model = load_model(model_file)
    except Exception as e:
        print(f"خطا در بارگذاری مدل: {e}")
        return
    
    hr_10 = evaluate_hr_at_10_universal(model, X_user_test, X_item_test, X_time_test, y_test, n_items)
    
    print("\n" + "="*50)
    print(f"✅ FINAL HIT RATIO @ 10 ({domain.upper()}): {hr_10 * 100:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()