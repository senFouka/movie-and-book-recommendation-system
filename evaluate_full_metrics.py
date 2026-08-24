"""
ارزیابی کامل مدل‌های از پیش آموزش‌دیده (بدون هیچ آموزش مجددی).

این اسکریپت فقط مدل ذخیره‌شده `final_hybrid_model.keras` را بارگذاری می‌کند و همه
سنجه‌ها را روی «مجموعه تست» محاسبه می‌کند. هیچ عددی در کد hardcode نشده است؛ تمام
مقادیر از روی پیش‌بینی واقعی مدل و آمار واقعی مجموعه آموزش محاسبه می‌شوند.

پروتکل ارزیابی (دقیقاً همان پروتکل train_final_hybrid_model.py):
    leave-one-out با ۱ آیتم مثبت در برابر ۹۹ آیتم منفی، seed = 42.
    دنباله نمونه‌برداری منفی‌ها عیناً بازتولید می‌شود تا اعداد با اجرای آموزش قابل مقایسه باشد،
    و هر سه روش (مدل پیشنهادی، Popularity، ItemKNN) دقیقاً روی همان ۱۰۰ کاندید امتیاز می‌گیرند.
"""

import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import sys
import os
import json

# روی ویندوز اگر خروجی به فایل/pipe هدایت شود، پایتون از cp1252 استفاده می‌کند
# و چاپ متن فارسی با UnicodeEncodeError کرش می‌کند. اجباراً UTF-8 می‌کنیم.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

# --- پارامترهای پروتکل ارزیابی ---
NUM_NEG = 99            # تعداد آیتم منفی به‌ازای هر کاربر تست
SEED = 42               # همان seed اجرای آموزش
EVAL_CHUNK = 512        # اندازه دسته برای model.predict
PREDICT_BATCH = 256
TOP_K_LIST = (5, 10)    # K هایی که سنجه‌های top-K برایشان گزارش می‌شود
REC_LIST_SIZE = 10      # طول لیست پیشنهاد برای Coverage/Diversity/Novelty
KNN_NEIGHBORS = 20      # تعداد همسایه در ItemKNN
SIM_BLOCK = 512         # اندازه بلوک برای محاسبه ماتریس شباهت آیتم-آیتم


# ----------------------------------------------------------------------------
# بازسازی تاریخچه آموزشی هر کاربر
# ----------------------------------------------------------------------------
def build_training_interactions(X_user_train, X_item_train, y_train, n_users, n_items):
    """
    تاریخچه آموزشی هر کاربر را از آرایه‌های آموزش بازسازی می‌کند.

    build_dataset.py برای هر کاربر جفت‌های «پیشوند → آیتم بعدی» می‌سازد، پس مجموعه
    برچسب‌های آموزشی یک کاربر دقیقاً برابر است با تعاملات آموزشی او به‌جز اولین آیتم
    (اولین آیتم هرگز برچسب نمی‌شود، فقط ورودی است). بنابراین:

        تاریخچه آموزشی کاربر u = {اولین آیتم u} ∪ {همه برچسب‌های آموزشی u}

    اولین آیتم از نخستین سطر آموزشی کاربر برداشت می‌شود: پدینگ 'pre' است، پس اولین
    عنصر ناصفرِ آن سطر همان اولین آیتم کاربر است.

    نکته صداقتی: تنها تعاملاتی که در این بازسازی نمی‌آیند، جفت‌هایی هستند که
    build_dataset.py به‌عنوان نشتی حذف کرده (برچسب آموزشی برابر با آیتم val/test).
    تعداد آن‌ها در گزارش build_dataset چاپ شده و برای هر دو دامنه صفر/ناچیز است.
    """
    order = np.argsort(X_user_train, kind='mergesort')
    su = X_user_train[order]
    # اندیس نخستین سطر هر کاربر
    first_pos = np.flatnonzero(np.concatenate(([True], su[1:] != su[:-1])))
    first_rows = order[first_pos]
    users_with_train = su[first_pos]

    first_items = np.empty(len(first_rows), dtype=np.int64)
    for k, row_idx in enumerate(first_rows):
        row = X_item_train[row_idx]
        nz = np.flatnonzero(row)
        first_items[k] = row[nz[0]]

    rows = np.concatenate([X_user_train.astype(np.int64), users_with_train.astype(np.int64)])
    cols = np.concatenate([y_train.astype(np.int64), first_items])

    # ماتریس بازخورد ضمنی (باینری) کاربر × آیتم
    R = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_items)
    )
    R.sum_duplicates()
    counts_per_pair = R.data.copy()          # تکرار واقعی هر جفت (کاربر، آیتم)
    R.data = np.ones_like(R.data)            # باینری‌سازی برای شباهت کسینوسی

    # فراوانی هر آیتم در مجموعه آموزش (با احتساب تکرارها)
    item_counts = np.bincount(cols, minlength=n_items).astype(np.float64)
    return R, item_counts, len(rows), counts_per_pair


def build_item_similarity(R, n_items):
    """
    ماتریس شباهت کسینوسی آیتم-آیتم روی ماتریس باینری کاربر × آیتم.
    S[i, j] = (r_i · r_j) / (||r_i|| * ||r_j||)
    به‌صورت بلوکی محاسبه می‌شود تا حافظه کنترل‌شده بماند.
    """
    col_norm = np.sqrt(np.asarray(R.sum(axis=0)).ravel()).astype(np.float32)
    inv = np.zeros_like(col_norm)
    np.divide(1.0, col_norm, out=inv, where=col_norm > 0)

    Rn = R.multiply(inv[None, :]).tocsc().astype(np.float32)
    RnT = Rn.T.tocsr()

    S = np.zeros((n_items, n_items), dtype=np.float32)
    for start in range(0, n_items, SIM_BLOCK):
        end = min(start + SIM_BLOCK, n_items)
        S[start:end] = (RnT[start:end] @ Rn).toarray()
    return S


# ----------------------------------------------------------------------------
# سنجه‌ها از روی آرایه رتبه‌ها
# ----------------------------------------------------------------------------
def metrics_from_ranks(ranks):
    """
    ranks: رتبه آیتم مثبت در میان ۱۰۰ کاندید (۱-ایندکس).

    چون در پروتکل leave-one-out دقیقاً یک آیتم مرتبط وجود دارد:
        HR@K        = نسبت کاربرانی که rank <= K
        Recall@K    = HR@K  (تنها ۱ آیتم مرتبط ⇒ نرخ بازیابی همان نرخ اصابت است)
        Precision@K = HR@K / K  (حداکثر ۱ آیتم مرتبط در لیست K تایی)
        NDCG@K      = میانگین 1/log2(rank+1) برای rank <= K (IDCG = 1)
        MRR         = میانگین 1/rank روی همه کاربران
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    out = {}
    for K in TOP_K_LIST:
        hit = ranks <= K
        hr = float(hit.mean())
        out[f'HR@{K}'] = hr
        out[f'NDCG@{K}'] = float(np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0).mean())
        out[f'Precision@{K}'] = hr / K
        out[f'Recall@{K}'] = hr
    out['MRR'] = float((1.0 / ranks).mean())
    out['MeanRank'] = float(ranks.mean())
    return out


def rank_of_target(scores, target_pos=0):
    """رتبه ۱-ایندکسِ کاندید شماره target_pos پس از مرتب‌سازی نزولی امتیازها."""
    order = np.argsort(-scores, kind='stable')
    return int(np.flatnonzero(order == target_pos)[0]) + 1


def top_k_from_scores(scores, k=REC_LIST_SIZE):
    """
    لیست پیشنهاد top-k روی «کل کاتالوگ» از روی بردار امتیاز کل آیتم‌ها.
    اندیس ۰ (پدینگ) همیشه حذف می‌شود. خروجی به ترتیب نزولی امتیاز مرتب است.
    این تابع برای هر سه روش یکسان به کار می‌رود تا رویه تولید لیست دقیقاً یکی باشد.
    """
    s = np.array(scores, dtype=np.float64, copy=True)
    s[0] = -np.inf
    idx = np.argpartition(-s, k)[:k]
    return idx[np.argsort(-s[idx])]


def knn_scores_full_catalog(S, hist):
    """
    امتیاز ItemKNN برای «همه» آیتم‌های کاتالوگ نسبت به تاریخچه آموزشی یک کاربر.

    دقیقاً همان تابع امتیازدهی‌ای که در جدول رتبه‌بندی روی ۱۰۰ کاندید به کار می‌رود،
    فقط اینجا به‌جای ۱۰۰ کاندید روی کل کاتالوگ اعمال می‌شود:
        score(i) = مجموع بزرگ‌ترین KNN_NEIGHBORS مقدارِ شباهت بین i و آیتم‌های تاریخچه کاربر
    (شباهت آیتم با خودش حذف می‌شود.)

    نکته پیاده‌سازی: S متقارن است، پس S[:, hist] برابر S[hist, :].T است؛ گرفتن سطرها
    به‌جای ستون‌ها دسترسی حافظه را پیوسته می‌کند و محاسبه را به‌طور محسوسی سریع‌تر.
    """
    if len(hist) == 0:
        return None
    sub = np.ascontiguousarray(S[hist, :].T)          # (n_items, |hist|)
    sub[hist, np.arange(len(hist))] = 0.0             # حذف خودشباهتی
    if sub.shape[1] > KNN_NEIGHBORS:
        part = np.partition(sub, -KNN_NEIGHBORS, axis=1)[:, -KNN_NEIGHBORS:]
        return part.sum(axis=1)
    return sub.sum(axis=1)


class BeyondAccuracy:
    """
    انباشتگر سنجه‌های فراتر از دقت روی لیست‌های top-N کل کاتالوگ.

    هر سه روش (مدل پیشنهادی، Popularity، ItemKNN) از یک نمونه از این کلاس استفاده
    می‌کنند، پس Coverage/Diversity/Novelty با تعریف کاملاً یکسان محاسبه می‌شوند:

      Coverage  = |اجتماع آیتم‌های ظاهرشده در لیست‌ها| / اندازه کاتالوگ
      Diversity = میانگین ناهمسانی درون‌لیستی = میانگین (۱ - شباهت کسینوسی) روی همه
                  زوج‌های لیست، در فضای embedding آیتمِ آموخته‌شده مدل، سپس میانگین
                  روی کاربران. (ژانر وارد مدل نشده، پس فضای embedding مبنای شباهت است.)
                  توجه: این فضا برای هر سه روش یکی است تا مقایسه منصفانه بماند؛ یعنی
                  Diversity کیفیت «لیست» را می‌سنجد، نه فضای بازنمایی روش را.
      Novelty   = میانگین self-information = میانگین -log2(p_i) روی آیتم‌های پیشنهادی،
                  که p_i فراوانی نسبی آیتم i در مجموعه آموزش (با هموارسازی لاپلاس) است.
    """

    def __init__(self, item_emb_unit, self_information, item_counts, list_size=REC_LIST_SIZE):
        self.item_emb_unit = item_emb_unit
        self.self_information = self_information
        self.item_counts = item_counts
        self.pair_i, self.pair_j = np.triu_indices(list_size, k=1)
        self.covered = set()
        self.div_sum = 0.0
        self.nov_sum = 0.0
        self.n_lists = 0
        self.n_slots = 0
        self.zero_pop = 0

    def add(self, rec):
        self.covered.update(int(x) for x in rec)
        E = self.item_emb_unit[rec]
        cos_pairs = np.einsum('ij,ij->i', E[self.pair_i], E[self.pair_j])
        self.div_sum += float((1.0 - cos_pairs).mean())
        self.nov_sum += float(self.self_information[rec - 1].sum())
        self.n_lists += 1
        self.n_slots += len(rec)
        self.zero_pop += int((self.item_counts[rec] == 0).sum())

    def result(self, catalog_size):
        return {
            'coverage': len(self.covered) / catalog_size,
            'coverage_unique_items': len(self.covered),
            'diversity': self.div_sum / self.n_lists,
            'novelty_bits': self.nov_sum / self.n_slots,
            'recommended_slots_with_zero_train_frequency': int(self.zero_pop),
            'total_recommended_slots': int(self.n_slots),
        }


# ----------------------------------------------------------------------------
def main(domain, variant=''):
    # variant='' → مدل پایه؛ variant='content' → مدل دارای شاخه محتوا
    suffix = f'_{variant}' if variant else ''
    data_dir = f"{domain}_data"
    processed_file = os.path.join(data_dir, 'processed_data.npz')
    model_file = os.path.join(data_dir, f'final_hybrid{suffix}_model.keras')
    out_file = os.path.join(data_dir, f'full_metrics{suffix}.json')

    print(f"--- ۱. بارگذاری داده و مدل ({domain}) ---")
    if not os.path.exists(processed_file):
        print(f"فایل '{processed_file}' پیدا نشد.")
        return
    if not os.path.exists(model_file):
        print(f"مدل '{model_file}' پیدا نشد. ابتدا train_final_hybrid_model.py را اجرا کنید.")
        return

    data = np.load(processed_file)
    X_user_train = data['X_user_train']
    X_item_train = data['X_item_train']
    y_train = data['y_train']
    X_user_test = data['X_user_test']
    X_item_test = data['X_item_test']
    X_time_test = data['X_time_test']
    y_test = data['y_test']
    n_users = int(data['n_users'])
    n_items = int(data['n_items'])

    n_test = len(y_test)
    print(f"کاربران: {n_users} | آیتم‌ها (با پدینگ): {n_items}")
    print(f"نمونه‌های آموزش: {len(y_train)} | نمونه‌های تست: {n_test}")
    print("ارزیابی فقط روی مجموعه تست انجام می‌شود؛ هیچ آموزشی انجام نمی‌شود.")

    model = tf.keras.models.load_model(model_file, compile=False)
    print(f"مدل از '{model_file}' بارگذاری شد.")

    item_emb = model.get_layer('ItemEmbedding').get_weights()[0]  # (n_items, d)
    emb_norm = np.linalg.norm(item_emb, axis=1, keepdims=True)
    item_emb_unit = np.divide(item_emb, emb_norm, out=np.zeros_like(item_emb), where=emb_norm > 0)

    print("\n--- ۲. بازسازی آمار مجموعه آموزش (برای Popularity / ItemKNN / Novelty) ---")
    R, item_counts, total_interactions, _ = build_training_interactions(
        X_user_train, X_item_train, y_train, n_users, n_items
    )
    print(f"تعاملات آموزشی بازسازی‌شده: {total_interactions}")
    print(f"آیتم‌های دیده‌شده در آموزش: {int((item_counts[1:] > 0).sum())} از {n_items - 1}")

    print("در حال محاسبه ماتریس شباهت کسینوسی آیتم-آیتم (برای ItemKNN)...")
    S = build_item_similarity(R, n_items)
    R_csr = R.tocsr()
    print(f"ماتریس شباهت ساخته شد: {S.shape}")

    # --- شکستن تساوی امتیازها به‌صورت قطعی (rng جدا تا جریان نمونه‌برداری منفی دست‌نخورده بماند) ---
    tb_rng = np.random.default_rng(SEED)
    pop_tiebreak = tb_rng.random(n_items).astype(np.float64) * 0.5   # شمارش‌ها صحیح‌اند ⇒ jitter<0.5 امن است
    knn_tiebreak = tb_rng.random(n_items).astype(np.float64) * 1e-6  # شباهت‌ها در [0,1]

    print("\n--- ۳. ارزیابی روی مجموعه تست (۱ مثبت در برابر ۹۹ منفی، seed=42) ---")
    rng = np.random.default_rng(seed=SEED)   # همان جریان و همان ترتیب اجرای آموزش

    ranks_model = np.zeros(n_test, dtype=np.int32)
    ranks_pop = np.zeros(n_test, dtype=np.int32)
    ranks_knn = np.zeros(n_test, dtype=np.int32)

    # --- انباشت RMSE/MAE ---
    # مدل، softmax روی کل واژگان آیتم تولید می‌کند و «امتیاز صریح» پیش‌بینی نمی‌کند،
    # پس RMSE/MAE به معنای کلاسیکِ خطای پیش‌بینی امتیاز تعریف‌شدنی نیست (پایین‌تر توضیح داده شده).
    # آنچه اینجا محاسبه می‌شود، خطای احتمالِ پیش‌بینی‌شده نسبت به هدف ۰/۱ (relevance) روی همان ۱۰۰ کاندید است.
    sq_err_raw = 0.0
    abs_err_raw = 0.0
    sq_err_norm = 0.0
    abs_err_norm = 0.0
    n_err_terms = 0

    # احتمال محبوبیت با هموارسازی لاپلاس (کاتالوگ = آیتم‌های ۱..n_items-1، بدون پدینگ)
    catalog_size = n_items - 1
    pop_prob = (item_counts[1:] + 1.0) / (total_interactions + catalog_size)
    self_information = -np.log2(pop_prob)     # اندیس k ↔ آیتم k+1

    # --- انباشت سنجه‌های فراتر از دقت: یک انباشتگر برای هر روش ---
    # هر سه با همان رویه «top-10 روی کل کاتالوگ» و همان تعاریف محاسبه می‌شوند.
    # هیچ روشی آیتم‌های دیده‌شده کاربر را فیلتر نمی‌کند (مدل هم نمی‌کند)، تا رویه یکسان بماند.
    ba_model = BeyondAccuracy(item_emb_unit, self_information, item_counts)
    ba_pop   = BeyondAccuracy(item_emb_unit, self_information, item_counts)
    ba_knn   = BeyondAccuracy(item_emb_unit, self_information, item_counts)

    # Popularity مستقل از کاربر است: لیست top-10 آن برای همه کاربران یکی است،
    # پس یک‌بار محاسبه و برای هر کاربر تکرار می‌شود (میانگین‌گیری روی همان تعداد کاربر).
    pop_rec = top_k_from_scores(item_counts.astype(np.float64) + pop_tiebreak)

    print(f"پیش‌بینی دسته‌ای برای {n_test} کاربر تست...")
    for start in range(0, n_test, EVAL_CHUNK):
        end = min(start + EVAL_CHUNK, n_test)
        preds = model.predict(
            [X_user_test[start:end], X_item_test[start:end], X_time_test[start:end]],
            batch_size=PREDICT_BATCH, verbose=0
        )

        # --- لیست پیشنهاد top-10 روی کل کاتالوگ (پدینگ حذف می‌شود) ---
        full = preds.copy()
        full[:, 0] = -np.inf
        top_idx = np.argpartition(-full, REC_LIST_SIZE, axis=1)[:, :REC_LIST_SIZE]
        row_ar = np.arange(top_idx.shape[0])[:, None]
        top_idx = np.take_along_axis(top_idx, np.argsort(-full[row_ar, top_idx], axis=1), axis=1)

        for local_i in range(end - start):
            gi = start + local_i
            target = int(y_test[gi])
            pred_vec = preds[local_i]

            # --- نمونه‌برداری منفی: عیناً همان ترتیب و همان جریان rng اجرای آموزش ---
            neg_pool = np.arange(1, n_items)
            neg_pool = neg_pool[neg_pool != target]
            negatives = rng.choice(neg_pool, size=NUM_NEG, replace=False)
            candidates = np.concatenate([[target], negatives])

            # ---------- ۱) مدل پیشنهادی ----------
            model_scores = pred_vec[candidates].astype(np.float64)
            ranks_model[gi] = rank_of_target(model_scores)

            # ---------- ۲) Popularity (TopPop) ----------
            ranks_pop[gi] = rank_of_target(item_counts[candidates] + pop_tiebreak[candidates])

            # ---------- ۳) ItemKNN ----------
            hist = R_csr.indices[R_csr.indptr[int(X_user_test[gi])]:R_csr.indptr[int(X_user_test[gi]) + 1]]
            if len(hist) == 0:
                knn_scores = knn_tiebreak[candidates]
            else:
                sims = S[np.ix_(candidates, hist)].astype(np.float64)
                # حذف شباهت با خودِ آیتم (کاندیدی که در تاریخچه کاربر هست)
                sims[candidates[:, None] == hist[None, :]] = 0.0
                if sims.shape[1] > KNN_NEIGHBORS:
                    part = np.partition(sims, -KNN_NEIGHBORS, axis=1)[:, -KNN_NEIGHBORS:]
                    knn_scores = part.sum(axis=1)
                else:
                    knn_scores = sims.sum(axis=1)
                knn_scores = knn_scores + knn_tiebreak[candidates]
            ranks_knn[gi] = rank_of_target(knn_scores)

            # ---------- RMSE / MAE روی همان ۱۰۰ کاندید ----------
            relevance = np.zeros(NUM_NEG + 1, dtype=np.float64)
            relevance[0] = 1.0
            probs_raw = model_scores
            denom = probs_raw.sum()
            probs_norm = probs_raw / denom if denom > 0 else np.full_like(probs_raw, 1.0 / len(probs_raw))
            d_raw = probs_raw - relevance
            d_norm = probs_norm - relevance
            sq_err_raw += float((d_raw ** 2).sum())
            abs_err_raw += float(np.abs(d_raw).sum())
            sq_err_norm += float((d_norm ** 2).sum())
            abs_err_norm += float(np.abs(d_norm).sum())
            n_err_terms += len(relevance)

            # ---------- Coverage / Diversity / Novelty (هر سه روش) ----------
            # مدل پیشنهادی: لیست از softmax کل کاتالوگ (بالاتر به‌صورت دسته‌ای محاسبه شد)
            ba_model.add(top_idx[local_i])

            # Popularity: لیست ثابت و مشترک همه کاربران
            ba_pop.add(pop_rec)

            # ItemKNN: همان تابع امتیازدهی جدول رتبه‌بندی، این‌بار روی کل کاتالوگ
            knn_full = knn_scores_full_catalog(S, hist)
            if knn_full is None:
                knn_full = np.zeros(n_items, dtype=np.float64)
            ba_knn.add(top_k_from_scores(knn_full.astype(np.float64) + knn_tiebreak))

        if (start // EVAL_CHUNK) % 10 == 0:
            print(f"  پیشرفت: {end}/{n_test}")

    # ------------------------------------------------------------------
    names_proposed = 'Proposed (Hybrid NCF+LSTM+Attention)'
    names_pop = 'Popularity (TopPop)'
    names_knn = 'ItemKNN (cosine, k=%d)' % KNN_NEIGHBORS

    results = {
        names_proposed: metrics_from_ranks(ranks_model),
        names_pop: metrics_from_ranks(ranks_pop),
        names_knn: metrics_from_ranks(ranks_knn),
    }

    beyond = {
        names_proposed: ba_model.result(catalog_size),
        names_pop: ba_pop.result(catalog_size),
        names_knn: ba_knn.result(catalog_size),
    }

    rmse_raw = float(np.sqrt(sq_err_raw / n_err_terms))
    mae_raw = float(abs_err_raw / n_err_terms)
    rmse_norm = float(np.sqrt(sq_err_norm / n_err_terms))
    mae_norm = float(abs_err_norm / n_err_terms)

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"  دامنه: {domain.upper()}   |   ارزیابی فقط روی مجموعه تست ({n_test} کاربر)")
    print(f"  پروتکل: leave-one-out، ۱ مثبت در برابر {NUM_NEG} منفی، seed={SEED}")
    print("=" * 78)

    metric_order = []
    for K in TOP_K_LIST:
        metric_order += [f'HR@{K}', f'NDCG@{K}', f'Precision@{K}', f'Recall@{K}']
    metric_order += ['MRR', 'MeanRank']

    names = list(results.keys())
    w0 = max(len(m) for m in metric_order) + 2
    header = 'Metric'.ljust(w0) + ''.join(n[:26].rjust(28) for n in names)
    print(header)
    print('-' * len(header))
    for m in metric_order:
        row = m.ljust(w0)
        for n in names:
            v = results[n][m]
            row += (f'{v:.4f}' if m == 'MeanRank' else f'{v * 100:.2f}%').rjust(28)
        print(row)
    print('-' * len(header))

    print("\n--- سنجه‌های خطای امتیاز (RMSE / MAE) ---")
    print("توجه: این مدل softmax روی کل واژگان آیتم تولید می‌کند و امتیاز صریح (rating) پیش‌بینی")
    print("      نمی‌کند؛ ستون Rating در داده اصلاً به مدل داده نشده است. بنابراین RMSE/MAE به معنای")
    print("      «خطای پیش‌بینی امتیاز» برای این معماری تعریف‌شدنی نیست و گزارش چنین عددی گمراه‌کننده است.")
    print("      آنچه زیر می‌آید صریحاً چیز دیگری است: خطای احتمالِ پیش‌بینی‌شده نسبت به هدف دودویی")
    print(f"      relevance ∈ {{0,1}} روی همان {NUM_NEG + 1} کاندید هر کاربر.")
    print(f"  RMSE (softmax خام در برابر relevance 0/1):            {rmse_raw:.6f}")
    print(f"  MAE  (softmax خام در برابر relevance 0/1):            {mae_raw:.6f}")
    print(f"  RMSE (softmax نرمال‌شده روی ۱۰۰ کاندید، در برابر 0/1): {rmse_norm:.6f}")
    print(f"  MAE  (softmax نرمال‌شده روی ۱۰۰ کاندید، در برابر 0/1): {mae_norm:.6f}")

    print("\n" + "=" * 78)
    print(f"  جدول ۲ — سنجه‌های فراتر از دقت (لیست top-{REC_LIST_SIZE} روی کل کاتالوگ، {n_test} کاربر)")
    print("=" * 78)
    print("  Coverage : نسبت آیتم‌های کاتالوگ که حداقل در یک لیست ظاهر شده‌اند")
    print("  Diversity: میانگین (۱ - شباهت کسینوسی) زوج‌های درون‌لیستی در فضای embedding آیتم")
    print("  Novelty  : میانگین -log2(p) آیتم‌های پیشنهادی (p = فراوانی آموزش، هموارسازی لاپلاس)")
    print("-" * 78)

    b_cols = [('Coverage', 'coverage'), ('Diversity', 'diversity'), ('Novelty (bits)', 'novelty_bits')]
    w_name = max(len(n) for n in beyond) + 2
    b_header = 'Method'.ljust(w_name) + ''.join(c.rjust(18) for c, _ in b_cols) + 'Unique items'.rjust(16)
    print(b_header)
    print('-' * len(b_header))
    for n in beyond:
        r = beyond[n]
        row = n.ljust(w_name)
        row += f"{r['coverage'] * 100:.2f}%".rjust(18)
        row += f"{r['diversity']:.4f}".rjust(18)
        row += f"{r['novelty_bits']:.4f}".rjust(18)
        row += f"{r['coverage_unique_items']} / {catalog_size}".rjust(16)
        print(row)
    print('-' * len(b_header))
    print(f"  یادآوری: Popularity مستقل از کاربر است، پس هر {n_test} کاربر لیست یکسانی می‌گیرند")
    print(f"           و Coverage آن ذاتاً روی {REC_LIST_SIZE}/{catalog_size} سقف می‌خورد.")
    print("           هیچ روشی آیتم‌های قبلاً دیده‌شده کاربر را فیلتر نمی‌کند (رویه یکسان برای هر سه).")
    print("=" * 78)

    payload = {
        'domain': domain,
        'protocol': {
            'name': 'leave-one-out, 1 positive vs %d sampled negatives' % NUM_NEG,
            'seed': SEED,
            'n_test_users': int(n_test),
            'n_train_samples': int(len(y_train)),
            'catalog_size_excluding_padding': int(catalog_size),
            'evaluated_on': 'test set only (never used during training or EarlyStopping)',
            'model_file': model_file,
            'retrained': False,
        },
        'ranking_metrics': results,
        'rating_error_metrics': {
            'applicable_as_rating_prediction': False,
            'why_not': ('The model outputs a softmax distribution over the item vocabulary and is trained '
                        'with sparse categorical cross-entropy for next-item ranking. It never receives or '
                        'predicts the explicit Rating column, so classical rating-prediction RMSE/MAE is '
                        'undefined for this architecture.'),
            'surrogate_definition': ('Error between the predicted probability of each of the %d candidates '
                                     'and a binary relevance target (1 for the held-out positive item, 0 for '
                                     'the %d sampled negatives), averaged over all candidate slots.'
                                     % (NUM_NEG + 1, NUM_NEG)),
            'RMSE_raw_softmax_vs_binary_relevance': rmse_raw,
            'MAE_raw_softmax_vs_binary_relevance': mae_raw,
            'RMSE_candidate_normalized_vs_binary_relevance': rmse_norm,
            'MAE_candidate_normalized_vs_binary_relevance': mae_norm,
        },
        'beyond_accuracy_metrics': {
            'recommendation_list_size': REC_LIST_SIZE,
            'scope': ('top-%d over the full catalog for every test user (padding index 0 excluded); computed '
                      'identically for the proposed model and both baselines' % REC_LIST_SIZE),
            'seen_item_filtering': ('none - no method filters out items already in the user\'s training '
                                    'history, so the list-generation procedure is identical across methods'),
            'coverage_definition': ('fraction of catalog items appearing in at least one test user\'s '
                                    'top-%d list' % REC_LIST_SIZE),
            'diversity_definition': ('mean intra-list dissimilarity: average over all item pairs in a list of '
                                     '(1 - cosine similarity) between the model\'s learned item embeddings, '
                                     'then averaged over users. No genre features are wired into the model, so '
                                     'the embedding space is used as the item similarity space. The SAME '
                                     'embedding space is used to score all three methods\' lists, so this '
                                     'measures list quality, not each method\'s own representation.'),
            'novelty_definition': ('mean self-information -log2(p_i) of recommended items, where '
                                   'p_i = (train_count_i + 1) / (total_train_interactions + catalog_size) '
                                   '(Laplace smoothing so that never-trained items stay finite).'),
            'popularity_caveat': ('TopPop is user-independent, so all test users receive the identical top-%d '
                                  'list; its coverage is therefore capped at %d/%d by construction.'
                                  % (REC_LIST_SIZE, REC_LIST_SIZE, catalog_size)),
            'per_method': beyond,
        },
        'baselines': {
            'Popularity (TopPop)': 'candidates ranked by their training-set frequency; ties broken by a fixed '
                                   'seed-42 random key',
            'ItemKNN (cosine, k=%d)' % KNN_NEIGHBORS:
                'item-based CF: cosine similarity on the binary user-item training matrix; each candidate is '
                'scored by the sum of its top-%d similarities to the items in the user\'s training history '
                '(self-similarity excluded)' % KNN_NEIGHBORS,
        },
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nنتایج کامل در '{out_file}' ذخیره شد.")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("خطا در اجرا. لطفاً دامنه را مشخص کنید.")
        print("مثال: python evaluate_full_metrics.py movie")
        print("   یا: python evaluate_full_metrics.py book")
        print("برای مدل دارای شاخه محتوا: python evaluate_full_metrics.py movie content")
