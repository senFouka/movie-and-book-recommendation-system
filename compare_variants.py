"""
مقایسه بازوی پایه (بدون شاخه محتوا) با بازوی محتوایی، بر اساس فایل‌های
full_metrics تولیدشده توسط evaluate_full_metrics.py.

هیچ عددی در این اسکریپت نوشته نشده است؛ همه مقادیر از روی فایل‌های JSON خوانده
می‌شوند. اگر فایلی وجود نداشته باشد، همان ستون خالی گزارش می‌شود.

اجرا:
    python compare_variants.py                 # هر دو دامنه
    python compare_variants.py movie book      # دامنه‌های دلخواه
"""
import json
import os
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        _stream.reconfigure(encoding='utf-8', errors='replace')

DEFAULT_DOMAINS = ('movie', 'book')

# (برچسب, مسیر در JSON, قالب‌بندی)
METRICS = (
    ('HR@10',          ('ranking_metrics', 'HR@10'),                      'percent'),
    ('NDCG@10',        ('ranking_metrics', 'NDCG@10'),                    'percent'),
    ('MRR',            ('ranking_metrics', 'MRR'),                        'percent'),
    ('Coverage',       ('beyond_accuracy_metrics', 'coverage'),           'percent'),
    ('Diversity',      ('beyond_accuracy_metrics', 'diversity'),          'float'),
    ('Novelty (bits)', ('beyond_accuracy_metrics', 'novelty_bits'),       'float'),
)

VARIANTS = (('baseline (seeded)', ''), ('content', 'content'))


def metrics_path(domain, variant):
    suffix = f'_{variant}' if variant else ''
    return os.path.join(f'{domain}_data', f'full_metrics{suffix}.json')


def find_proposed_key(section):
    """کلید روش پیشنهادی را پیدا می‌کند بدون آنکه برچسب کامل hardcode شود."""
    for key in section:
        if 'proposed' in key.lower():
            return key
    return None


def read_variant(domain, variant):
    path = metrics_path(domain, variant)
    if not os.path.exists(path):
        return None, path, f"فایل پیدا نشد"
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)

    ranking = payload.get('ranking_metrics', {})
    beyond = payload.get('beyond_accuracy_metrics', {}).get('per_method', {})
    r_key, b_key = find_proposed_key(ranking), find_proposed_key(beyond)
    if r_key is None or b_key is None:
        return None, path, "کلید روش پیشنهادی در فایل نیست"

    values = {}
    for label, (section, field), _fmt in METRICS:
        source = ranking[r_key] if section == 'ranking_metrics' else beyond[b_key]
        values[label] = source.get(field)

    meta = payload.get('protocol', {})

    # مانیفست آموزش (در صورت وجود): تأیید اینکه این ستون واقعاً از همان بازو آمده است
    suffix = f'_{variant}' if variant else ''
    manifest_path = os.path.join(f'{domain}_data', f'run_manifest{suffix}.json')
    manifest = None
    if os.path.exists(manifest_path):
        with open(manifest_path, encoding='utf-8') as f:
            manifest = json.load(f)

    return {
        'values': values,
        'model_file': meta.get('model_file'),
        'n_test_users': meta.get('n_test_users'),
        'catalog_size': meta.get('catalog_size_excluding_padding'),
        'eval_seed': meta.get('seed'),
        'method_label': r_key,
        'manifest': manifest,
        'manifest_path': manifest_path,
    }, path, None


def fmt(value, kind):
    if value is None:
        return '—'
    if kind == 'percent':
        return f'{value * 100:.2f}%'
    return f'{value:.4f}'


def delta(base, new, kind):
    if base is None or new is None:
        return '—'
    diff = new - base
    if kind == 'percent':
        return f'{diff * 100:+.2f} pp'
    return f'{diff:+.4f}'


def compare(domain):
    print('=' * 78)
    print(f'دامنه: {domain.upper()}')
    print('=' * 78)

    columns = []
    for label, variant in VARIANTS:
        result, path, error = read_variant(domain, variant)
        if error:
            print(f"هشدار: ستون «{label}» در دسترس نیست → {path} ({error})")
        columns.append((label, result))

    if all(result is None for _, result in columns):
        print('هیچ فایل متریکی برای این دامنه پیدا نشد؛ ابتدا evaluate_full_metrics.py را اجرا کنید.\n')
        return None

    # --- شناسنامه هر ستون + بررسی اینکه مقایسه واقعاً منصفانه است ---
    training_seeds, expected_arms = set(), {'baseline (seeded)': False, 'content': True}
    for label, result in columns:
        if result is None:
            continue
        manifest = result['manifest']
        print(f"  {label:<20} → {result['model_file']} "
              f"(test users={result['n_test_users']}, catalog={result['catalog_size']}, "
              f"eval seed={result['eval_seed']})")
        if manifest is None:
            print(f"{'':<22}  هشدار: مانیفست آموزش پیدا نشد ({result['manifest_path']}) — "
                  f"نمی‌توان تأیید کرد این خروجی با بذر جدید و همین بازو تولید شده است.")
            continue
        training_seeds.add(manifest.get('training_seed'))
        print(f"{'':<22}  بازو={manifest.get('arm')} | use_content={manifest.get('use_content')} | "
              f"training seed={manifest.get('training_seed')} | "
              f"epochs={manifest.get('epochs_run')} | "
              f"params={manifest.get('params', {}).get('total'):,} | "
              f"پایان={manifest.get('finished_at')}")
        if manifest.get('use_content') != expected_arms.get(label):
            print(f"{'':<22}  هشدار جدی: این ستون باید use_content={expected_arms.get(label)} "
                  f"می‌بود اما مانیفست {manifest.get('use_content')} را نشان می‌دهد!")

    if len(training_seeds) > 1:
        print(f"\n  هشدار جدی: بذرهای آموزش یکسان نیستند {sorted(training_seeds)} — "
              f"مقایسه از نقطه شروع یکسان انجام نشده است.")
    elif len(training_seeds) == 1:
        print(f"\n  [check] هر دو بازو از بذر آموزش یکسان شروع شده‌اند: {training_seeds.pop()}")
    print()

    head = f"{'Metric':<16}" + ''.join(f'{label:>22}' for label, _ in columns) + f"{'Δ (content−base)':>22}"
    print(head)
    print('-' * len(head))

    table = {}
    for label, (_section, _field), kind in METRICS:
        row = f'{label:<16}'
        cells = []
        for _col_label, result in columns:
            value = result['values'][label] if result else None
            cells.append(value)
            row += f'{fmt(value, kind):>22}'
        row += f'{delta(cells[0], cells[1], kind):>22}'
        print(row)
        table[label] = {columns[i][0]: cells[i] for i in range(len(columns))}

    print()
    return table


if __name__ == '__main__':
    domains = sys.argv[1:] or list(DEFAULT_DOMAINS)
    summary = {}
    for domain in domains:
        result = compare(domain)
        if result:
            summary[domain] = result

    if summary:
        out_file = 'variant_comparison.json'
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"جدول مقایسه در '{out_file}' ذخیره شد.")
