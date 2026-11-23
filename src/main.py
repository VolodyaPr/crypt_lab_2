import os
import csv
import argparse
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import PASSWORD, IMGS_DIR, RESULTS_DIR, CRYPTO_OVERHEAD, SECRET_MESSAGE
from src.crypto_utils import encrypt_message, decrypt_message
from src.stego import embed_data, extract_data
from src.metrics import calculate_metrics, lsb_distribution, chi2_lsb


def print_table(headers, rows, title):
    print(f"\n--- {title} ---")
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    fmt_str = " | ".join([f"{{:<{w}}}" for w in col_widths])
    print(fmt_str.format(*headers))
    print("-+-".join(["-" * w for w in col_widths]))
    for row in rows:
        print(fmt_str.format(*row))


def save_diff_map(cover, stego, path, gain=255):
    diff = np.abs(cover[..., :3].astype(np.int16) - stego[..., :3].astype(np.int16))
    diff_rgb = np.clip(diff * gain, 0, 255).astype(np.uint8)
    h, w, _ = diff_rgb.shape
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([diff_rgb, alpha], axis=2)).save(path)


def save_hist(cover, stego, path):
    h_cover = np.bincount(cover[..., :3].flatten(), minlength=256)
    h_stego = np.bincount(stego[..., :3].flatten(), minlength=256)
    diff = h_stego - h_cover

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
    bins = np.arange(257)

    ax1.hist(cover[..., :3].flatten(), bins=bins, alpha=0.6, label='Оригинал', color='blue')
    ax1.hist(stego[..., :3].flatten(), bins=bins, alpha=0.6, label='Стегограмма', color='salmon')
    ax1.set_title('Оригинал/Стегограммы', fontsize=12)
    ax1.set_xlabel('байт')
    ax1.set_ylabel('Количество')
    ax1.set_xlim(0, 256)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.6)

    colors = ['tab:blue' if val > 0 else 'salmon' for val in diff]
    ax2.bar(range(256), diff, color=colors, alpha=0.8)
    ax2.set_title('Изменение частот (Δ = Stego - Cover)', fontsize=12)
    ax2.set_xlabel('байт')
    ax2.set_ylabel('изменение')
    ax2.set_xlim(-0.5, 255.5)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True, linestyle='--', alpha=0.6)

    legend_elements = [Patch(facecolor='salmon', label='Убыль Частоты'),
                       Patch(facecolor='tab:blue', label='Прирост частоты')]
    ax2.legend(handles=legend_elements, loc='upper right')
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def get_payload_bytes(target_size):
    """Безопасно формируем байты заданного размера без разрыва UTF-8 символов"""
    if target_size <= 0:
        return b""

    base_bytes = SECRET_MESSAGE.encode('utf-8')
    if len(base_bytes) == 0:
        return b' ' * target_size

    repeat_count = (target_size // len(base_bytes)) + 1
    long_bytes = base_bytes * repeat_count
    truncated_bytes = long_bytes[:target_size]

    safe_str = truncated_bytes.decode('utf-8', errors='ignore')
    result_bytes = safe_str.encode('utf-8')

    if len(result_bytes) < target_size:
        result_bytes += b' ' * (target_size - len(result_bytes))
    return result_bytes[:target_size]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Стеганография LSB")
    parser.add_argument('images', nargs='*', default=['gradient.png', 'checkerboard.png', 'IMG_1.png'])
    args = parser.parse_args()

    os.makedirs(IMGS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_csv_rows = []
    headers = ['Image', 'Метод', 'Заполнение', 'PSNR, дБ', 'SSIM', 'Changed Ratio', 'P(LSB=1)', 'χ²', 'p-value']

    for cover_filename in args.images:
        cover_path = os.path.join(IMGS_DIR, cover_filename)
        base_name = os.path.splitext(cover_filename)[0]

        if not os.path.exists(cover_path):
            print(f"Файл {cover_path} не найден, пропускаем")
            continue

        cover = np.array(Image.open(cover_path).convert('RGBA'), dtype=np.uint8)
        height, width, _ = cover.shape
        max_capacity_bytes = (height * width * 3) // 8 - 4

        print(f"\n{'=' * 95}")
        print(f"Обработка изображения: {cover_filename}")
        print(f"{'=' * 95}")

        size_1 = int(max_capacity_bytes * 0.01)
        size_5 = int(max_capacity_bytes * 0.05)

        # Формируем сообщения. Одинаковый объем данных берем базовый текст для шифрования ровно на 44 байта меньше
        # а для PlainLSB берем этот же текст, но дополняем его до полного размера
        src_msg_1 = get_payload_bytes(size_1 - CRYPTO_OVERHEAD)
        plain_payload_1 = src_msg_1 + b' ' * CRYPTO_OVERHEAD
        cipher_payload_1 = encrypt_message(src_msg_1, PASSWORD)

        src_msg_5 = get_payload_bytes(size_5 - CRYPTO_OVERHEAD)
        plain_payload_5 = src_msg_5 + b' ' * CRYPTO_OVERHEAD
        cipher_payload_5 = encrypt_message(src_msg_5, PASSWORD)

        assert len(plain_payload_1) == len(cipher_payload_1)
        assert len(plain_payload_5) == len(cipher_payload_5)

        experiments = [
            ('PlainLSB', '1%', plain_payload_1, src_msg_1, False),
            ('PlainCipherLSB', '1%', cipher_payload_1, src_msg_1, True),
            ('RandLSB', '1%', cipher_payload_1, src_msg_1, True),
            ('PlainLSB', '5%', plain_payload_5, src_msg_5, False),
            ('PlainCipherLSB', '5%', cipher_payload_5, src_msg_5, True),
            ('RandLSB', '5%', cipher_payload_5, src_msg_5, True),
        ]

        # Cover
        c_p1 = lsb_distribution(cover)
        c_chi2, c_pval = chi2_lsb(cover)
        all_csv_rows.append([
            base_name, 'Cover', '0%', '+inf', '1.0000', '0.0000', f"{c_p1:.4f}", f"{c_chi2:.2f}", f"{c_pval:.4f}"
        ])

        last_stego_random_loaded = None

        for method, fill_pct, payload, original_msg, is_cipher in experiments:
            # 1. Встраивание
            if method == 'RandLSB':
                stego = embed_data(cover, payload, mode='RandLSB', password=PASSWORD)
            else:
                stego = embed_data(cover, payload, mode=method)

            # 2. Сохранение
            if method == 'PlainLSB':
                method_tag = 'plainLsb'
            elif method == 'PlainCipherLSB':
                method_tag = 'plainLsbAes'
            else:
                method_tag = 'encryptedRandomLsb'

            stego_filename = f"{base_name}-{method_tag}_{fill_pct.replace('%', '')}-percent.png"
            stego_filepath = os.path.join(RESULTS_DIR, stego_filename)
            Image.fromarray(stego).save(stego_filepath)

            # 3. Извлечение из файла
            stego_loaded = np.array(Image.open(stego_filepath).convert('RGBA'), dtype=np.uint8)

            # Запоминаем последний загруженный RandLSB для графиков
            if method == 'RandLSB' and fill_pct == '5%':
                last_stego_random_loaded = stego_loaded

            if method == 'RandLSB':
                extracted_payload = extract_data(stego_loaded, mode='RandLSB', password=PASSWORD)
            else:
                extracted_payload = extract_data(stego_loaded, mode=method)

            # 4. Расшифровка и проверка
            if is_cipher:
                extracted_msg = decrypt_message(extracted_payload, PASSWORD)
                assert extracted_msg == original_msg, f"Ошибка расшифровки {method} ({fill_pct})!"
            else:
                assert extracted_payload == payload, f"Ошибка извлечения {method} ({fill_pct})!"

            print(f"  [+] {method:<15} | {fill_pct:<3} | Встроено/Извлечено. Сохранено: {stego_filename}")

            # 5. Метрики (считаем по stego_loaded)
            m = calculate_metrics(cover, stego_loaded)
            p1 = lsb_distribution(stego_loaded)
            chi2_val, p_val = chi2_lsb(stego_loaded)

            all_csv_rows.append([
                base_name, method, fill_pct,
                f"{m['psnr']:.2f}", f"{m['ssim']:.4f}", f"{m['ratio']:.4f}",
                f"{p1:.4f}", f"{chi2_val:.2f}", f"{p_val:.4f}"
            ])

        # Визуализация
        if last_stego_random_loaded is not None:
            save_diff_map(cover, last_stego_random_loaded, os.path.join(RESULTS_DIR, f'{base_name}_difference.png'))
            save_hist(cover, last_stego_random_loaded, os.path.join(RESULTS_DIR, f'{base_name}_histogram.png'))

    # Сохранение таблицы в CSV
    csv_path = os.path.join(RESULTS_DIR, 'all_metrics.csv')
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(all_csv_rows)

    print_table(headers, all_csv_rows, 'Сводная таблица метрик')
    print(f"\nВсе данные сохранены в {csv_path}")
