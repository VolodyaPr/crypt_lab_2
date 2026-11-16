import os

PASSWORD = "SecurityPassword27"

SECRET_MESSAGE = ("Ингредиенты (примерно на 12–15 блинов):"
                  "Молоко — 500 мл"
                  "Мука пшеничная — 250 г — 3 шт."
                  "Сахар — 2 ст. л."
                  "Соль — 1/2 ч. л."
                  "Растительное или растопленное сливочное масло — 2–3 ст.л. в тесто + немного для смазывания сковороды.")

SALT_LEN = 16
NONCE_LEN = 12
TAG_LEN = 16
CRYPTO_OVERHEAD = SALT_LEN + NONCE_LEN + TAG_LEN
KDF_ITER = 100_000

IMGS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'imgs'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))