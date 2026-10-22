import re
import unicodedata

PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "برای", "با", "را", "این", "آن", "تا",
    "یا", "هم", "روی", "بر", "یک", "های", "می", "است", "بود",
}


def normalize_persian_text(value):
    """Normalize Persian/Arabic variants, digits, punctuation and whitespace for matching."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.translate(str.maketrans({
        "ي": "ی", "ى": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه",
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
        "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
        "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
    }))
    text = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", text)
    text = re.sub(r"[\u200c\u200d\u200e\u200f\ufeff]", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_persian(value, remove_stopwords=True):
    tokens = normalize_persian_text(value).split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in PERSIAN_STOPWORDS]
    return tokens
