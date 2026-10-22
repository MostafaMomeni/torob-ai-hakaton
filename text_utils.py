import re
import unicodedata

# Minimal Persian stopwords set for token filtering
PERSIAN_STOPWORDS = set(
    [
        "و",
        "در",
        "به",
        "از",
        "که",
        "برای",
        "با",
        "را",
        "این",
        "آن",
        "تا",
        "یا",
        "هم",
        "روی",
        "بر",
        "یک",
        "های",
    ]
)


def normalize_persian_text(s: str) -> str:
    """Normalize Persian/Arabic text for matching while preserving display original.

    - Normalize Arabic/Persian variants: ي->ی, ك->ک, etc.
    - Remove zero-width joiners/non-joiners and control chars.
    - Collapse whitespace.
    - Unicode normalize (NFKC) to handle forms.
    - Convert Arabic-Indic digits to European digits.
    """
    if s is None:
        return ""
    # Ensure str
    s = str(s)
    # Unicode normalization
    s = unicodedata.normalize("NFKC", s)
    # Map Arabic variants to Persian equivalents
    trans_map = {
        '\u064a': '\u06cc',  # ي -> ی
        '\u0643': '\u06a9',  # ك -> ک
        '\u06c0': '\u06c0',
    }
    # digits: Arabic-Indic ٠١٢٣٤٥٦٧٨٩ (U+0660..U+0669) and Eastern Arabic-Indic U+06F0..U+06F9
    # map them to ASCII digits
    s = s.translate({
        0x0660: ord('0'), 0x0661: ord('1'), 0x0662: ord('2'), 0x0663: ord('3'), 0x0664: ord('4'),
        0x0665: ord('5'), 0x0666: ord('6'), 0x0667: ord('7'), 0x0668: ord('8'), 0x0669: ord('9'),
        0x06F0: ord('0'), 0x06F1: ord('1'), 0x06F2: ord('2'), 0x06F3: ord('3'), 0x06F4: ord('4'),
        0x06F5: ord('5'), 0x06F6: ord('6'), 0x06F7: ord('7'), 0x06F8: ord('8'), 0x06F9: ord('9'),
    })
    # replace specific Arabic letters with Persian ones
    s = s.replace('ي', 'ی').replace('ك', 'ک')
    # Remove zero-width and diacritics that could break joins
    # common zero-width joiner/non-joiner: \u200c (ZWNJ), \u200d (ZWJ)
    s = re.sub(r'[\u200c\u200d\u200e\u200f]', ' ', s)
    # Remove diacritics (harakat) in Arabic block
    s = re.sub(r'[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e8\u06ea-\u06ed]', '', s)
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Lowercasing is safe for Latin parts; Persian has no case but do it for safety
    s = s.lower()
    return s


def tokenize_persian(s: str, remove_stopwords: bool = True) -> list:
    """Split normalized Persian string into tokens (words) and optionally remove stopwords."""
    n = normalize_persian_text(s)
    # split on whitespace and punctuation
    toks = re.split(r"[\s،,\.;:\-()\[\]/]+", n)
    toks = [t for t in toks if t]
    if remove_stopwords:
        toks = [t for t in toks if t not in PERSIAN_STOPWORDS]
    return toks
