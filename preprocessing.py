import re
import string
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Converts raw email/SMS text into a cleaned string where spam signals
    (URLs, money amounts, phone numbers, ALL-CAPS words, repeated punctuation)
    are replaced with meaningful tokens instead of being stripped entirely.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self._clean(str(text)) for text in X]

    @staticmethod
    def _clean(text):
        text = str(text)

        # Replace URLs — covers http(s), www, and bare domains with common TLDs
        text = re.sub(
            r'https?://\S+'
            r'|www\.\S+'
            r'|\b\S+\.(?:com|net|org|co\.uk|info|biz|win|club|xyz|tv|io|co)\b',
            ' urltoken ',
            text,
            flags=re.IGNORECASE,
        )

        # Replace email addresses
        text = re.sub(r'\S+@\S+\.\S+', ' emailtoken ', text)

        # Mark ALL-CAPS words (strong spam signal) before lowercasing
        text = re.sub(r'\b([A-Z]{2,})\b', r'\1 capstoken ', text)

        text = text.lower()

        # Replace money amounts  (e.g. $1,000  £500  500$)
        text = re.sub(
            r'[$\xa3€]\s*[\d,]+(?:\.\d+)?|[\d,]+(?:\.\d+)?\s*[$\xa3€]',
            ' moneytoken ',
            text,
        )

        # Repeated exclamation / question marks (urgency indicators)
        text = re.sub(r'!{2,}', ' multiexclaim ', text)
        text = re.sub(r'\?{2,}', ' multiquestion ', text)

        # Replace all remaining numbers (phone numbers, short codes, etc.)
        text = re.sub(r'\d+', ' numtoken ', text)

        # Strip remaining punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))

        # Normalize whitespace
        return re.sub(r'\s+', ' ', text).strip()
