"""
models.py — ML models for VoiceFinance
────────────────────────────────────────────────────────────────────────────────
1. IntentClassifier   — TF-IDF + MultinomialNB  → classifies voice command intent
2. CategoryClassifier — TF-IDF + LogisticRegression → detects expense/income category
3. SalesForecaster    — LinearRegression on rolling 30-day data → 7-day prediction
4. AnomalyDetector    — Z-score on daily totals → flags unusual spending days
────────────────────────────────────────────────────────────────────────────────
All models are trained on built-in seed data and retrain incrementally from the
user's own transaction history stored in SQLite.
"""

import re, sqlite3
import numpy as np
from datetime import datetime, timedelta

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import NotFittedError
from scipy import stats

# ── Seed training data ────────────────────────────────────────────────────────
# Intent: maps raw utterance → action label
INTENT_SEED = [
    # income
    ("sold 200 rice",           "income"),
    ("sale of 500",             "income"),
    ("received 1000 payment",   "income"),
    ("earned 300 today",        "income"),
    ("got 250 from customer",   "income"),
    ("income 400",              "income"),
    ("sell milk for 50",        "income"),
    ("sold biscuits 100",       "income"),
    ("விற்றது 200",              "income"),
    ("வந்தது 500",               "income"),
    ("கிடைத்தது 300",            "income"),
    # expense
    ("bought milk for 300",     "expense"),
    ("spent 500 on groceries",  "expense"),
    ("paid 200 for oil",        "expense"),
    ("purchased rice 400",      "expense"),
    ("expense 150",             "expense"),
    ("cost 600 electricity",    "expense"),
    ("வாங்கினேன் 300",           "expense"),
    ("செலவு 200",                "expense"),
    ("கொடுத்தேன் 100",           "expense"),
    # credit
    ("ravi took 200 credit",    "credit"),
    ("kumar borrowed 500",      "credit"),
    ("credit 300 to suresh",    "credit"),
    ("priya owes 150",          "credit"),
    ("give credit 400 to mohan","credit"),
    ("கடன் 200 ரவி",             "credit"),
    ("எடுத்தான் 300",            "credit"),
    # inventory_add
    ("add 10 rice bags",        "inventory_add"),
    ("stock 20 milk packets",   "inventory_add"),
    ("added 50 biscuits",       "inventory_add"),
    ("received 100 sugar kg",   "inventory_add"),
    ("சேர் 10 அரிசி",            "inventory_add"),
    ("வைத்தேன் 20 பால்",         "inventory_add"),
    # report
    ("show today report",       "report"),
    ("summary for this week",   "report"),
    ("today sales report",      "report"),
    ("show me the report",      "report"),
    ("இன்று அறிக்கை",            "report"),
    ("கணக்கு காட்டு",            "report"),
]

# Category: maps text → category label
CATEGORY_SEED = [
    ("rice wheat flour dal sugar salt",     "Groceries"),
    ("bought rice 5 kg",                    "Groceries"),
    ("sugar oil salt grocery",              "Groceries"),
    ("tomato onion potato carrot greens",   "Vegetables"),
    ("vegetable market onion",              "Vegetables"),
    ("milk curd butter cheese dairy",       "Dairy"),
    ("milk packet curd cup",                "Dairy"),
    ("tea coffee juice drink water soda",   "Beverages"),
    ("tea powder coffee pack",              "Beverages"),
    ("biscuit chips snack chocolate candy", "Snacks"),
    ("biscuit pack chips",                  "Snacks"),
    ("electricity bill rent internet",      "Utilities"),
    ("electricity bill paid",               "Utilities"),
    ("transport fuel petrol auto bus",      "Transport"),
    ("petrol vehicle fuel",                 "Transport"),
    ("medicine tablet syrup pharmacy",      "Medicine"),
    ("tablet medicine bought",              "Medicine"),
    ("soap shampoo toothpaste detergent",   "Personal Care"),
    ("soap bar shampoo bottle",             "Personal Care"),
    ("pen notebook stationery paper",       "Stationery"),
    ("notebook pen bought",                 "Stationery"),
]


# ── 1. Intent Classifier ──────────────────────────────────────────────────────
class IntentClassifier:
    """
    TF-IDF (char n-grams 2-4) + MultinomialNB pipeline.
    Char n-grams handle Tamil script and partial words well.
    Retrains from user history when enough labelled data exists.
    """
    LABELS = ["income", "expense", "credit", "inventory_add", "report", "unknown"]

    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(2, 4),
                min_df=1, sublinear_tf=True
            )),
            ('clf', MultinomialNB(alpha=0.5))
        ])
        self._trained = False
        self._train_seed()

    def _train_seed(self):
        texts  = [t for t, _ in INTENT_SEED]
        labels = [l for _, l in INTENT_SEED]
        self.pipeline.fit(texts, labels)
        self._trained = True

    def retrain(self, extra_texts, extra_labels):
        """Merge seed + user data and retrain."""
        texts  = [t for t, _ in INTENT_SEED] + list(extra_texts)
        labels = [l for _, l in INTENT_SEED] + list(extra_labels)
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> dict:
        """Returns {'intent': str, 'confidence': float}"""
        if not self._trained:
            return {'intent': 'unknown', 'confidence': 0.0}
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        idx = int(np.argmax(proba))
        return {
            'intent':     classes[idx],
            'confidence': round(float(proba[idx]), 3),
            'all_scores': {c: round(float(p), 3) for c, p in zip(classes, proba)}
        }


# ── 2. Category Classifier ────────────────────────────────────────────────────
class CategoryClassifier:
    """
    TF-IDF (word unigrams + bigrams) + LogisticRegression.
    Predicts the expense/income category from free text.
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 2),
                min_df=1, sublinear_tf=True, lowercase=True
            )),
            ('clf', LogisticRegression(max_iter=500, C=2.0, solver='lbfgs'))
        ])
        self._trained = False
        self._train_seed()

    def _train_seed(self):
        texts  = [t for t, _ in CATEGORY_SEED]
        labels = [l for _, l in CATEGORY_SEED]
        self.pipeline.fit(texts, labels)
        self._trained = True

    def retrain(self, extra_texts, extra_labels):
        texts  = [t for t, _ in CATEGORY_SEED] + list(extra_texts)
        labels = [l for _, l in CATEGORY_SEED] + list(extra_labels)
        self.pipeline.fit(texts, labels)

    def predict(self, text: str) -> dict:
        if not self._trained:
            return {'category': 'General', 'confidence': 0.0}
        proba   = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        idx     = int(np.argmax(proba))
        conf    = float(proba[idx])
        # Fall back to 'General' if confidence is too low
        category = classes[idx] if conf >= 0.25 else 'General'
        return {
            'category':   category,
            'confidence': round(conf, 3)
        }


# ── 3. Sales Forecaster ───────────────────────────────────────────────────────
class SalesForecaster:
    """
    LinearRegression on the last N days of income data.
    Features: [day_index, day_of_week, rolling_3day_avg]
    Predicts next 7 days individually and returns total + daily breakdown.
    """
    def __init__(self):
        self.model   = LinearRegression()
        self._fitted = False

    def _build_features(self, values: list) -> np.ndarray:
        n = len(values)
        X = []
        for i, v in enumerate(values):
            dow   = i % 7                                          # day of week proxy
            r3    = np.mean(values[max(0, i-2):i+1])              # 3-day rolling avg
            r7    = np.mean(values[max(0, i-6):i+1])              # 7-day rolling avg
            trend = (values[i] - values[max(0, i-3)]) / 4 if i >= 3 else 0
            X.append([i, dow, r3, r7, trend])
        return np.array(X, dtype=float)

    def fit(self, daily_income: list):
        """Train on a list of daily income values (oldest first)."""
        if len(daily_income) < 5:
            self._fitted = False
            return
        X = self._build_features(daily_income)
        y = np.array(daily_income, dtype=float)
        self.model.fit(X, y)
        self._fitted = True

    def predict_next_7(self, daily_income: list) -> dict:
        """
        Returns {'total': float, 'daily': [7 floats], 'confidence': str}
        """
        if not self._fitted or len(daily_income) < 5:
            # Simple fallback: weighted average
            non_zero = [v for v in daily_income if v > 0]
            if not non_zero:
                return {'total': 0, 'daily': [0]*7, 'confidence': 'low'}
            avg = float(np.mean(non_zero))
            return {'total': round(avg * 7, 2), 'daily': [round(avg, 2)]*7, 'confidence': 'low'}

        n      = len(daily_income)
        values = list(daily_income)
        preds  = []
        for step in range(7):
            i   = n + step
            dow = i % 7
            r3  = float(np.mean(values[-3:]))
            r7  = float(np.mean(values[-7:]))
            trend = float((values[-1] - values[-4]) / 4) if len(values) >= 4 else 0
            x   = np.array([[i, dow, r3, r7, trend]])
            p   = max(0.0, float(self.model.predict(x)[0]))
            preds.append(round(p, 2))
            values.append(p)

        # Confidence based on R² of training fit
        X_train = self._build_features(daily_income)
        y_train = np.array(daily_income, dtype=float)
        r2      = self.model.score(X_train, y_train)
        conf    = 'high' if r2 > 0.7 else ('medium' if r2 > 0.4 else 'low')

        return {
            'total':      round(sum(preds), 2),
            'daily':      preds,
            'r2':         round(float(r2), 3),
            'confidence': conf
        }


# ── 4. Anomaly Detector ───────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Z-score based anomaly detection on daily expense totals.
    Flags days where spending is unusually high (z > threshold).
    Also detects sudden income drops.
    """
    Z_THRESHOLD = 2.0   # flag if z-score > 2 (top ~2.3% of distribution)

    def detect_expense_anomalies(self, daily_expenses: list) -> list:
        """
        Returns list of {'day_index': int, 'value': float, 'z_score': float, 'message': str}
        """
        if len(daily_expenses) < 4:
            return []
        arr = np.array(daily_expenses, dtype=float)
        z   = np.abs(stats.zscore(arr))
        anomalies = []
        for i, (val, zs) in enumerate(zip(arr, z)):
            if zs > self.Z_THRESHOLD and val > 0:
                anomalies.append({
                    'day_index': i,
                    'value':     round(float(val), 2),
                    'z_score':   round(float(zs), 2),
                    'message':   f"Unusually high expense on day {i+1}: ₹{val:.0f} (z={zs:.1f})"
                })
        return anomalies

    def detect_income_drop(self, daily_income: list) -> dict | None:
        """
        Detects if the most recent day's income dropped significantly vs rolling avg.
        Returns a warning dict or None.
        """
        if len(daily_income) < 4:
            return None
        arr     = np.array(daily_income, dtype=float)
        recent  = arr[-1]
        history = arr[:-1]
        avg     = float(np.mean(history[history > 0])) if np.any(history > 0) else 0
        if avg > 0 and recent < avg * 0.4:
            drop_pct = round((1 - recent / avg) * 100, 1)
            return {
                'type':    'income_drop',
                'value':   round(float(recent), 2),
                'avg':     round(avg, 2),
                'drop_pct': drop_pct,
                'message': f"Income dropped {drop_pct}% vs recent average (₹{avg:.0f})"
            }
        return None

    def budget_burn_rate(self, daily_expenses: list, monthly_budget: float = 0) -> dict:
        """
        Calculates current burn rate and projects month-end expense.
        """
        if not daily_expenses:
            return {'burn_rate': 0, 'projected_monthly': 0, 'status': 'ok'}
        arr       = np.array(daily_expenses, dtype=float)
        burn_rate = float(np.mean(arr[arr > 0])) if np.any(arr > 0) else 0
        projected = round(burn_rate * 30, 2)
        status    = 'ok'
        if monthly_budget > 0:
            if projected > monthly_budget * 1.2:
                status = 'danger'
            elif projected > monthly_budget:
                status = 'warning'
        return {
            'burn_rate':         round(burn_rate, 2),
            'projected_monthly': projected,
            'status':            status
        }


# ── Model registry (singleton instances) ─────────────────────────────────────
_intent_clf   = IntentClassifier()
_category_clf = CategoryClassifier()
_forecaster   = SalesForecaster()
_anomaly_det  = AnomalyDetector()


def get_intent_classifier()   -> IntentClassifier:   return _intent_clf
def get_category_classifier() -> CategoryClassifier: return _category_clf
def get_forecaster()          -> SalesForecaster:    return _forecaster
def get_anomaly_detector()    -> AnomalyDetector:    return _anomaly_det


# ── Retrain from DB ───────────────────────────────────────────────────────────
def retrain_from_db(db_path: str, user_id: int):
    """
    Pull labelled transactions from SQLite and retrain intent + category models.
    Called after each new transaction is added.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        'SELECT type, category, description FROM transactions WHERE user_id=? AND description != ""',
        (user_id,)
    ).fetchall()
    conn.close()

    if len(rows) < 5:   # not enough data yet
        return

    intent_texts, intent_labels     = [], []
    category_texts, category_labels = [], []

    for r in rows:
        desc = r['description'].strip()
        if not desc:
            continue
        # Intent labels: map type to intent
        intent_map = {'income': 'income', 'expense': 'expense'}
        if r['type'] in intent_map:
            intent_texts.append(desc)
            intent_labels.append(intent_map[r['type']])
        # Category labels
        if r['category'] and r['category'] != 'General':
            category_texts.append(desc)
            category_labels.append(r['category'])

    if len(intent_texts) >= 3:
        _intent_clf.retrain(intent_texts, intent_labels)
    if len(category_texts) >= 3:
        _category_clf.retrain(category_texts, category_labels)


# ── Convenience: full NLP parse using models ──────────────────────────────────
def model_parse(text: str, fallback_fn) -> dict:
    """
    Uses IntentClassifier + CategoryClassifier to parse a voice command.
    Falls back to regex NLP if model confidence is below threshold.
    Returns the same dict shape as nlp_parse().
    """
    import re as _re

    raw  = text
    low  = text.lower()
    nums = _re.findall(r'\d+(?:\.\d+)?', low)
    amount = float(nums[0]) if nums else 0

    intent_result   = _intent_clf.predict(low)
    category_result = _category_clf.predict(low)

    intent     = intent_result['intent']
    intent_conf = intent_result['confidence']
    category   = category_result['category']

    # If model is not confident enough, fall back to regex
    if intent_conf < 0.40:
        result = fallback_fn(text)
        # Still upgrade the category if model is more confident
        if category_result['confidence'] > 0.50 and result.get('category','General') == 'General':
            result['category'] = category
        result['model_used']       = False
        result['intent_confidence'] = intent_conf
        return result

    base = {
        'text':             raw,
        'amount':           amount,
        'model_used':       True,
        'intent_confidence': intent_conf,
        'category_confidence': category_result['confidence'],
    }

    if intent == 'income':
        return {**base, 'action': 'income',  'category': category}
    if intent == 'expense':
        return {**base, 'action': 'expense', 'category': category}
    if intent == 'credit':
        # Extract customer name via regex even when model handles intent
        name = _extract_name_model(low)
        return {**base, 'action': 'credit', 'customer': name}
    if intent == 'inventory_add':
        product, qty = _extract_product_qty_model(low)
        return {**base, 'action': 'inventory_add', 'product': product, 'quantity': qty or amount}
    if intent == 'report':
        return {**base, 'action': 'report'}

    return {**base, 'action': 'unknown'}


def _extract_name_model(text: str) -> str:
    stop = {'took','credit','rupees','rs','for','கடன்','எடுத்தான்','amount','paid','owes','borrowed'}
    nums = set(re.findall(r'\d+', text))
    words = [w.capitalize() for w in text.split()
             if w not in stop and w not in nums and len(w) > 2]
    return words[0] if words else 'Customer'


def _extract_product_qty_model(text: str) -> tuple:
    stop = {'add','added','stock','packets','packet','units','kg','litre','liters',
            'ml','nos','bags','bag','box','boxes','சேர்','வைத்தேன்'}
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    qty  = float(nums[0]) if nums else 0
    words = [w for w in text.split() if w not in stop and not w.replace('.','').isdigit() and len(w) > 1]
    product = ' '.join(words[:2]).capitalize() if words else 'Product'
    return product, qty
