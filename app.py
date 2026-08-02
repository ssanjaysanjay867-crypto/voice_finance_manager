from flask import Flask, render_template, request, jsonify, session, redirect
import sqlite3, hashlib, re
from datetime import datetime, timedelta
from models import (
    get_intent_classifier, get_category_classifier,
    get_forecaster, get_anomaly_detector,
    model_parse, retrain_from_db
)

app = Flask(__name__, template_folder='templates', static_folder='templates/static')
app.secret_key = 'vfm_secret_2024_xK9'

DB = 'database.db'

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def init_db():
    conn = get_db()
    conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            name     TEXT    NOT NULL,
            phone    TEXT    UNIQUE,
            password TEXT    NOT NULL,
            language TEXT    DEFAULT 'en'
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            type        TEXT    NOT NULL,
            amount      REAL    NOT NULL,
            category    TEXT    DEFAULT 'General',
            description TEXT    DEFAULT '',
            date        TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS customers (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            name        TEXT    NOT NULL,
            phone       TEXT    DEFAULT '',
            pending     REAL    DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS credit_entries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id INTEGER NOT NULL,
            user_id     INTEGER NOT NULL,
            amount      REAL    NOT NULL,
            description TEXT    DEFAULT '',
            date        TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inventory (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER NOT NULL,
            product             TEXT    NOT NULL,
            quantity            REAL    DEFAULT 0,
            price               REAL    DEFAULT 0,
            unit                TEXT    DEFAULT 'units',
            low_stock_threshold REAL    DEFAULT 5
        );
    ''')
    conn.commit()
    conn.close()

init_db()

# ── Auth ──────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return redirect('/dashboard') if 'user_id' in session else render_template('login.html')

@app.route('/api/register', methods=['POST'])
def register():
    d = request.json
    name  = d.get('name','').strip()
    phone = d.get('phone','').strip()
    pw    = d.get('password','').strip()
    lang  = d.get('language','en')
    if not name or not pw:
        return jsonify({'error': 'Name and password required'}), 400
    conn = get_db()
    if conn.execute('SELECT id FROM users WHERE phone=?', (phone,)).fetchone():
        conn.close()
        return jsonify({'error': 'Phone already registered'}), 409
    conn.execute('INSERT INTO users (name,phone,password,language) VALUES (?,?,?,?)',
                 (name, phone, hash_pw(pw), lang))
    conn.commit()
    conn.close()
    return jsonify({'status': 'ok'})

@app.route('/api/login', methods=['POST'])
def login():
    d = request.json
    phone = d.get('phone','').strip()
    pw    = d.get('password','').strip()
    lang  = d.get('language','en')
    conn  = get_db()
    user  = conn.execute('SELECT * FROM users WHERE phone=? AND password=?',
                         (phone, hash_pw(pw))).fetchone()
    conn.close()
    if not user:
        return jsonify({'error': 'Invalid phone or password'}), 401
    session['user_id']   = user['id']
    session['user_name'] = user['name']
    session['language']  = user['language'] or lang
    return jsonify({'status': 'ok', 'name': user['name'], 'language': session['language']})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'status': 'ok'})

@app.route('/api/me')
def me():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify({'name': session['user_name'], 'language': session.get('language','en')})

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html') if 'user_id' in session else redirect('/')

# ── Transactions ──────────────────────────────────────────────────────────────
@app.route('/api/transactions', methods=['GET'])
def get_transactions():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    uid    = session['user_id']
    period = request.args.get('period','today')
    today  = datetime.now().strftime('%Y-%m-%d')
    conn   = get_db()
    if period == 'today':
        rows = conn.execute('SELECT * FROM transactions WHERE user_id=? AND date=? ORDER BY id DESC',(uid,today)).fetchall()
    elif period == 'week':
        since = (datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')
        rows = conn.execute('SELECT * FROM transactions WHERE user_id=? AND date>=? ORDER BY id DESC',(uid,since)).fetchall()
    elif period == 'month':
        since = datetime.now().strftime('%Y-%m-01')
        rows = conn.execute('SELECT * FROM transactions WHERE user_id=? AND date>=? ORDER BY id DESC',(uid,since)).fetchall()
    else:
        rows = conn.execute('SELECT * FROM transactions WHERE user_id=? ORDER BY id DESC',(uid,)).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

@app.route('/api/transactions', methods=['POST'])
def add_transaction():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d   = request.json
    uid = session['user_id']
    conn = get_db()
    conn.execute('INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
                 (uid, d['type'], d['amount'],
                  d.get('category','General'), d.get('description',''),
                  datetime.now().strftime('%Y-%m-%d')))
    conn.commit(); conn.close()
    # Retrain models with latest user data (non-blocking best-effort)
    try:
        retrain_from_db(DB, uid)
    except Exception:
        pass
    return jsonify({'status':'ok'})

@app.route('/api/transactions/<int:tid>', methods=['PUT'])
def update_transaction(tid):
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d = request.json
    conn = get_db()
    conn.execute('UPDATE transactions SET type=?,amount=?,category=?,description=? WHERE id=? AND user_id=?',
                 (d['type'],d['amount'],d.get('category','General'),d.get('description',''),tid,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/transactions/<int:tid>', methods=['DELETE'])
def delete_transaction(tid):
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    conn = get_db()
    conn.execute('DELETE FROM transactions WHERE id=? AND user_id=?',(tid,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

# ── Summary / AI ──────────────────────────────────────────────────────────────
@app.route('/api/summary')
def summary():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    uid   = session['user_id']
    today = datetime.now().strftime('%Y-%m-%d')
    conn  = get_db()

    # Today totals
    rows = conn.execute('SELECT type,SUM(amount) t FROM transactions WHERE user_id=? AND date=? GROUP BY type',(uid,today)).fetchall()
    income = expense = 0
    for r in rows:
        if r['type']=='income': income=r['t']
        else: expense=r['t']

    # 7-day chart
    labels,inc_data,exp_data = [],[],[]
    for i in range(6,-1,-1):
        d = (datetime.now()-timedelta(days=i)).strftime('%Y-%m-%d')
        labels.append(d[5:])
        dr = conn.execute('SELECT type,SUM(amount) t FROM transactions WHERE user_id=? AND date=? GROUP BY type',(uid,d)).fetchall()
        di=de=0
        for r in dr:
            if r['type']=='income': di=r['t']
            else: de=r['t']
        inc_data.append(di or 0); exp_data.append(de or 0)

    # Category pie
    cat_rows = conn.execute('SELECT category,SUM(amount) t FROM transactions WHERE user_id=? AND date=? AND type="expense" GROUP BY category',(uid,today)).fetchall()

    # Weekly totals for AI
    week_ago = (datetime.now()-timedelta(days=7)).strftime('%Y-%m-%d')
    prev_week = (datetime.now()-timedelta(days=14)).strftime('%Y-%m-%d')
    w_inc = conn.execute('SELECT COALESCE(SUM(amount),0) t FROM transactions WHERE user_id=? AND type="income" AND date>=?',(uid,week_ago)).fetchone()['t']
    w_exp = conn.execute('SELECT COALESCE(SUM(amount),0) t FROM transactions WHERE user_id=? AND type="expense" AND date>=?',(uid,week_ago)).fetchone()['t']
    pw_inc = conn.execute('SELECT COALESCE(SUM(amount),0) t FROM transactions WHERE user_id=? AND type="income" AND date>=? AND date<?',(uid,prev_week,week_ago)).fetchone()['t']
    pw_exp = conn.execute('SELECT COALESCE(SUM(amount),0) t FROM transactions WHERE user_id=? AND type="expense" AND date>=? AND date<?',(uid,prev_week,week_ago)).fetchone()['t']

    # Top expense category this week
    top_cat = conn.execute('SELECT category,SUM(amount) t FROM transactions WHERE user_id=? AND type="expense" AND date>=? GROUP BY category ORDER BY t DESC LIMIT 1',(uid,week_ago)).fetchone()

    # 30-day history for ML models
    thirty_ago = (datetime.now()-timedelta(days=29)).strftime('%Y-%m-%d')
    hist_rows  = conn.execute(
        'SELECT date, type, SUM(amount) t FROM transactions WHERE user_id=? AND date>=? GROUP BY date,type',
        (uid, thirty_ago)
    ).fetchall()
    conn.close()

    # Build 30-day daily arrays (oldest → newest)
    daily_inc_map = {}; daily_exp_map = {}
    for r in hist_rows:
        if r['type'] == 'income': daily_inc_map[r['date']] = r['t']
        else:                      daily_exp_map[r['date']] = r['t']
    inc_30 = [daily_inc_map.get((datetime.now()-timedelta(days=29-i)).strftime('%Y-%m-%d'), 0) for i in range(30)]
    exp_30 = [daily_exp_map.get((datetime.now()-timedelta(days=29-i)).strftime('%Y-%m-%d'), 0) for i in range(30)]

    # ── ML Forecaster ─────────────────────────────────────────────────────────
    forecaster = get_forecaster()
    forecaster.fit(inc_30)
    forecast   = forecaster.predict_next_7(inc_30)

    # ── ML Anomaly Detector ───────────────────────────────────────────────────
    anomaly_det   = get_anomaly_detector()
    exp_anomalies = anomaly_det.detect_expense_anomalies(exp_30)
    income_drop   = anomaly_det.detect_income_drop(inc_30)
    burn_rate     = anomaly_det.budget_burn_rate(exp_30)

    profit  = income - expense
    ai_tips = generate_ai_tips(income, expense, profit, w_inc, w_exp, pw_inc, pw_exp,
                                top_cat, inc_data, exp_anomalies, income_drop, burn_rate)

    return jsonify({
        'income': income, 'expense': expense, 'profit': profit,
        'chart': {'labels': labels, 'income': inc_data, 'expense': exp_data},
        'categories': [{'name':r['category'],'amount':r['t']} for r in cat_rows],
        'ai_tips':    ai_tips,
        'prediction': forecast['total'],
        'forecast':   forecast,
        'anomalies':  {
            'expense_spikes': exp_anomalies,
            'income_drop':    income_drop,
            'burn_rate':      burn_rate
        }
    })

def generate_ai_tips(income, expense, profit, w_inc, w_exp, pw_inc, pw_exp,
                     top_cat, inc_data, exp_anomalies=None, income_drop=None, burn_rate=None):
    tips = []
    lang = session.get('language','en')

    if income == 0 and expense == 0:
        tips.append("📝 No transactions today. Start recording your sales!" if lang=='en'
                    else "📝 இன்று பரிவர்த்தனைகள் இல்லை. விற்பனையை பதிவு செய்யுங்கள்!")
        return tips

    # Profit / loss
    if profit > 0:
        margin = (profit/income*100) if income else 0
        tips.append(f"✅ Profit margin is {margin:.0f}% today. Great work!" if lang=='en'
                    else f"✅ இன்று லாப விகிதம் {margin:.0f}%. சிறப்பு!")
    elif expense > income:
        tips.append(f"⚠️ Expenses exceed income by ₹{abs(profit):.0f}. Review spending." if lang=='en'
                    else f"⚠️ செலவு வருமானத்தை ₹{abs(profit):.0f} மிகுந்துள்ளது.")

    # Week-over-week
    if w_exp > pw_exp * 1.2 and pw_exp > 0:
        tips.append("📈 Expenses up 20%+ vs last week. Consider reducing costs." if lang=='en'
                    else "📈 கடந்த வாரத்தை விட செலவு 20%+ அதிகரித்துள்ளது.")
    if w_inc > pw_inc * 1.1 and pw_inc > 0:
        tips.append("🚀 Sales up 10%+ vs last week. Keep it up!" if lang=='en'
                    else "🚀 கடந்த வாரத்தை விட விற்பனை 10%+ அதிகரித்துள்ளது!")

    # Top category
    if top_cat:
        tips.append(f"💡 Top spend: {top_cat['category']} (₹{top_cat['t']:.0f} this week)." if lang=='en'
                    else f"💡 அதிக செலவு: {top_cat['category']} (₹{top_cat['t']:.0f}).")

    # ML Anomaly tips
    if exp_anomalies:
        a = exp_anomalies[-1]
        tips.append(f"🔴 Anomaly detected: ₹{a['value']:.0f} spike (z={a['z_score']}) — unusual expense day." if lang=='en'
                    else f"🔴 அசாதாரண செலவு: ₹{a['value']:.0f} (z={a['z_score']}).")
    if income_drop:
        tips.append(f"📉 Income dropped {income_drop['drop_pct']}% vs recent avg (₹{income_drop['avg']:.0f})." if lang=='en'
                    else f"📉 வருமானம் {income_drop['drop_pct']}% குறைந்துள்ளது.")
    if burn_rate and burn_rate['status'] in ('warning','danger'):
        tips.append(f"🔥 Burn rate ₹{burn_rate['burn_rate']:.0f}/day → projected ₹{burn_rate['projected_monthly']:.0f}/month." if lang=='en'
                    else f"🔥 தினசரி செலவு ₹{burn_rate['burn_rate']:.0f} → மாதாந்திர கணிப்பு ₹{burn_rate['projected_monthly']:.0f}.")

    # Avg daily income
    non_zero = [x for x in inc_data if x > 0]
    if len(non_zero) >= 3:
        avg = sum(non_zero)/len(non_zero)
        tips.append(f"📊 Avg daily income this week: ₹{avg:.0f}." if lang=='en'
                    else f"📊 இந்த வாரம் தினசரி சராசரி வருமானம்: ₹{avg:.0f}.")

    return tips[:4]

# ── Customers / Credit ────────────────────────────────────────────────────────
@app.route('/api/customers', methods=['GET'])
def get_customers():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    rows = get_db().execute('SELECT * FROM customers WHERE user_id=? ORDER BY name',(session['user_id'],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/customers', methods=['POST'])
def add_customer():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d = request.json
    conn = get_db()
    conn.execute('INSERT INTO customers (user_id,name,phone) VALUES (?,?,?)',
                 (session['user_id'],d['name'],d.get('phone','')))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/customers/<int:cid>', methods=['DELETE'])
def delete_customer(cid):
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    conn = get_db()
    conn.execute('DELETE FROM customers WHERE id=? AND user_id=?',(cid,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/credit', methods=['POST'])
def add_credit():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d   = request.json
    uid = session['user_id']
    conn = get_db()
    cust = conn.execute('SELECT * FROM customers WHERE user_id=? AND LOWER(name) LIKE ?',
                        (uid, f"%{d['customer'].lower()}%")).fetchone()
    if not cust:
        conn.execute('INSERT INTO customers (user_id,name) VALUES (?,?)',(uid,d['customer']))
        conn.commit()
        cust = conn.execute('SELECT * FROM customers WHERE user_id=? AND name=?',(uid,d['customer'])).fetchone()
    conn.execute('INSERT INTO credit_entries (customer_id,user_id,amount,description,date) VALUES (?,?,?,?,?)',
                 (cust['id'],uid,d['amount'],d.get('description',''),datetime.now().strftime('%Y-%m-%d')))
    conn.execute('UPDATE customers SET pending=pending+? WHERE id=?',(d['amount'],cust['id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/credit/pay', methods=['POST'])
def pay_credit():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d = request.json
    conn = get_db()
    conn.execute('UPDATE customers SET pending=MAX(0,pending-?) WHERE id=? AND user_id=?',
                 (d['amount'],d['customer_id'],session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

# ── Inventory ─────────────────────────────────────────────────────────────────
@app.route('/api/inventory', methods=['GET'])
def get_inventory():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    rows = get_db().execute('SELECT * FROM inventory WHERE user_id=? ORDER BY product',(session['user_id'],)).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/inventory', methods=['POST'])
def add_inventory():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d   = request.json
    uid = session['user_id']
    conn = get_db()
    existing = conn.execute('SELECT * FROM inventory WHERE user_id=? AND LOWER(product)=?',
                            (uid, d['product'].lower())).fetchone()
    if existing:
        conn.execute('UPDATE inventory SET quantity=quantity+?,price=? WHERE id=?',
                     (d['quantity'], d.get('price', existing['price']), existing['id']))
    else:
        conn.execute('INSERT INTO inventory (user_id,product,quantity,price,unit) VALUES (?,?,?,?,?)',
                     (uid, d['product'], d['quantity'], d.get('price',0), d.get('unit','units')))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/inventory/<int:iid>', methods=['PUT'])
def update_inventory(iid):
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    d = request.json
    conn = get_db()
    conn.execute('UPDATE inventory SET product=?,quantity=?,price=?,unit=? WHERE id=? AND user_id=?',
                 (d['product'],d['quantity'],d['price'],d.get('unit','units'),iid,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/inventory/<int:iid>', methods=['DELETE'])
def delete_inventory(iid):
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    conn = get_db()
    conn.execute('DELETE FROM inventory WHERE id=? AND user_id=?',(iid,session['user_id']))
    conn.commit(); conn.close()
    return jsonify({'status':'ok'})

@app.route('/api/inventory/alerts')
def inventory_alerts():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    rows = get_db().execute('SELECT * FROM inventory WHERE user_id=? AND quantity<=low_stock_threshold',(session['user_id'],)).fetchall()
    return jsonify([dict(r) for r in rows])

# ── Product lookup (mock online API) ─────────────────────────────────────────
PRODUCT_DB = {
    'rice':{'price':55,'unit':'kg'},'wheat':{'price':35,'unit':'kg'},
    'sugar':{'price':45,'unit':'kg'},'oil':{'price':130,'unit':'litre'},
    'milk':{'price':25,'unit':'packet'},'curd':{'price':40,'unit':'cup'},
    'butter':{'price':55,'unit':'pack'},'tea':{'price':200,'unit':'pack'},
    'coffee':{'price':180,'unit':'pack'},'biscuit':{'price':20,'unit':'pack'},
    'chips':{'price':30,'unit':'pack'},'salt':{'price':20,'unit':'kg'},
    'dal':{'price':90,'unit':'kg'},'flour':{'price':40,'unit':'kg'},
    'onion':{'price':30,'unit':'kg'},'tomato':{'price':40,'unit':'kg'},
    'potato':{'price':25,'unit':'kg'},'soap':{'price':35,'unit':'bar'},
    'shampoo':{'price':120,'unit':'bottle'},'toothpaste':{'price':80,'unit':'tube'},
}

@app.route('/api/product/lookup')
def product_lookup():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    q = request.args.get('q','').lower().strip()
    matches = {k:v for k,v in PRODUCT_DB.items() if q in k}
    return jsonify({'results': [{'name':k,'price':v['price'],'unit':v['unit']} for k,v in matches.items()]})

# ── NLP Voice Parser (ML-powered) ────────────────────────────────────────────
@app.route('/api/voice/parse', methods=['POST'])
def parse_voice():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    text = request.json.get('text','').strip()
    # Use ML model_parse with regex nlp_parse as fallback
    return jsonify(model_parse(text, nlp_parse))

# ── Model info endpoint ───────────────────────────────────────────────────────
@app.route('/api/models/info')
def models_info():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    clf  = get_intent_classifier()
    ccat = get_category_classifier()
    return jsonify({
        'intent_classes':   list(clf.pipeline.classes_),
        'category_classes': list(ccat.pipeline.classes_),
        'models': ['IntentClassifier (TF-IDF + NaiveBayes)',
                   'CategoryClassifier (TF-IDF + LogisticRegression)',
                   'SalesForecaster (LinearRegression)',
                   'AnomalyDetector (Z-score)']
    })

# ── Model test endpoint ───────────────────────────────────────────────────────
@app.route('/api/models/test', methods=['POST'])
def models_test():
    if 'user_id' not in session: return jsonify({'error':'Unauthorized'}), 401
    text = request.json.get('text','').strip()
    clf  = get_intent_classifier()
    ccat = get_category_classifier()
    intent   = clf.predict(text)
    category = ccat.predict(text)
    parsed   = model_parse(text, nlp_parse)
    return jsonify({
        'input':    text,
        'intent':   intent,
        'category': category,
        'parsed':   parsed
    })

def nlp_parse(text):
    raw  = text
    text = text.lower()

    # Extract all numbers
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    amount = float(nums[0]) if nums else 0

    # ── Keyword banks (EN + Tamil) ────────────────────────────────────────────
    income_kw  = ['sold','sale','received','income','earned','got','sell','விற்றது','வந்தது','கிடைத்தது','விற்பனை']
    expense_kw = ['bought','spent','paid','expense','purchase','cost','வாங்கினேன்','செலவு','கொடுத்தேன்','வாங்கியது']
    credit_kw  = ['credit','took','owes','borrowed','கடன்','எடுத்தான்','எடுத்தாள்','கடன் வாங்கினான்']
    inv_add_kw = ['add','added','stock','received stock','சேர்','வைத்தேன்','சேர்த்தேன்']
    inv_sell_kw= ['sold','used','consumed','removed','விற்றது','பயன்படுத்தினேன்']
    report_kw  = ['report','summary','today','show','கணக்கு','இன்று','அறிக்கை']
    delete_kw  = ['delete','remove','cancel','நீக்கு']

    # ── Pattern matching ──────────────────────────────────────────────────────
    # "Ravi took 200 credit" / "credit 200 to Ravi"
    credit_pattern = re.search(r'(\w+)\s+(?:took|owes|borrowed|எடுத்தான்|எடுத்தாள்)\s+(\d+)', text)
    credit_pattern2 = re.search(r'credit\s+(\d+)\s+(?:to|for)\s+(\w+)', text)

    if credit_pattern:
        return {'action':'credit','customer':credit_pattern.group(1).capitalize(),
                'amount':float(credit_pattern.group(2)),'text':raw}
    if credit_pattern2:
        return {'action':'credit','customer':credit_pattern2.group(2).capitalize(),
                'amount':float(credit_pattern2.group(1)),'text':raw}
    if any(k in text for k in credit_kw):
        name = _extract_name(text)
        return {'action':'credit','customer':name,'amount':amount,'text':raw}

    if any(k in text for k in report_kw):
        return {'action':'report','text':raw}

    # Inventory: "add 10 rice bags" / "sold 2 milk"
    if any(k in text for k in inv_add_kw):
        product, qty = _extract_product_qty(text, inv_add_kw)
        return {'action':'inventory_add','product':product,'quantity':qty or amount,'text':raw}

    if any(k in text for k in inv_sell_kw) and amount > 0:
        product, qty = _extract_product_qty(text, inv_sell_kw)
        # Could be a sale (income) + inventory deduction
        category = _detect_category(text)
        return {'action':'sale','product':product,'quantity':qty or amount,
                'amount':amount,'category':category,'text':raw}

    if any(k in text for k in income_kw):
        category = _detect_category(text)
        return {'action':'income','amount':amount,'category':category,'text':raw}

    if any(k in text for k in expense_kw):
        category = _detect_category(text)
        return {'action':'expense','amount':amount,'category':category,'text':raw}

    if any(k in text for k in delete_kw):
        return {'action':'delete','text':raw}

    return {'action':'unknown','amount':amount,'text':raw}

def _extract_name(text):
    stop = {'took','credit','rupees','rs','for','கடன்','எடுத்தான்','amount','paid','owes','borrowed'}
    nums = set(re.findall(r'\d+', text))
    words = [w.capitalize() for w in text.split()
             if w not in stop and w not in nums and len(w) > 2]
    return words[0] if words else 'Customer'

def _extract_product_qty(text, skip_kw):
    stop = set(skip_kw) | {'packets','packet','units','kg','litre','liters','ml','nos','bags','bag','box','boxes'}
    nums = re.findall(r'\d+(?:\.\d+)?', text)
    qty  = float(nums[0]) if nums else 0
    words = [w for w in text.split() if w not in stop and not w.replace('.','').isdigit() and len(w) > 1]
    product = ' '.join(words[:2]).capitalize() if words else 'Product'
    return product, qty

def _detect_category(text):
    cats = {
        'Groceries':  ['rice','wheat','sugar','oil','salt','flour','dal','grocery'],
        'Vegetables': ['vegetable','tomato','onion','potato','carrot','greens'],
        'Dairy':      ['milk','curd','butter','cheese','dairy','packet'],
        'Beverages':  ['tea','coffee','juice','drink','water','soda'],
        'Snacks':     ['biscuit','chips','snack','chocolate','candy'],
        'Utilities':  ['electricity','bill','rent','utility','internet'],
        'Transport':  ['transport','fuel','petrol','auto','vehicle','bus'],
        'Medicine':   ['medicine','tablet','syrup','medical','pharmacy'],
    }
    for cat, kws in cats.items():
        if any(k in text for k in kws):
            return cat
    return 'General'

# ── Demo / Seed Data ──────────────────────────────────────────────────────────
import random, math

@app.route('/api/seed', methods=['POST'])
def seed_demo_data():
    """
    Populates the logged-in user's account with 60 days of realistic
    shop data: transactions, inventory, customers + credit entries.
    Safe to call multiple times — clears existing data first.
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    uid  = session['user_id']
    conn = get_db()

    # ── Wipe existing data for this user ──────────────────────────────────────
    for tbl in ('transactions','inventory','credit_entries','customers'):
        conn.execute(f'DELETE FROM {tbl} WHERE user_id=?', (uid,))
    conn.commit()

    today = datetime.now()

    # ── 1. Transactions — 60 days of realistic shop income & expenses ─────────
    income_templates = [
        ('Groceries',  'Rice sale',          180, 420),
        ('Groceries',  'Wheat flour sale',   120, 280),
        ('Dairy',      'Milk packets sold',   80, 200),
        ('Dairy',      'Curd sale',           60, 150),
        ('Snacks',     'Biscuits & chips',    50, 180),
        ('Beverages',  'Tea & coffee sale',   40, 120),
        ('Vegetables', 'Vegetable sale',      90, 250),
        ('General',    'Miscellaneous sale', 100, 350),
    ]
    expense_templates = [
        ('Groceries',  'Rice stock purchase',   300, 700),
        ('Dairy',      'Milk stock purchase',   150, 350),
        ('Utilities',  'Electricity bill',      200, 500),
        ('Transport',  'Delivery charges',       80, 200),
        ('Snacks',     'Biscuit stock',         100, 250),
        ('Vegetables', 'Vegetable purchase',    120, 300),
        ('General',    'Shop maintenance',       50, 150),
        ('Medicine',   'First aid supplies',     30,  80),
    ]

    rng = random.Random(42)   # fixed seed → reproducible data

    tx_rows = []
    for day_offset in range(59, -1, -1):
        date_str = (today - timedelta(days=day_offset)).strftime('%Y-%m-%d')
        dow      = (today - timedelta(days=day_offset)).weekday()   # 0=Mon

        # Weekend boost
        multiplier = 1.4 if dow >= 5 else 1.0
        # Slight upward trend over 60 days
        trend = 1 + (59 - day_offset) * 0.003

        # 3-6 income entries per day
        for _ in range(rng.randint(3, 6)):
            cat, desc, lo, hi = rng.choice(income_templates)
            amt = round(rng.uniform(lo, hi) * multiplier * trend, 2)
            tx_rows.append((uid, 'income', amt, cat, desc, date_str))

        # 2-4 expense entries per day
        for _ in range(rng.randint(2, 4)):
            cat, desc, lo, hi = rng.choice(expense_templates)
            amt = round(rng.uniform(lo, hi), 2)
            tx_rows.append((uid, 'expense', amt, cat, desc, date_str))

    conn.executemany(
        'INSERT INTO transactions (user_id,type,amount,category,description,date) VALUES (?,?,?,?,?,?)',
        tx_rows
    )

    # ── 2. Inventory — 20 realistic shop products ─────────────────────────────
    inventory_items = [
        ('Rice',          120,  55, 'kg',      10),
        ('Wheat Flour',    80,  40, 'kg',       8),
        ('Sugar',          60,  45, 'kg',       5),
        ('Cooking Oil',    30, 130, 'litre',    4),
        ('Salt',           50,  20, 'kg',       5),
        ('Dal (Toor)',      40,  90, 'kg',       5),
        ('Milk Packets',    3,  25, 'packets',  5),   # low stock!
        ('Curd',           15,  40, 'cups',     5),
        ('Butter',         20,  55, 'packs',    4),
        ('Tea Powder',     25, 200, 'packs',    3),
        ('Coffee Powder',  18, 180, 'packs',    3),
        ('Biscuits',       60,  20, 'packs',    8),
        ('Chips',          45,  30, 'packs',    6),
        ('Tomato',         10,  40, 'kg',       5),   # low stock!
        ('Onion',          25,  30, 'kg',       5),
        ('Potato',         35,  25, 'kg',       5),
        ('Soap',           40,  35, 'bars',     6),
        ('Shampoo',        12, 120, 'bottles',  3),
        ('Toothpaste',     20,  80, 'tubes',    4),
        ('Detergent',       2, 150, 'packs',    3),   # low stock!
    ]
    conn.executemany(
        'INSERT INTO inventory (user_id,product,quantity,price,unit,low_stock_threshold) VALUES (?,?,?,?,?,?)',
        [(uid, p, q, pr, u, t) for p, q, pr, u, t in inventory_items]
    )

    # ── 3. Customers + credit entries ─────────────────────────────────────────
    customers = [
        ('Ravi Kumar',    '9876543210', 850),
        ('Priya Devi',    '9123456780', 0),
        ('Suresh Babu',   '9988776655', 1200),
        ('Meena Kumari',  '9765432109', 450),
        ('Arjun Sharma',  '9654321098', 0),
        ('Lakshmi Bai',   '9543210987', 320),
        ('Karthik Raja',  '9432109876', 0),
        ('Anitha Raj',    '9321098765', 680),
        ('Vijay Mohan',   '9210987654', 0),
        ('Deepa Nair',    '9109876543', 150),
    ]
    for cname, cphone, pending in customers:
        conn.execute(
            'INSERT INTO customers (user_id,name,phone,pending) VALUES (?,?,?,?)',
            (uid, cname, cphone, pending)
        )
    conn.commit()

    # Add credit entries for customers with pending > 0
    cust_rows = conn.execute(
        'SELECT id,name,pending FROM customers WHERE user_id=? AND pending>0', (uid,)
    ).fetchall()
    credit_entries = []
    for c in cust_rows:
        # Split pending into 1-3 credit entries over last 30 days
        remaining = c['pending']
        parts = rng.randint(1, 3)
        for i in range(parts):
            amt  = round(remaining / (parts - i), 2)
            remaining -= amt
            days_ago = rng.randint(1, 30)
            credit_entries.append((
                c['id'], uid, amt,
                f"{c['name']} credit purchase",
                (today - timedelta(days=days_ago)).strftime('%Y-%m-%d')
            ))
    conn.executemany(
        'INSERT INTO credit_entries (customer_id,user_id,amount,description,date) VALUES (?,?,?,?,?)',
        credit_entries
    )
    conn.commit()
    conn.close()

    # Retrain ML models with the new data
    try:
        retrain_from_db(DB, uid)
    except Exception:
        pass

    return jsonify({
        'status':       'ok',
        'transactions': len(tx_rows),
        'inventory':    len(inventory_items),
        'customers':    len(customers),
        'message':      'Demo data loaded successfully!'
    })


@app.route('/api/seed/clear', methods=['POST'])
def clear_demo_data():
    """Wipe all data for the current user."""
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    uid  = session['user_id']
    conn = get_db()
    for tbl in ('transactions','inventory','credit_entries','customers'):
        conn.execute(f'DELETE FROM {tbl} WHERE user_id=?', (uid,))
    conn.commit(); conn.close()
    return jsonify({'status': 'ok', 'message': 'All data cleared.'})


if __name__ == '__main__':
    app.run(debug=True)
