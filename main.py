import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
import csv
import json
import re
import math
from datetime import datetime, timedelta
from collections import Counter

# --- CONFIGURACIÓN v69.0 (DIXON-COLES & DYNAMIC WEIGHTS) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "03:13" 

# AJUSTES DE MODELO
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.88          
WEIGHT_MODEL_DEFAULT = 0.30 # Fallback

SEASON = '2526'       
HISTORY_FILE = "historial_omni_hybrid.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        
MIN_EV_THRESHOLD = 0.02     

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

try:
    from google import genai
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False

# --- CONFIGURACIÓN DE LIGAS CON PESOS DINÁMICOS ---
# weight: Cuánto confiamos en el MERCADO (0.0 a 1.0)
# Ligas eficientes (E0, SP1) -> Alto peso al mercado (0.85)
# Ligas volátiles (T1, B1) -> Menor peso al mercado (0.60), más al modelo
LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1, 'weight': 0.85},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1, 'weight': 0.85},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1, 'weight': 0.80},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1, 'weight': 0.80},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 1, 'weight': 0.75},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 2, 'weight': 0.70},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 2, 'weight': 0.70},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 2, 'weight': 0.60},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 2, 'weight': 0.60}
}

class OmniHybridBot:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self.daily_picks_buffer = [] 
        self._check_creds()
        self._init_history_file()
        
        self.ai_client = None
        if SDK_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini SDK: INICIALIZADO (v2.0 Flash)", flush=True)
            except Exception as e:
                print(f"⚠️ Error Init Gemini: {e}", flush=True)

    def _check_creds(self):
        print("--- ENGINE v69 DIXON-COLES STARTED ---", flush=True)

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Market', 'Prob', 'Odd', 'EV', 'Result', 'Profit'])

    # --- TELEGRAM ---
    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', '', text) 
        text = text.replace('*', '').replace('_', '').replace('`', '')
        return text

    def send_msg(self, text, retry_count=0, use_html=True):
        if not TELEGRAM_TOKEN: return
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks: self.send_msg(chunk, retry_count, use_html)
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML" if use_html else None}
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 400 and use_html:
                self.send_msg(self.clean_text(text), retry_count, use_html=False)
                return
            if r.status_code == 429:
                retry = int(r.json().get('parameters', {}).get('retry_after', 30))
                time.sleep(retry + 2)
                if retry_count < 2: self.send_msg(text, retry_count + 1, use_html)
                return
        except Exception as e: print(f"Error Telegram: {e}", flush=True)
        time.sleep(2)

    def dec_to_am(self, decimal_odd):
        if decimal_odd <= 1.01: return "-10000"
        if decimal_odd >= 2.00: return f"+{int((decimal_odd - 1) * 100)}"
        else: return f"{int(-100 / (decimal_odd - 1))}"

    def call_gemini(self, prompt):
        if not SDK_AVAILABLE or not self.ai_client: return "❌ SDK no disponible."
        try:
            r = self.ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return r.text
        except Exception as e: return f"❌ Error Gemini: {str(e)[:100]}"

    # --- CÁLCULO ATAQUE Y DEFENSA ---
    def calculate_team_stats(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(6)
        if len(matches) < 3: return 1.0, 1.0
        
        w_att = 0; w_def = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, 5 - i); total_w += weight
            if row['HomeTeam'] == team:
                att = (row['FTHG'] * 0.6) + ((row.get('HST', row['FTHG']*3)/3) * 0.4)
                def_weak = (row['FTAG'] * 0.6) + ((row.get('AST', row['FTAG']*3)/3) * 0.4)
            else:
                att = (row['FTAG'] * 0.6) + ((row.get('AST', row['FTAG']*3)/3) * 0.4)
                def_weak = (row['FTHG'] * 0.6) + ((row.get('HST', row['FTHG']*3)/3) * 0.4)
            w_att += att * weight; w_def += def_weak * weight
            
        return w_att / total_w, w_def / total_w

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            if r.status_code != 200: return None
            try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
            df = df.dropna(subset=['FTHG', 'FTAG'])
            
            avg_g = df.FTHG.mean() + df.FTAG.mean()
            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            team_stats = {}
            avg_att = 0; avg_def = 0; cnt = 0
            for t in teams:
                a, d = self.calculate_team_stats(df, t)
                team_stats[t] = {'att': a, 'def': d}
                avg_att += a; avg_def += d; cnt += 1
            avg_att /= cnt; avg_def /= cnt
            
            norm_stats = {t: {'att': s['att']/avg_att, 'def': s['def']/avg_def} for t, s in team_stats.items()}
            # Guardamos también el peso del mercado para esta liga
            league_weight = LEAGUE_CONFIG.get(div, {}).get('weight', 0.70)
            self.history_cache[div] = {'stats': norm_stats, 'teams': teams, 'raw_df': df, 'avg_g': avg_g, 'market_weight': league_weight}
            return self.history_cache[div]
        except: return None

    # --- MOTOR MATEMÁTICO: DIXON-COLES ---
    def poisson_prob(self, k, lamb):
        return (math.pow(lamb, k) * math.exp(-lamb)) / math.factorial(k)

    def calculate_dixon_coles_1x2(self, lambda_h, lambda_a):
        # Ajuste Rho para corregir la subestimación de empates
        rho = -0.13 
        prob_h, prob_d, prob_a = 0.0, 0.0, 0.0
        
        # Iteramos hasta 6 goles (suficiente precisión)
        for x in range(7):
            for y in range(7):
                p = self.poisson_prob(x, lambda_h) * self.poisson_prob(y, lambda_a)
                
                # Factor de corrección Dixon-Coles para marcadores bajos
                correction = 1.0
                if x == 0 and y == 0: correction = 1 - (lambda_h * lambda_a * rho)
                elif x == 0 and y == 1: correction = 1 + (lambda_h * rho)
                elif x == 1 and y == 0: correction = 1 + (lambda_a * rho)
                elif x == 1 and y == 1: correction = 1 - (rho)
                
                final_p = p * correction
                
                if x > y: prob_h += final_p
                elif x == y: prob_d += final_p
                else: prob_a += final_p
                
        # Renormalizar por si acaso
        total = prob_h + prob_d + prob_a
        return prob_h/total, prob_d/total, prob_a/total

    def calibrate_goal_prob(self, p):
        return 0.5 + (p - 0.5) * 0.75

    def simulate_match(self, home, away, league_data, market_odds):
        stats = league_data['stats']
        avg_g = league_data['avg_g'] / 2
        m_weight = league_data.get('market_weight', 0.70) # Peso dinámico
        model_weight = 1.0 - m_weight
        
        h_st = stats.get(home, {'att':1.0, 'def':1.0})
        a_st = stats.get(away, {'att':1.0, 'def':1.0})
        
        lambda_h = min(3.5, h_st['att'] * a_st['def'] * avg_g * 1.20)
        lambda_a = min(3.5, a_st['att'] * h_st['def'] * avg_g)
        
        # 1. Probabilidades MODELO (Usando Dixon-Coles analítico, más preciso)
        model_h, model_d, model_a = self.calculate_dixon_coles_1x2(lambda_h, lambda_a)
        
        # 2. Simulacion Monte Carlo (Para Handicaps y Goles complejos)
        h_sim = np.random.poisson(lambda_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(lambda_a, SIMULATION_RUNS)
        
        # 3. FUSIÓN DINÁMICA
        if market_odds['H'] > 0:
            margin = 1.05 
            implied_h = (1 / market_odds['H']) / margin
            implied_a = (1 / market_odds['A']) / margin
            implied_d = (1 / market_odds['D']) / margin
            
            raw_h = (implied_h * m_weight) + (model_h * model_weight)
            raw_a = (implied_a * m_weight) + (model_a * model_weight)
            raw_d = (implied_d * m_weight) + (model_d * model_weight)
            
            total = raw_h + raw_a + raw_d
            final_h, final_a, final_d = raw_h/total, raw_a/total, raw_d/total
        else:
            final_h, final_d, final_a = model_h, model_d, model_a

        # Goles
        over25_raw = np.mean((h_sim + a_sim) > 2.5)
        over25 = self.calibrate_goal_prob(over25_raw)
        
        if (lambda_h + lambda_a) > 2.6 and abs(lambda_h - lambda_a) > 1.4: 
            over25 *= 0.88 
            
        implied_over = 0.5
        if market_odds.get('O25', 0) > 1:
            implied_over = (1 / market_odds['O25']) / 1.05
            over25 = (over25 * 0.70) + (implied_over * 0.30)
            
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        
        # GCS Score
        xg_sum = lambda_h + lambda_a
        xg_diff = abs(lambda_h - lambda_a)
        xg_score = min(1, max(0, (xg_sum - 1.8) / 1.8))
        balance = max(0, 1 - (xg_diff / xg_sum)) if xg_sum > 0 else 0
        gcs = (0.30 * xg_score + 0.20 * balance + 0.20 * abs(over25 - 0.5)*2 + 0.15 * (1-abs(btts-over25)) + 0.15 * min(1, abs(over25-implied_over)/0.12)) * 100

        # Marcador Probable
        sim_scores = list(zip(h_sim, a_sim))
        most_common, count = Counter(sim_scores).most_common(1)[0]
        cs_str = f"{most_common[0]}-{most_common[1]}"
        cs_prob = (count / SIMULATION_RUNS) * 100

        ah_h_minus = np.mean((h_sim - 1.5) > a_sim)
        ah_a_minus = np.mean((a_sim - 1.5) > h_sim)
        ah_h_plus = np.mean((h_sim + 1.5) > a_sim)
        ah_a_plus = np.mean((a_sim + 1.5) > h_sim)

        return {
            'lambdas': (lambda_h, lambda_a),
            'stats': (h_st, a_st),
            '1x2': (final_h, final_d, final_a),
            'goals': (over25, btts),
            'dc': (final_h + final_d, final_a + final_d),
            'dnb': (final_h/(final_h+final_a), final_a/(final_h+final_a)),
            'ah': (ah_h_minus, ah_a_minus, ah_h_plus, ah_a_plus),
            'gcs': gcs, 'cs': (cs_str, cs_prob), 'm_weight': m_weight
        }

    # --- MULTI-BOOKIE ---
    def get_avg_odds(self, row):
        def get_avg(cols):
            vals = [float(row[c]) for c in cols if row.get(c) and str(row[c]).replace('.','').isdigit()]
            return sum(vals)/len(vals) if vals else 0.0
        return {
            'H': get_avg(['B365H', 'PSH', 'WHH']),
            'D': get_avg(['B365D', 'PSD', 'WHD']),
            'A': get_avg(['B365A', 'PSA', 'WHA']),
            'O25': get_avg(['B365>2.5', 'P>2.5', 'WH>2.5'])
        }

    # --- SELECCIÓN ---
    def find_best_value(self, sim, odds):
        candidates = []
        
        def add(name, market, prob, odd, gcs=None):
            if odd < 1.05: return
            ev = (prob * odd) - 1
            status = "VALID"; reason = "OK"
            
            if ev < MIN_EV_THRESHOLD: status="REJECTED"; reason=f"EV Bajo ({ev*100:.1f}%)"
            elif prob < 0.35: status="REJECTED"; reason=f"Riesgo ({prob*100:.0f}%)"
            elif ev > 0.45: status="REJECTED"; reason="Error Modelo"
            
            if market == 'GOALS':
                if gcs < 55: status="REJECTED"; reason=f"GCS Pobre ({gcs:.0f})"
                elif prob > 0.65 or prob < 0.35: status="REJECTED"; reason="Prob Extrema"

            score = ev * (prob ** 1.5)
            candidates.append({'pick': name, 'market': market, 'prob': prob, 'odd': odd, 'ev': ev, 'score': score, 'status': status, 'reason': reason})

        if odds['H'] > 0:
            add("GANA HOME", "1X2", sim['1x2'][0], odds['H'])
            add("GANA AWAY", "1X2", sim['1x2'][2], odds['A'])
            
            odd_dnb_h = (odds['H'] * (1 - (1/odds['D']))) * 0.94
            odd_dnb_a = (odds['A'] * (1 - (1/odds['D']))) * 0.94
            add("DNB HOME", "DNB", sim['dnb'][0], odd_dnb_h)
            add("DNB AWAY", "DNB", sim['dnb'][1], odd_dnb_a)
            
            odd_dc_1x = 1 / ((1/odds['H']) + (1/odds['D'])) * 0.94
            odd_dc_x2 = 1 / ((1/odds['A']) + (1/odds['D'])) * 0.94
            add("DC 1X", "Double Chance", sim['dc'][0], odd_dc_1x)
            add("DC X2", "Double Chance", sim['dc'][1], odd_dc_x2)

        if odds['O25'] > 0:
            o_u25 = 1 / (1 - (1/odds['O25'] * 1.05))
            add("OVER 2.5 GOLES", "GOALS", sim['goals'][0], odds['O25'], sim['gcs'])
            add("UNDER 2.5 GOLES", "GOALS", 1-sim['goals'][0], o_u25, sim['gcs'])

        if not candidates: return None
        validos = [c for c in candidates if c['status'] == "VALID"]
        if validos:
            validos.sort(key=lambda x: x['score'], reverse=True)
            return validos[0]
        else:
            candidates.sort(key=lambda x: x['ev'], reverse=True)
            return candidates[0]

    def get_kelly_stake(self, prob, odds, market):
        if odds <= 1.0: return 0.0
        q = 1 - prob; b = odds - 1
        full = (b * prob - q) / b
        stake = full * KELLY_FRACTION
        if market in ['GOALS', 'BTTS']: stake *= 0.70
        return max(0.0, min(stake, MAX_STAKE_PCT))

    def get_team_form_icon(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
        if len(matches) == 0: return "➡️"
        points = 0; possible = len(matches) * 3
        for _, row in matches.iterrows():
            if row['HomeTeam'] == team:
                if row['FTHG'] > row['FTAG']: points += 3
                elif row['FTHG'] == row['FTAG']: points += 1
            else:
                if row['FTAG'] > row['FTHG']: points += 3
                elif row['FTAG'] == row['FTHG']: points += 1
        pct = points / possible
        if pct >= 0.7: return "🔥"; 
        if pct <= 0.3: return "🧊"; 
        return "➡️"

    def generate_final_summary(self):
        if not self.daily_picks_buffer: return
        self.send_msg("⏳ <b>El Jefe de Estrategia está auditando la cartera...</b>")
        picks_text = "\n".join(self.daily_picks_buffer)
        
        prompt = f"""
        Eres el JEFE DE ESTRATEGIA de un tipster cuantitativo.
        
        DATOS (Picks procesados):
        ===
        {picks_text}
        ===

        TAREAS:
        1. Filtra los "NO BET".
        2. De los ACTIVOS, selecciona LA JOYA y EL BANKER.
        3. Comenta la estrategia.
        
        FORMATO HTML (Sin Markdown):
        🧠 <b>DICTAMEN FINAL</b>
        💎 <b>LA JOYA:</b> [Selección]
        🛡️ <b>EL BANKER:</b> [Selección]
        ✅ <b>LISTA DE VALOR:</b> [Breve lista]
        📊 <b>ESTRATEGIA:</b> [Consejo]
        """
        
        ai_resp = self.call_gemini(prompt)
        if ai_resp: self.send_msg(ai_resp)

    def run_analysis(self):
        self.daily_picks_buffer = [] 
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando v69 DIXON-COLES: {today}", flush=True)
        
        ts = int(time.time())
        url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
        try:
            r = requests.get(url_fixt, headers={'User-Agent': USER_AGENTS[0]}, timeout=20)
            if r.status_code!=200: return
            try: content = r.content.decode('utf-8-sig')
            except: content = r.content.decode('latin-1')
            df = pd.read_csv(io.StringIO(content), on_bad_lines='skip')
            df.columns = df.columns.str.strip().str.replace('ï»¿', '')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        except: return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = df[(df['Date'] >= target_date) & (df['Date'] <= target_date + timedelta(days=1))]
        
        bets_found = 0
        self.send_msg(f"🔎 <b>Analizando {len(daily)} partidos (Dixon-Coles + Weights)...</b>")
        
        for idx, row in daily.iterrows():
            div = row.get('Div')
            if div not in LEAGUE_CONFIG: continue
            
            data = self.get_league_data(div)
            if not data: continue
            
            rh = difflib.get_close_matches(row['HomeTeam'], data['teams'], n=1, cutoff=0.6)
            ra = difflib.get_close_matches(row['AwayTeam'], data['teams'], n=1, cutoff=0.6)
            if not rh or not ra: continue
            rh = rh[0]; ra = ra[0]
            
            m_odds = self.get_avg_odds(row)
            sim = self.simulate_match(rh, ra, data, m_odds)
            best_bet = self.find_best_value(sim, m_odds)
            
            if best_bet:
                is_valid = best_bet['status'] == "VALID"
                if is_valid:
                    bets_found += 1
                    icon = "🎯"; status_line = "✅ <b>PICK ACTIVO</b>"
                    stake = self.get_kelly_stake(best_bet['prob'], best_bet['odd'], best_bet['market'])
                    stake_txt = f"{stake*100:.2f}%"
                else:
                    icon = "⛔"; status_line = f"⚠️ <b>NO BET:</b> {best_bet['reason']}"
                    stake = 0.0; stake_txt = "Skipped"

                form_h = self.get_team_form_icon(data['raw_df'], rh)
                form_a = self.get_team_form_icon(data['raw_df'], ra)
                
                ph, pd_raw, pa = sim['1x2']
                dc1x, dcx2 = sim['dc']
                dnb_h, dnb_a = sim['dnb']
                btts = sim['goals'][1]
                ov25 = sim['goals'][0]
                ah_h_m15, ah_a_m15, ah_h_p15, ah_a_p15 = sim['ah']
                h_stats, a_stats = sim['stats']
                lambdas = sim['lambdas']
                cs_str, cs_prob = sim['cs']
                fair_odd_us = self.dec_to_am(1/best_bet['prob'])
                gcs_info = f" | 🎯 GCS: <b>{sim['gcs']:.0f}</b>" if best_bet['market'] == 'GOALS' else ""
                
                msg = (
                    f"🛡️ <b>ANÁLISIS v69</b> | {LEAGUE_CONFIG[div]['name']}\n"
                    f"⚽ <b>{rh}</b> {form_h} vs {form_a} <b>{ra}</b>\n"
                    f"───────────────\n"
                    f"{status_line}\n"
                    f"{icon} PICK: <b>{best_bet['pick']}</b> ({best_bet['market']}){gcs_info}\n"
                    f"⚖️ Cuota Avg: <b>{self.dec_to_am(best_bet['odd'])}</b> ({best_bet['odd']:.2f})\n"
                    f"🧠 Prob: <b>{best_bet['prob']*100:.1f}%</b> (Fair: {fair_odd_us})\n"
                    f"📈 EV: <b>+{best_bet['ev']*100:.1f}%</b>\n"
                    f"🏦 Stake: {stake_txt}\n"
                    f"───────────────\n"
                    f"📊 <b>X-RAY (Probabilidades):</b>\n"
                    f"• 1X2: {ph*100:.0f}% | {pd_raw*100:.0f}% | {pa*100:.0f}%\n"
                    f"• DC: 1X {dc1x*100:.0f}% | X2 {dcx2*100:.0f}%\n"
                    f"• DNB: H {dnb_h*100:.0f}% | A {dnb_a*100:.0f}%\n"
                    f"• BTTS: Sí {btts*100:.0f}% | No {(1-btts)*100:.0f}%\n"
                    f"• Goals: Over {ov25*100:.0f}% | Under {(1-ov25)*100:.0f}%\n"
                    f"• Handi -1.5: H {ah_h_m15*100:.0f}% | A {ah_a_m15*100:.0f}%\n"
                    f"• Handi +1.5: H {ah_h_p15*100:.0f}% | A {ah_a_p15*100:.0f}%\n"
                    f"───────────────\n"
                    f"🎯 Marcador Probable: <b>{cs_str}</b> ({cs_prob:.1f}%)\n"
                    f"⚔️ PODER (Att / Def / Exp.Goals):\n"
                    f"🏠 {rh}: {h_stats['att']:.2f} / {h_stats['def']:.2f} => <b>{lambdas[0]:.2f}</b> gls\n"
                    f"✈️ {ra}: {a_stats['att']:.2f} / {a_stats['def']:.2f} => <b>{lambdas[1]:.2f}</b> gls\n"
                    f"⚖️ Confianza en Mercado: {sim['m_weight']*100:.0f}%"
                )
                self.send_msg(msg)
                
                if is_valid:
                    self.daily_picks_buffer.append(f"- {rh} vs {ra}: {best_bet['pick']} @ {best_bet['odd']:.2f} (EV: {best_bet['ev']*100:.1f}%)")
                
                with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([today, div, rh, ra, best_bet['pick'], best_bet['market'], best_bet['prob'], best_bet['odd'], best_bet['ev'], best_bet['status'], 0])

        if bets_found > 0:
            self.generate_final_summary()
        else:
            self.send_msg("🧹 Barrido completado: Sin oportunidades claras hoy.")

if __name__ == "__main__":
    bot = OmniHybridBot()
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
