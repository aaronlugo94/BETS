import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
import csv
from datetime import datetime, timedelta

# --- INTENTO DE IMPORTACIÓN SEGURA DE GEMINI ---
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini no detectado. Modo Matemático Puro.")

# --- CONFIGURACIÓN EURO-SNIPER v51.0 (OMNI-MARKET) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "06:04" 

# AJUSTES DE MODELO (SHOT-BASED XG)
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.89          
WEIGHT_GOALS = 0.60         
WEIGHT_SOT = 0.40           
SEASON = '2526'             
HISTORY_FILE = "historial_omni_bets.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        
MIN_EV_THRESHOLD = 0.02     # Umbral más bajo porque buscamos "la mejor" de muchas

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 1},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 2},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 2},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 2},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 2}
}

class OmniSniperBot:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self._check_creds()
        self._init_history_file()
        
        self.ai_client = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini AI: ACTIVO (Modo Omni-Analista)", flush=True)
            except: pass

    def _check_creds(self):
        print("--- OMNI-MARKET SNIPER v51 STARTED ---", flush=True)

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick_Type', 'Market', 'Prob', 'Odd', 'EV', 'Stake', 'Result', 'Profit'])

    def send_msg(self, text):
        if not TELEGRAM_TOKEN: 
            print(f"[MOCK MSG] {text}")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        try: requests.post(url, json=payload, timeout=10)
        except: pass

    def dec_to_am(self, decimal_odd):
        if decimal_odd <= 1.01: return "-10000"
        if decimal_odd >= 2.00: return f"+{int((decimal_odd - 1) * 100)}"
        else: return f"{int(-100 / (decimal_odd - 1))}"

    def get_ai_analysis(self, match, pick):
        if not self.ai_client: return ""
        prompt = f"""
        ANALISTA APUESTAS PRO.
        Partido: {match}
        El algoritmo ha elegido EL MEJOR PICK POSIBLE entre todos los mercados: {pick}
        
        Tarea:
        1. ¿Por qué este mercado es mejor que simplemente apostar al ganador?
        2. Breve riesgo a considerar.
        
        Output HTML: 🤖 <b>AI TACTICS:</b> [Texto]
        """
        try:
            r = self.ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return f"\n───────────────\n{r.text}"
        except: return ""

    # --- MOTOR MATEMÁTICO ---
    def calculate_advanced_stats(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(8)
        if len(matches) < 3: return 1.0
        
        w_sot = 0; w_goals = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, 7 - i)
            total_w += weight
            if row['HomeTeam'] == team:
                sot = row.get('HST', row['FTHG'] * 2.5) 
                goals = row['FTHG']
            else:
                sot = row.get('AST', row['FTAG'] * 2.5)
                goals = row['FTAG']
            w_sot += sot * weight
            w_goals += goals * weight
            
        avg_sot = w_sot / total_w
        avg_goals = w_goals / total_w
        return (avg_sot * 0.40) + (avg_goals * 0.60)

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            if r.status_code != 200: return None
            try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
            
            df = df.dropna(subset=['FTHG', 'FTAG'])
            avg_xg_league = 0; cnt = 0
            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            team_xgs = {}
            for t in teams:
                xg = self.calculate_advanced_stats(df, t)
                team_xgs[t] = xg
                avg_xg_league += xg
                cnt += 1
            avg_xg_league /= cnt
            
            normalized_strength = {t: val/avg_xg_league for t, val in team_xgs.items()}
            self.history_cache[div] = {'strength': normalized_strength, 'teams': teams, 'avg_goals': df.FTHG.mean() + df.FTAG.mean()}
            return self.history_cache[div]
        except: return None

    def simulate_match(self, home, away, league_data):
        s = league_data['strength']
        avg_g = league_data['avg_goals'] / 2
        
        xg_h = min(3.5, s.get(home, 1.0) * avg_g * 1.25) # Home advantage
        xg_a = min(3.5, s.get(away, 1.0) * avg_g)
        
        h_sim = np.random.poisson(xg_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(xg_a, SIMULATION_RUNS)
        
        # Probabilidades Fundamentales
        win_h = np.mean(h_sim > a_sim)
        win_a = np.mean(h_sim < a_sim)
        draw = np.mean(h_sim == a_sim)
        
        # Mercados de Goles
        total_goals = h_sim + a_sim
        over25 = np.mean(total_goals > 2.5)
        under25 = np.mean(total_goals < 2.5)
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        btts_no = 1.0 - btts
        
        # Mercados Derivados
        dc_1x = win_h + draw
        dc_x2 = win_a + draw
        dnb_h = win_h / (win_h + win_a)
        dnb_a = win_a / (win_h + win_a)
        
        return {
            'xg': (xg_h, xg_a),
            '1x2': (win_h, draw, win_a),
            'ou': (over25, under25),
            'btts': (btts, btts_no),
            'dc': (dc_1x, dc_x2),
            'dnb': (dnb_h, dnb_a)
        }

    # --- EL CEREBRO: SELECCIONA EL MEJOR PICK DE TODOS ---
    def find_best_value(self, sim, odds_row):
        candidates = []
        
        # Helper para calcular EV y añadir a candidatos
        def add_cand(name, market_type, prob, odd):
            if odd < 1.05: return
            ev = (prob * odd) - 1
            # Score = Mix de EV y Probabilidad (Preferimos alta probabilidad para stakes altos)
            score = ev * (prob ** 0.5) 
            if ev > MIN_EV_THRESHOLD:
                candidates.append({
                    'pick': name,
                    'market': market_type,
                    'prob': prob,
                    'odd': odd,
                    'ev': ev,
                    'score': score
                })

        # 1. Mercado 1X2
        try:
            o_h = float(odds_row.get('B365H', odds_row.get('AvgH', 0)))
            o_d = float(odds_row.get('B365D', odds_row.get('AvgD', 0)))
            o_a = float(odds_row.get('B365A', odds_row.get('AvgA', 0)))
            add_cand("GANA HOME", "1X2", sim['1x2'][0], o_h)
            add_cand("GANA AWAY", "1X2", sim['1x2'][2], o_a)
        except: pass

        # 2. Mercado Goles (Over/Under)
        try:
            o_o25 = float(odds_row.get('B365>2.5', odds_row.get('Avg>2.5', 0)))
            o_u25 = float(odds_row.get('B365<2.5', odds_row.get('Avg<2.5', 0)))
            add_cand("OVER 2.5 GOLES", "GOALS", sim['ou'][0], o_o25)
            add_cand("UNDER 2.5 GOLES", "GOALS", sim['ou'][1], o_u25)
        except: pass

        # 3. Mercados Sintéticos (Estimamos cuota si no existe, basándonos en 1X2 con margen)
        # Esto permite evaluar DNB/DC aunque no vengan en el CSV
        if o_h > 0 and o_d > 0 and o_a > 0:
            # DNB (Draw No Bet)
            margin_dnb = 0.93
            odd_dnb_h = (o_h * (1 - (1/o_d))) * margin_dnb
            odd_dnb_a = (o_a * (1 - (1/o_d))) * margin_dnb
            add_cand("DNB HOME (Sin Empate)", "DNB", sim['dnb'][0], odd_dnb_h)
            add_cand("DNB AWAY (Sin Empate)", "DNB", sim['dnb'][1], odd_dnb_a)
            
            # DC (Double Chance)
            margin_dc = 0.92
            odd_1x = (1 / ((1/o_h) + (1/o_d))) * margin_dc
            odd_x2 = (1 / ((1/o_a) + (1/o_d))) * margin_dc
            add_cand("DC 1X (Local o Empate)", "DC", sim['dc'][0], odd_1x)
            add_cand("DC X2 (Visita o Empate)", "DC", sim['dc'][1], odd_x2)

        # 4. BTTS (Si hay datos, si no estimamos aprox con cuotas de goles)
        # Nota: CSVs básicos a veces no traen BTTS, usaremos estimación si falta
        # Estimación cruda: Relacionada al Over 2.5
        # Si Over 2.5 < 1.60 -> BTTS Yes suele estar ~1.65
        try:
            # Intenta leer si existe columna BTTS (raro en csv standard, pero intentamos)
            # Si no, omitimos para no inventar demasiado
            pass 
        except: pass

        if not candidates: return None
        
        # Ordenar por SCORE (Balance EV/Prob)
        # Queremos la que tenga mejor ratio riesgo/beneficio
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0] # Retorna LA MEJOR

    def get_kelly_stake(self, prob, odds):
        if odds <= 1.0: return 0.0
        q = 1 - prob
        b = odds - 1
        kelly = (b * prob - q) / b
        return max(0.0, min(kelly * KELLY_FRACTION, MAX_STAKE_PCT))

    def run_analysis(self):
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando OMNI-MARKET SCAN: {today}", flush=True)
        
        ts = int(time.time())
        url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
        try:
            r = requests.get(url_fixt, headers={'User-Agent': USER_AGENTS[0]}, timeout=20)
            if r.status_code!=200: return
            try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
            df.columns = df.columns.str.strip().str.replace('ï»¿', '')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        except: return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = df[(df['Date'] >= target_date) & (df['Date'] <= target_date + timedelta(days=1))]
        
        bets_found = 0
        
        for idx, row in daily.iterrows():
            div = row.get('Div')
            if div not in LEAGUE_CONFIG: continue
            
            data = self.get_league_data(div)
            if not data: continue
            
            rh = difflib.get_close_matches(row['HomeTeam'], data['teams'], n=1, cutoff=0.6)
            ra = difflib.get_close_matches(row['AwayTeam'], data['teams'], n=1, cutoff=0.6)
            if not rh or not ra: continue
            rh = rh[0]; ra = ra[0]
            
            # 1. Simular Partido COMPLETO
            sim = self.simulate_match(rh, ra, data)
            
            # 2. Encontrar LA MEJOR APUESTA entre todos los mercados
            best_bet = self.find_best_value(sim, row)
            
            if best_bet:
                bets_found += 1
                stake = self.get_kelly_stake(best_bet['prob'], best_bet['odd'])
                stake_viz = "🟩" * int(stake * 100 * 2) + "⬜" * (5 - int(stake * 100 * 2))
                
                # Datos X-Ray para el reporte
                ph, pd_raw, pa = sim['1x2']
                
                msg = (
                    f"🛡️ <b>OMNI-SNIPER DETECTED</b> | {LEAGUE_CONFIG[div]['name']}\n"
                    f"⚽ <b>{rh}</b> vs {ra}\n"
                    f"───────────────\n"
                    f"🏆 <b>MEJOR OPCIÓN: {best_bet['pick']}</b>\n"
                    f"📊 Mercado: {best_bet['market']}\n"
                    f"⚖️ Cuota: <b>{self.dec_to_am(best_bet['odd'])}</b> ({best_bet['odd']:.2f})\n"
                    f"🧠 Probabilidad: <b>{best_bet['prob']*100:.1f}%</b>\n"
                    f"📈 Valor (EV): <b>+{best_bet['ev']*100:.1f}%</b>\n"
                    f"🏦 Stake: {stake_viz} ({stake*100:.2f}%)\n"
                    f"───────────────\n"
                    f"🔍 <b>CONTEXTO:</b>\n"
                    f"• 1X2: {ph*100:.0f}% | {pd_raw*100:.0f}% | {pa*100:.0f}%\n"
                    f"• Goles Exp: {sim['xg'][0]:.2f} - {sim['xg'][1]:.2f}\n"
                    f"• Over 2.5: {sim['ou'][0]*100:.0f}%\n"
                )
                
                ai_msg = self.get_ai_analysis(f"{rh} vs {ra}", f"{best_bet['pick']} @ {best_bet['odd']:.2f}")
                self.send_msg(msg + ai_msg)
                
                with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([today, div, rh, ra, best_bet['pick'], best_bet['market'], best_bet['prob'], best_bet['odd'], best_bet['ev'], stake, "PENDING", 0])

        if bets_found == 0: self.send_msg("🧹 Sin oportunidades de alto valor hoy.")

if __name__ == "__main__":
    bot = OmniSniperBot()
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
