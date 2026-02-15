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

# --- CONFIGURACIÓN EURO-SNIPER v50.0 (ASIAN HANDICAP MASTER) ---

try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Gemini no detectado. Ejecutando en modo Matemático Puro.")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "05:57" 

# AJUSTES DE MODELO (SHOT-BASED XG)
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.89          # Mayor memoria para detectar tendencias de tiro
WEIGHT_SHOTS_ON_TARGET = 0.55 # El dato más importante
WEIGHT_GOALS = 0.30         # Goles reales
WEIGHT_CORNERS = 0.15       # Presión ofensiva

SEASON = '2526'             
HISTORY_FILE = "historial_asian_bets.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       # Más conservador para mercados asiáticos
MAX_STAKE_PCT = 0.04        # Max 4% del bank
MIN_EV_THRESHOLD = 0.03     # Min 3% EV

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

class AsianSniperBot:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self._check_creds()
        self._init_history_file()
        
        self.ai_client = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini AI: ACTIVO (Modo Analista Asiático)", flush=True)
            except: pass

    def _check_creds(self):
        print("--- ASIAN SNIPER ENGINE v50 STARTED ---", flush=True)

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick_Type', 'Line', 'Prob', 'Odd', 'EV', 'Stake', 'Result', 'Profit'])

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

    def get_ai_analysis(self, match_info, pick_info):
        if not self.ai_client: return ""
        prompt = f"""
        ANALISTA ASIÁTICO EXPERTO.
        
        Partido: {match_info}
        Mi Modelo sugiere: {pick_info}
        
        Tu tarea:
        1. ¿Es el mercado seleccionado (DNB, AH, etc.) más seguro que el 1X2 directo?
        2. Menciona un riesgo táctico brevemente.
        
        Formato HTML:
        🤖 <b>AI TACTICS:</b> [Texto]
        """
        try:
            response = self.ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return f"\n───────────────\n{response.text}"
        except: return ""

    # --- MOTOR XG AVANZADO (SHOT-BASED) ---
    def calculate_advanced_stats(self, df, team):
        # Filtramos partidos donde jugó el equipo
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(8)
        if len(matches) < 3: return {'att_strength': 1.0, 'def_strength': 1.0}
        
        w_sot = 0; w_goals = 0; w_corners = 0; total_w = 0
        
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, 7 - i)
            total_w += weight
            
            if row['HomeTeam'] == team:
                # Usamos HST (Home Shots on Target) y HC (Home Corners)
                sot = row.get('HST', row['FTHG'] * 3) # Fallback si no hay datos de tiros
                goals = row['FTHG']
                corners = row.get('HC', 4)
            else:
                sot = row.get('AST', row['FTAG'] * 3)
                goals = row['FTAG']
                corners = row.get('AC', 4)
                
            w_sot += sot * weight
            w_goals += goals * weight
            w_corners += corners * weight
            
        avg_sot = w_sot / total_w
        avg_goals = w_goals / total_w
        avg_corners = w_corners / total_w
        
        # Fórmula Maestra de xG Aproximado
        # Un tiro a puerta vale mucho más que un gol fortuito
        # Un corner indica presión constante
        xg_proxy = (avg_sot * 0.35) + (avg_goals * 0.50) + (avg_corners * 0.05)
        
        return xg_proxy

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            if r.status_code != 200: return None
            try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
            
            df = df.dropna(subset=['FTHG', 'FTAG'])
            
            # Pre-cálculo de medias de la liga para normalizar
            avg_xg_league = 0
            cnt = 0
            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            team_xgs = {}
            
            for t in teams:
                xg = self.calculate_advanced_stats(df, t)
                team_xgs[t] = xg
                avg_xg_league += xg
                cnt += 1
            
            avg_xg_league /= cnt
            
            # Normalizar fuerzas
            normalized_strength = {t: val/avg_xg_league for t, val in team_xgs.items()}
            
            self.history_cache[div] = {'df': df, 'strength': normalized_strength, 'teams': teams, 'avg_goals': df.FTHG.mean() + df.FTAG.mean()}
            return self.history_cache[div]
        except: return None

    def simulate_match(self, home, away, league_data):
        s = league_data['strength']
        avg_g = league_data['avg_goals'] / 2 # Goles promedio por equipo
        
        # xG Proyectado (Ataque Local * Defensa Visitante * Factor Campo)
        # Nota: Simplificado para el ejemplo, idealmente separaríamos Atk/Def
        home_adv = 1.20 # Factor de campo
        
        xg_h = s.get(home, 1.0) * avg_g * home_adv
        xg_a = s.get(away, 1.0) * avg_g
        
        # Simulación Monte Carlo
        h_sim = np.random.poisson(xg_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(xg_a, SIMULATION_RUNS)
        
        diff = h_sim - a_sim
        
        # Probabilidades Base
        win_h = np.mean(diff > 0)
        win_a = np.mean(diff < 0)
        draw = np.mean(diff == 0)
        
        # --- CÁLCULO DE MERCADOS ASIÁTICOS ---
        
        # 1. Asian Handicap 0.0 (DNB)
        # Si empatan, se devuelve el dinero (no cuenta como apuesta perdida ni ganada en probabilidad pura)
        # Prob Ajustada = Win / (Win + Loss)
        dnb_h_prob = win_h / (win_h + win_a)
        dnb_a_prob = win_a / (win_h + win_a)
        
        # 2. Asian Handicap -0.25 / +0.25
        # H -0.25: Ganas si gana H. Pierdes mitad si empata.
        # Eq: (Win_H) + (0.5 * Draw * Refund_Factor) -> Complejo, simplificamos a probabilidad de éxito
        
        # 3. Asian Handicap +0.5 (Doble Oportunidad)
        ah_h_plus05 = win_h + draw # 1X
        ah_a_plus05 = win_a + draw # X2
        
        return {
            'xg': (xg_h, xg_a),
            '1x2': (win_h, draw, win_a),
            'dnb': (dnb_h_prob, dnb_a_prob),
            'dc': (ah_h_plus05, ah_a_plus05)
        }

    def select_best_market(self, sim_res, odds_1x2):
        # odds_1x2 = {'H': 2.10, 'D': 3.40, 'A': 3.50}
        
        best_pick = None
        best_ev = MIN_EV_THRESHOLD
        
        wh, dr, wa = sim_res['1x2']
        
        # A. Evaluar 1X2 Directo
        ev_h = (wh * odds_1x2['H']) - 1
        ev_a = (wa * odds_1x2['A']) - 1
        
        if ev_h > best_ev: 
            best_pick = {'type': 'WIN HOME', 'line': '1X2', 'prob': wh, 'odd': odds_1x2['H'], 'ev': ev_h}
            best_ev = ev_h
        if ev_a > best_ev:
            best_pick = {'type': 'WIN AWAY', 'line': '1X2', 'prob': wa, 'odd': odds_1x2['A'], 'ev': ev_a}
            best_ev = ev_a
            
        # B. Evaluar DNB (Estimamos Cuota DNB basándonos en 1x2 con margen)
        # DNB Odd approx = Odd / (1 - (1/Odd_Draw))
        odd_dnb_h = odds_1x2['H'] * (1 - (1/odds_1x2['D'])) * 0.92 # 0.92 margen de bookie
        odd_dnb_a = odds_1x2['A'] * (1 - (1/odds_1x2['D'])) * 0.92
        
        ev_dnb_h = (sim_res['dnb'][0] * odd_dnb_h) - 1
        ev_dnb_a = (sim_res['dnb'][1] * odd_dnb_a) - 1
        
        # Lógica de Preferencia: Si el EV es similar, PREFERIMOS DNB (Menor Varianza)
        risk_tolerance = 0.02 # Sacrificamos 2% de EV por seguridad
        
        if ev_dnb_h > (best_ev - risk_tolerance) and ev_dnb_h > MIN_EV_THRESHOLD:
            best_pick = {'type': 'DNB HOME', 'line': '0.0 (Sin Empate)', 'prob': sim_res['dnb'][0], 'odd': odd_dnb_h, 'ev': ev_dnb_h}
            best_ev = ev_dnb_h
            
        if ev_dnb_a > (best_ev - risk_tolerance) and ev_dnb_a > MIN_EV_THRESHOLD:
            best_pick = {'type': 'DNB AWAY', 'line': '0.0 (Sin Empate)', 'prob': sim_res['dnb'][1], 'odd': odd_dnb_a, 'ev': ev_dnb_a}
            best_ev = ev_dnb_a

        # C. Evaluar Doble Oportunidad (Para Underdogs)
        # Si la prob de ganar es baja (<35%) pero la de no perder es alta
        if wh < 0.35 and sim_res['dc'][0] > 0.60:
            odd_dc_h = 1 / ((1/odds_1x2['H']) + (1/odds_1x2['D'])) * 0.90
            ev_dc_h = (sim_res['dc'][0] * odd_dc_h) - 1
            if ev_dc_h > best_ev:
                best_pick = {'type': 'DC 1X', 'line': '+0.5 (Doble Op)', 'prob': sim_res['dc'][0], 'odd': odd_dc_h, 'ev': ev_dc_h}
        
        return best_pick

    def get_kelly_stake(self, prob, odds):
        if odds <= 1.0: return 0.0
        q = 1 - prob
        b = odds - 1
        kelly = (b * prob - q) / b
        return max(0.0, min(kelly * KELLY_FRACTION, MAX_STAKE_PCT))

    def run_analysis(self):
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando ASIAN MARKET SCAN: {today}", flush=True)
        
        ts = int(time.time())
        url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
        try:
            r = requests.get(url_fixt, headers={'User-Agent': USER_AGENTS[0]}, timeout=20)
            if r.status_code!=200: return
            df = pd.read_csv(io.StringIO(r.content.decode('latin-1')), on_bad_lines='skip')
            df.columns = df.columns.str.strip()
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        except: return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = df[(df['Date'] >= target_date) & (df['Date'] <= target_date + timedelta(days=1))]
        
        bets_found = 0
        
        for idx, row in daily.iterrows():
            div = row.get('Div')
            if div not in LEAGUE_CONFIG: continue
            
            # Extraer Odds 1X2 Base
            try:
                odds = {
                    'H': float(row.get('B365H', row.get('AvgH', 0))),
                    'D': float(row.get('B365D', row.get('AvgD', 0))),
                    'A': float(row.get('B365A', row.get('AvgA', 0)))
                }
            except: continue
            
            if odds['H'] < 1.05: continue # Basura
            
            # Análisis
            data = self.get_league_data(div)
            if not data: continue
            
            rh = difflib.get_close_matches(row['HomeTeam'], data['teams'], n=1, cutoff=0.6)
            ra = difflib.get_close_matches(row['AwayTeam'], data['teams'], n=1, cutoff=0.6)
            if not rh or not ra: continue
            
            rh = rh[0]; ra = ra[0]
            
            # Simulación
            sim = self.simulate_match(rh, ra, data)
            
            # SELECCIÓN DEL MEJOR MERCADO (EL CEREBRO NUEVO)
            pick = self.select_best_market(sim, odds)
            
            if pick:
                bets_found += 1
                stake = self.get_kelly_stake(pick['prob'], pick['odd'])
                stake_viz = "🟦" * int(stake * 100 * 2)
                
                # Reporte
                msg = (
                    f"🐉 <b>ASIAN SNIPER</b> | {LEAGUE_CONFIG[div]['name']}\n"
                    f"⚽ <b>{rh}</b> vs {ra}\n"
                    f"───────────────\n"
                    f"🛡️ <b>MERCADO ÓPTIMO: {pick['type']}</b>\n"
                    f"⚖️ Línea: <b>{pick['line']}</b> @ {pick['odd']:.2f} ({self.dec_to_am(pick['odd'])})\n"
                    f"🧠 Prob Real: <b>{pick['prob']*100:.1f}%</b>\n"
                    f"📈 Valor (EV): <b>+{pick['ev']*100:.1f}%</b>\n"
                    f"💰 Stake: {stake_viz} ({stake*100:.2f}%)\n"
                    f"───────────────\n"
                    f"📊 xG Real: {rh} {sim['xg'][0]:.2f} - {sim['xg'][1]:.2f} {ra}"
                )
                
                # Gemini Táctico
                ai_msg = self.get_ai_analysis(f"{rh} vs {ra}", f"{pick['type']} @ {pick['odd']}")
                self.send_msg(msg + ai_msg)
                
                # Log
                with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([today, div, rh, ra, pick['type'], pick['line'], pick['prob'], pick['odd'], pick['ev'], stake, "PENDING", 0])

        if bets_found == 0: self.send_msg("🧘 Paciencia. Sin valor en mercados asiáticos hoy.")

if __name__ == "__main__":
    bot = AsianSniperBot()
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
