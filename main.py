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

# --- IMPORTACIÓN DE LA LIBRERÍA OFICIAL ---
try:
    from google import genai
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("❌ ERROR CRÍTICO: Debes instalar la librería: pip install google-genai", flush=True)

# --- CONFIGURACIÓN v57.0 (SDK OFFICIAL) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "21:27" 

# AJUSTES DE MODELO
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.88          
WEIGHT_GOALS = 0.60         
WEIGHT_SOT = 0.40           
SEASON = '2526'             
HISTORY_FILE = "historial_omni_hybrid.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        
MIN_EV_THRESHOLD = 0.03     

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

class OmniHybridBot:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self.daily_picks_buffer = [] 
        self._check_creds()
        self._init_history_file()
        
        # --- CLIENTE GEMINI NATIVO ---
        self.ai_client = None
        if SDK_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini SDK: CONECTADO CORRECTAMENTE", flush=True)
            except Exception as e:
                print(f"⚠️ Error conectando Gemini SDK: {e}", flush=True)

    def _check_creds(self):
        print("--- OMNI-HYBRID v57 (SDK EDITION) STARTED ---", flush=True)

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Market', 'Prob', 'Odd', 'EV', 'Result', 'Profit'])

    # --- TELEGRAM ROBUSTO ---
    def send_msg(self, text, retry_count=0):
        if not TELEGRAM_TOKEN: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        try:
            r = requests.post(url, json=payload, timeout=15)
            if r.status_code == 429:
                retry_after = int(r.json().get('parameters', {}).get('retry_after', 30))
                time.sleep(retry_after + 2)
                if retry_count < 3: self.send_msg(text, retry_count + 1)
        except: pass
        time.sleep(3.5)

    def dec_to_am(self, decimal_odd):
        if decimal_odd <= 1.01: return "-10000"
        if decimal_odd >= 2.00: return f"+{int((decimal_odd - 1) * 100)}"
        else: return f"{int(-100 / (decimal_odd - 1))}"

    # --- MOTOR GEMINI NATIVO (EL QUE TE GUSTA) ---
    def call_gemini_sdk(self, prompt_text):
        if not self.ai_client: return None
        try:
            # Usamos el modelo 2.0 Flash como en tu otro script
            response = self.ai_client.models.generate_content(
                model="gemini-2.0-flash", 
                contents=prompt_text
            )
            return response.text
        except Exception as e:
            print(f"⚠️ Error Generando IA: {e}", flush=True)
            return f"⚠️ Error IA: {str(e)}"

    # --- ANÁLISIS FINAL ---
    def generate_final_summary(self):
        if not self.daily_picks_buffer: return

        self.send_msg("⏳ <b>El Jefe de Estrategia (Gemini 2.0) está analizando los datos...</b>")

        picks_text = "\n".join(self.daily_picks_buffer)
        
        prompt = f"""
        ERES EL JEFE DE ESTRATEGIA DE UN FONDO DE INVERSIÓN DEPORTIVA.
        
        Analiza esta cartera de apuestas de hoy:
        ---
        {picks_text}
        ---
        
        Tu tarea es filtrar el ruido y darme solo lo mejor.
        
        FORMATO DE SALIDA (HTML):
        
        🧠 <b>DICTAMEN FINAL DEL DÍA</b>
        
        💎 <b>LA JOYA (Mejor Valor):</b>
        [Selecciona UN pick. Explica por qué el mercado está equivocado y nosotros tenemos la ventaja. Sé técnico.]
        
        🛡️ <b>EL BANKER (Seguridad):</b>
        [El pick más sólido para combinar o stake alto.]
        
        💣 <b>TRAMPA MORTAL:</b>
        [Advierte sobre 1 partido donde el favorito tiene riesgo de fallar.]
        
        📊 <b>Estrategia Sugerida:</b> [Ej: "Jugar planos al 2%", o "Agresivo en la Joya"]
        """
        
        ai_response = self.call_gemini_sdk(prompt)
        
        if ai_response:
            self.send_msg(ai_response)
        else:
            self.send_msg("❌ <b>Error:</b> La IA no respondió.")

    # --- MATEMÁTICAS Y LÓGICA ---
    def calculate_xg_stats(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(6)
        if len(matches) < 3: return 1.0
        w_sot = 0; w_goals = 0; total_w = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, 5 - i); total_w += weight
            if row['HomeTeam'] == team: sot = row.get('HST', row['FTHG']*3); goals = row['FTHG']
            else: sot = row.get('AST', row['FTAG']*3); goals = row['FTAG']
            w_sot += sot * weight; w_goals += goals * weight
        avg_sot = w_sot / total_w; avg_goals = w_goals / total_w
        return (avg_sot * 0.40) + (avg_goals * 0.60)

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
                xg = self.calculate_xg_stats(df, t)
                team_xgs[t] = xg; avg_xg_league += xg; cnt += 1
            avg_xg_league /= cnt
            
            norm_strength = {t: val/avg_xg_league for t, val in team_xgs.items()}
            self.history_cache[div] = {'strength': norm_strength, 'teams': teams, 'raw_df': df, 'avg_g': df.FTHG.mean()+df.FTAG.mean()}
            return self.history_cache[div]
        except: return None

    def simulate_match(self, home, away, league_data):
        s = league_data['strength']
        avg_g = league_data['avg_g'] / 2
        xg_h = min(3.2, s.get(home, 1.0) * avg_g * 1.20)
        xg_a = min(3.2, s.get(away, 1.0) * avg_g)
        
        h_sim = np.random.poisson(xg_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(xg_a, SIMULATION_RUNS)
        
        win_h = np.mean(h_sim > a_sim)
        win_a = np.mean(h_sim < a_sim)
        draw = np.mean(h_sim == a_sim)
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        over25 = np.mean((h_sim + a_sim) > 2.5)
        dc_1x = win_h + draw; dc_x2 = win_a + draw
        dnb_h = win_h / (win_h + win_a); dnb_a = win_a / (win_h + win_a)
        ah_h_m15 = np.mean((h_sim - 1.5) > a_sim); ah_h_p15 = np.mean((h_sim + 1.5) > a_sim)
        ah_a_m15 = np.mean((a_sim - 1.5) > h_sim); ah_a_p15 = np.mean((a_sim + 1.5) > h_sim)
        
        return {
            'xg': (xg_h, xg_a), '1x2': (win_h, draw, win_a), 'goals': (over25, btts),
            'dc': (dc_1x, dc_x2), 'dnb': (dnb_h, dnb_a), 'ah_h': (ah_h_m15, ah_h_p15), 'ah_a': (ah_a_m15, ah_a_p15)
        }

    def find_best_value(self, sim, odds_row):
        candidates = []
        def add(name, market, prob, odd):
            if odd < 1.05: return
            ev = (prob * odd) - 1
            score = ev * (prob * prob) 
            if ev > MIN_EV_THRESHOLD:
                candidates.append({'pick': name, 'market': market, 'prob': prob, 'odd': odd, 'ev': ev, 'score': score})

        try:
            o_h = float(odds_row.get('B365H', 0)); o_d = float(odds_row.get('B365D', 0)); o_a = float(odds_row.get('B365A', 0))
            o_o25 = float(odds_row.get('B365>2.5', 0)); o_u25 = float(odds_row.get('B365<2.5', 0))
        except: return None

        if o_h > 0:
            add("GANA HOME", "1X2", sim['1x2'][0], o_h)
            add("GANA AWAY", "1X2", sim['1x2'][2], o_a)
            add("DNB HOME", "DNB", sim['dnb'][0], (o_h * (1 - (1/o_d))) * 0.93)
            add("DNB AWAY", "DNB", sim['dnb'][1], (o_a * (1 - (1/o_d))) * 0.93)
            add("DC 1X", "Double Chance", sim['dc'][0], 1 / ((1/o_h) + (1/o_d)) * 0.92)
            add("DC X2", "Double Chance", sim['dc'][1], 1 / ((1/o_a) + (1/o_d)) * 0.92)

        if o_o25 > 0:
            add("OVER 2.5 GOLES", "GOALS", sim['goals'][0], o_o25)
            add("UNDER 2.5 GOLES", "GOALS", 1-sim['goals'][0], o_u25)
            
        o_btts_y = 1.0 / sim['goals'][1] * 0.9 if sim['goals'][1] > 0 else 0
        if o_btts_y > 1.4: add("BTTS SÍ", "BTTS", sim['goals'][1], o_btts_y)

        if not candidates: return None
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates[0]

    def get_kelly_stake(self, prob, odds):
        if odds <= 1.0: return 0.0
        q = 1 - prob; b = odds - 1
        return max(0.0, min(((b * prob - q) / b) * KELLY_FRACTION, MAX_STAKE_PCT))

    def run_analysis(self):
        self.daily_picks_buffer = [] 
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando OMNI-HYBRID SCAN: {today}", flush=True)
        
        ts = int(time.time())
        url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
        try:
            r = requests.get(url_fixt, headers={'User-Agent': USER_AGENTS[0]}, timeout=20)
            if r.status_code!=200: 
                self.send_msg(f"⚠️ Error descarga CSV: Status {r.status_code}")
                return
            try: content = r.content.decode('utf-8-sig')
            except: content = r.content.decode('latin-1')
            df = pd.read_csv(io.StringIO(content), on_bad_lines='skip')
            df.columns = df.columns.str.strip().str.replace('ï»¿', '')
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
        except Exception as e:
            self.send_msg(f"⚠️ Error crítico CSV: {e}")
            return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = df[(df['Date'] >= target_date) & (df['Date'] <= target_date + timedelta(days=1))]
        
        if daily.empty:
            self.send_msg(f"💤 <b>{today}:</b> No se encontraron partidos en la base de datos.")
            return

        bets_found = 0
        self.send_msg(f"🔎 <b>Analizando {len(daily)} partidos para hoy...</b>")
        
        for idx, row in daily.iterrows():
            div = row.get('Div')
            if div not in LEAGUE_CONFIG: continue
            
            data = self.get_league_data(div)
            if not data: continue
            
            rh = difflib.get_close_matches(row['HomeTeam'], data['teams'], n=1, cutoff=0.6)
            ra = difflib.get_close_matches(row['AwayTeam'], data['teams'], n=1, cutoff=0.6)
            if not rh or not ra: continue
            rh = rh[0]; ra = ra[0]
            
            sim = self.simulate_match(rh, ra, data)
            best_bet = self.find_best_value(sim, row)
            
            if best_bet:
                bets_found += 1
                stake = self.get_kelly_stake(best_bet['prob'], best_bet['odd'])
                stake_viz = "🟩" * int(stake * 100 * 2) + "⬜" * (5 - int(stake * 100 * 2))
                
                form_h = self.get_team_form_icon(data['raw_df'], rh)
                form_a = self.get_team_form_icon(data['raw_df'], ra)
                
                ph, pd_raw, pa = sim['1x2']
                dc1x, dcx2 = sim['dc']
                btts = sim['goals'][1]
                ov25 = sim['goals'][0]
                dnb_h, dnb_a = sim['dnb']
                ah_h_15, ah_a_15 = sim['ah_h'][0], sim['ah_a'][0]
                ah_h_p15, ah_a_p15 = sim['ah_h'][1], sim['ah_a'][1]
                
                fair_odd_us = self.dec_to_am(1/best_bet['prob'])
                
                msg = (
                    f"💎 <b>VALUE DETECTADO</b> | {LEAGUE_CONFIG[div]['name']}\n"
                    f"⚽ <b>{rh}</b> {form_h} vs {form_a} <b>{ra}</b>\n"
                    f"───────────────\n"
                    f"🎯 PICK: <b>{best_bet['pick']}</b> ({best_bet['market']})\n"
                    f"⚖️ Cuota: <b>{self.dec_to_am(best_bet['odd'])}</b> ({best_bet['odd']:.2f})\n"
                    f"🧠 Prob: <b>{best_bet['prob']*100:.1f}%</b> (Fair: {fair_odd_us})\n"
                    f"📈 EV: <b>+{best_bet['ev']*100:.1f}%</b>\n"
                    f"🏦 Stake: {stake_viz} ({stake*100:.2f}%)\n"
                    f"───────────────\n"
                    f"📊 <b>ANALÍTICA (X-RAY):</b>\n"
                    f"• 1X2: {ph*100:.0f}% | {pd_raw*100:.0f}% | {pa*100:.0f}%\n"
                    f"• D.Oport: 1X {dc1x*100:.0f}% | X2 {dcx2*100:.0f}%\n"
                    f"• BTTS: Sí {btts*100:.0f}% | No {(1-btts)*100:.0f}%\n"
                    f"• Goals 2.5: Ov {ov25*100:.0f}% | Un {(1-ov25)*100:.0f}%\n"
                    f"• DNB: H {dnb_h*100:.0f}% | A {dnb_a*100:.0f}%\n"
                    f"• AH -1.5: H {ah_h_15*100:.0f}% | A {ah_a_15*100:.0f}%\n"
                    f"• AH +1.5: H {ah_h_p15*100:.0f}% | A {ah_a_p15*100:.0f}%\n"
                    f"───────────────\n"
                    f"📊 xG: {rh} {sim['xg'][0]:.2f} - {sim['xg'][1]:.2f} {ra}"
                )
                
                self.send_msg(msg)
                
                self.daily_picks_buffer.append(
                    f"- {rh} vs {ra}: {best_bet['pick']} @ {best_bet['odd']:.2f} (EV: {best_bet['ev']*100:.1f}%)"
                )
                
                with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([today, div, rh, ra, best_bet['pick'], best_bet['market'], best_bet['prob'], best_bet['odd'], best_bet['ev'], "PENDING", 0])

        if bets_found > 0:
            self.generate_final_summary()
        else:
            self.send_msg("🧹 Barrido completado: Sin oportunidades de alto valor hoy.")

if __name__ == "__main__":
    bot = OmniHybridBot()
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
