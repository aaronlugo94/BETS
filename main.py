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
import traceback
import random # Para simulación de lineups (reemplazar con API real)
from datetime import datetime, timedelta
from collections import Counter

# --- CONFIGURACIÓN v81.0 (PERSISTENCE & SMART MATH) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "17:50" 

# AJUSTES DE MODELO
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.88          
MIN_EV_THRESHOLD = 0.02
SEASON = '2526'

# --- 💾 PERSISTENCIA RAILWAY ---
# Si existe la carpeta del volumen, guardamos ahí. Si no, local.
VOLUME_PATH = "/app/data"
if os.path.exists(VOLUME_PATH):
    HISTORY_FILE = os.path.join(VOLUME_PATH, "historial_omni_v81.csv")
    print(f"💾 USANDO VOLUMEN PERSISTENTE: {HISTORY_FILE}", flush=True)
else:
    HISTORY_FILE = "historial_omni_v81.csv"
    print("⚠️ USANDO ALMACENAMIENTO EFÍMERO (Se borrará al reiniciar)", flush=True)

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.20       
MAX_STAKE_PCT = 0.04        

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
]

# --- DIAGNÓSTICO DE IMPORTACIÓN ---
SDK_STATUS = "UNKNOWN"
try:
    from google import genai
    from google.genai import types
    SDK_AVAILABLE = True
    SDK_STATUS = "✅ LIBRERÍA 'google-genai' INSTALADA."
except ImportError as ie:
    SDK_AVAILABLE = False
    SDK_STATUS = f"❌ ERROR IMPORTACIÓN: {ie}"

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
        self.handicap_buffer = [] 
        
        print("--- ENGINE v81.0 SMART MATH STARTED ---", flush=True)
        self.send_msg(f"🔧 <b>INICIANDO v81.0</b>\n📂 Historial: {HISTORY_FILE}\nEstado SDK: {SDK_STATUS}")
        
        self._init_history_file()
        
        self.ai_client = None
        if SDK_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini SDK: Cliente Creado.", flush=True)
            except Exception as e:
                self.send_msg(f"⚠️ Error Cliente Gemini: {e}")

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Market', 'Prob', 'Odd', 'EV', 'Status', 'Stake', 'Profit', 'FTHG', 'FTAG'])

    def sanitize_for_telegram(self, text):
        text = text.replace("```html", "").replace("```", "")
        text = re.sub(r'<!DOCTYPE.*?>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<html.*?>|</html>|<head>.*?</head>|<body.*?>|</body>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = text.replace("**", "") 
        return text.strip()

    def send_msg(self, text, retry_count=0, use_html=True):
        if not TELEGRAM_TOKEN: return
        if use_html: text = self.sanitize_for_telegram(text)
        if len(text) > 4000:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            for chunk in chunks: self.send_msg(chunk, retry_count, use_html)
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML" if use_html else None}
        try:
            r = requests.post(url, json=payload, timeout=20)
            if r.status_code == 400 and use_html:
                clean_plain = re.sub(r'<[^>]+>', '', text)
                self.send_msg(clean_plain, retry_count, use_html=False)
        except Exception as e: print(f"Error Telegram: {e}", flush=True)

    def dec_to_am(self, decimal_odd):
        if decimal_odd <= 1.01: return "-10000"
        if decimal_odd >= 2.00: return f"+{int((decimal_odd - 1) * 100)}"
        else: return f"{int(-100 / (decimal_odd - 1))}"

    def call_gemini(self, prompt):
        if not SDK_AVAILABLE or not self.ai_client: return "❌ SDK no disponible."
        try:
            r = self.ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            return r.text if r.text else "⚠️ IA sin respuesta."
        except Exception as e: return f"❌ Error Gemini: {str(e)[:100]}"

    # --- MÓDULO MATEMÁTICO AVANZADO (AJUSTE POR RIVAL) ---
    
    def calculate_defense_strength_map(self, df):
        # Fase 1: Calcular fuerza defensiva base de todos los equipos
        # (Goles recibidos vs Promedio Liga)
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        avg_goals_league = (df['FTHG'].mean() + df['FTAG'].mean()) / 2
        
        def_map = {}
        for t in teams:
            matches = df[(df['HomeTeam'] == t) | (df['AwayTeam'] == t)]
            if len(matches) < 3: 
                def_map[t] = 1.0
                continue
                
            conceded = 0
            for _, r in matches.iterrows():
                if r['HomeTeam'] == t: conceded += r['FTAG']
                else: conceded += r['FTHG']
            
            avg_conceded = conceded / len(matches)
            # Def Strength: Si recibes 0.5 y la liga es 1.5 => Strength 0.33 (Muy fuerte)
            # Invertimos para que sea multiplicador de dificultad:
            # Si recibes poco, el multiplicador debe ser bajo (difícil anotar)
            strength = avg_conceded / avg_goals_league
            def_map[t] = strength
            
        return def_map

    def calculate_team_stats_weighted(self, df, team, def_map):
        # Fase 2: Calcular ataque ajustado por la defensa del rival
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(6)
        if len(matches) < 3: return 1.0, 1.0
        
        w_att = 0; w_def = 0; total_w = 0
        
        for i, (_, row) in enumerate(matches.iterrows()):
            time_weight = pow(DECAY_ALPHA, 5 - i)
            total_w += time_weight
            
            opponent = row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam']
            opp_def_strength = def_map.get(opponent, 1.0)
            
            # FACTOR DE AJUSTE POR RIVAL
            # Si le metí 2 goles al Real Madrid (Def 0.6), valen más que 2 al Almería (Def 1.8)
            # Adjusted Goal = Real Goal * (1 / Opponent_Def_Strength)
            # Limitamos el multiplicador para no locuras (entre 0.7 y 1.5)
            difficulty_mult = max(0.7, min(1.5, 1 / opp_def_strength if opp_def_strength > 0 else 1.5))
            
            if row['HomeTeam'] == team:
                raw_goals = (row['FTHG'] * 0.6) + ((row.get('HST', row['FTHG']*3)/3) * 0.4)
                raw_conc = (row['FTAG'] * 0.6) + ((row.get('AST', row['FTAG']*3)/3) * 0.4)
            else:
                raw_goals = (row['FTAG'] * 0.6) + ((row.get('AST', row['FTAG']*3)/3) * 0.4)
                raw_conc = (row['FTHG'] * 0.6) + ((row.get('HST', row['FTHG']*3)/3) * 0.4)
                
            # Aplicamos ajuste
            adjusted_att = raw_goals * difficulty_mult
            
            w_att += adjusted_att * time_weight
            w_def += raw_conc * time_weight # Defensa se deja igual por ahora
            
        return w_att / total_w, w_def / total_w

    def get_league_data(self, div):
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            if r.status_code != 200: return None
            df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
            matches_played = df.dropna(subset=['FTHG', 'FTAG'])
            
            avg_g = matches_played.FTHG.mean() + matches_played.FTAG.mean() if not matches_played.empty else 2.5
            teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            
            # 1. Mapa de defensas primero
            def_map = self.calculate_defense_strength_map(matches_played)
            
            # 2. Stats ajustadas
            team_stats = {}
            for t in teams:
                a, d = self.calculate_team_stats_weighted(matches_played, t, def_map)
                team_stats[t] = {'att': a, 'def': d}
            
            avg_att = sum(s['att'] for s in team_stats.values()) / len(team_stats)
            avg_def = sum(s['def'] for s in team_stats.values()) / len(team_stats)
            
            norm_stats = {t: {'att': s['att']/avg_att, 'def': s['def']/avg_def} for t, s in team_stats.items()}
            return {'stats': norm_stats, 'teams': teams, 'raw_df': df, 'avg_g': avg_g}
        except: return None

    # --- FILTRO DE ALINEACIONES (SIMULADO) ---
    def check_lineups_penalty(self, home, away):
        """
        NOTA: Para hacer esto REAL, necesitas una API Key de API-Football o similar.
        Como football-data.co.uk no tiene alineaciones, aquí SIMULAMOS la lógica.
        Si quieres conectarlo real, reemplaza el random por la llamada a tu API.
        """
        home_penalty = 1.0
        away_penalty = 1.0
        alert = ""
        
        # --- SIMULACIÓN (Quitar esto cuando tengas API Key) ---
        # 5% de chance de que falte una estrella
        if random.random() < 0.05:
            home_penalty = 0.85 # -15% ataque
            alert += f"🚨 {home}: Baja Clave (Simulada)\n"
        if random.random() < 0.05:
            away_penalty = 0.85
            alert += f"🚨 {away}: Baja Clave (Simulada)\n"
        # -----------------------------------------------------
        
        return home_penalty, away_penalty, alert

    def simulate_match(self, home, away, league_data, market_odds):
        h_st = league_data['stats'].get(home, {'att':1.0, 'def':1.0})
        a_st = league_data['stats'].get(away, {'att':1.0, 'def':1.0})
        
        # APLICAR PENALIZACIÓN DE ALINEACIONES
        h_pen, a_pen, lineup_alert = self.check_lineups_penalty(home, away)
        
        avg_g = league_data['avg_g'] / 2
        lambda_h = (h_st['att'] * h_pen) * a_st['def'] * avg_g * 1.10
        lambda_a = (a_st['att'] * a_pen) * h_st['def'] * avg_g
        
        h_sim = np.random.poisson(lambda_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(lambda_a, SIMULATION_RUNS)
        
        prob_h = np.mean(h_sim > a_sim); prob_d = np.mean(h_sim == a_sim); prob_a = np.mean(a_sim > h_sim)
        ov25 = np.mean((h_sim + a_sim) > 2.5)
        gcs = (0.5 + abs(ov25 - 0.5)) * 100 
        most_common = Counter(zip(h_sim, a_sim)).most_common(1)[0]
        
        return {
            '1x2': (prob_h, prob_d, prob_a), 'goals': (ov25, np.mean((h_sim>0)&(a_sim>0))),
            'dc': (prob_h+prob_d, prob_a+prob_d), 'dnb': (prob_h/(prob_h+prob_a), prob_a/(prob_h+prob_a)),
            'ah': (np.mean((h_sim+1.5)>a_sim), np.mean((a_sim+1.5)>h_sim)),
            'gcs': gcs, 'cs': most_common, 'lambdas': (lambda_h, lambda_a), 'lineup_alert': lineup_alert
        }

    # --- MOTOR MATEMÁTICO ---
    def poisson_prob(self, k, lamb): return (math.pow(lamb, k) * math.exp(-lamb)) / math.factorial(k)
    def calculate_dixon_coles_1x2(self, lambda_h, lambda_a): return 0.33, 0.33, 0.33 # Placeholder simplificado para focus en logica

    # --- SELECTOR DE VALOR ---
    def find_best_value(self, sim, odds):
        candidates = []
        best_handi = None
        
        def add(name, market, prob, odd, gcs=None):
            if odd < 1.05: return
            ev = (prob * odd) - 1
            status = "VALID"
            if ev < MIN_EV_THRESHOLD or prob < 0.35: status = "REJECTED"
            score = ev * (prob ** 1.5)
            if odd < 1.60: score *= 0.1 
            elif 1.70 <= odd <= 2.50: score *= 1.5
            candidates.append({'pick': name, 'market': market, 'prob': prob, 'odd': odd, 'ev': ev, 'score': score, 'status': status})

        if odds['H'] > 0:
            add("GANA HOME", "1X2", sim['1x2'][0], odds['H'])
            add("GANA AWAY", "1X2", sim['1x2'][2], odds['A'])
            add("DNB AWAY", "DNB", sim['dnb'][1], (odds['A']*(1-(1/odds['D'])))*0.94)
            add("DC X2", "DC", sim['dc'][1], 1/((1/odds['A'])+(1/odds['D']))*0.94)
        
        if odds['O25'] > 0:
            add("OVER 2.5", "GOALS", sim['goals'][0], odds['O25'])
            add("UNDER 2.5", "GOALS", 1-sim['goals'][0], 1/(1-(1/odds['O25']*1.05)))

        if sim['ah'][1] > 0.90: best_handi = f"Handicap A +1.5 @ 1.15"

        if not candidates: return None, best_handi
        validos = [c for c in candidates if c['status'] == "VALID"]
        if validos:
            validos.sort(key=lambda x: x['score'], reverse=True)
            return validos[0], best_handi
        candidates.sort(key=lambda x: x['ev'], reverse=True)
        return {**candidates[0], 'status': 'GHOST'}, best_handi

    # --- PnL & AUDITORÍA ---
    def check_bet_result(self, pick, market, fthg, ftag):
        if math.isnan(fthg): return "PENDING"
        hg = int(fthg); ag = int(ftag); win = False
        if market == "1X2":
            if "HOME" in pick and hg > ag: win=True
            elif "AWAY" in pick and ag > hg: win=True
        elif market == "DNB":
            if hg == ag: return "PUSH"
            if ("HOME" in pick and hg > ag) or ("AWAY" in pick and ag > hg): win=True
        elif market == "DC":
            if ("X2" in pick and ag >= hg): win=True
        elif market == "GOALS":
            if "OVER" in pick and (hg+ag) > 2.5: win=True
            elif "UNDER" in pick and (hg+ag) < 2.5: win=True
        return "WIN" if win else "LOSS"

    def run_audit(self):
        if not os.path.exists(HISTORY_FILE): return
        league_data_map = {}
        for div in LEAGUE_CONFIG.keys(): league_data_map[div] = self.get_league_data(div)
        
        rows = []; audit_buffer = []
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f); header = next(reader); rows.append(header)
            for row in reader:
                status = row[9]
                if status in ['VALID', '0'] and row[1] in league_data_map:
                    div = row[1]; home = row[2]; away = row[3]; pick = row[4]; market = row[5]; odd = float(row[7]); stake = float(row[10]) if row[10] else 0.0
                    data = league_data_map.get(div)
                    if data and not data['raw_df'].empty:
                        match = data['raw_df'][(data['raw_df']['HomeTeam']==home) & (data['raw_df']['AwayTeam']==away)]
                        if not match.empty:
                            fthg = match.iloc[0]['FTHG']; ftag = match.iloc[0]['FTAG']
                            res = self.check_bet_result(pick, market, fthg, ftag)
                            if res in ["WIN", "LOSS", "PUSH"]:
                                row[9] = res; row[12] = fthg; row[13] = ftag
                                if res == "WIN": row[11] = round((stake * odd) - stake, 2)
                                elif res == "LOSS": row[11] = round(-stake, 2)
                                audit_buffer.append(f"{pick}: {res}")
                rows.append(row)
        with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerows(rows)
        
        if audit_buffer: self.send_msg(f"🔬 <b>AUDITORÍA:</b>\n" + "\n".join(audit_buffer))

    def calculate_pnl(self):
        if not os.path.exists(HISTORY_FILE): return
        try:
            df = pd.read_csv(HISTORY_FILE)
            df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce').fillna(0)
            total = df['Profit'].sum()
            self.send_msg(f"💰 <b>PnL TOTAL:</b> {total:+.2f} U")
        except: pass

    # --- EXEC ---
    def run_analysis(self):
        self.run_audit()
        self.calculate_pnl()
        self.daily_picks_buffer = []; self.handicap_buffer = []
        today = datetime.now().strftime('%d/%m/%Y')
        try:
            df = pd.read_csv(f"https://www.football-data.co.uk/fixtures.csv?t={int(time.time())}")
            df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
            daily = df[df['Date'] == pd.to_datetime(today, dayfirst=True)]
        except: return

        self.send_msg(f"🔎 <b>Analizando {len(daily)} partidos (v81.0)...</b>")
        
        for idx, row in daily.iterrows():
            div = row.get('Div')
            if div not in LEAGUE_CONFIG: continue
            data = self.get_league_data(div)
            if not data: continue
            
            rh = difflib.get_close_matches(row['HomeTeam'], data['teams'], n=1, cutoff=0.6)
            ra = difflib.get_close_matches(row['AwayTeam'], data['teams'], n=1, cutoff=0.6)
            if not rh or not ra: continue
            rh, ra = rh[0], ra[0]
            
            m_odds = {'H': row.get('B365H',0), 'D': row.get('B365D',0), 'A': row.get('B365A',0), 'O25': row.get('B365>2.5',0)}
            sim = self.simulate_match(rh, ra, data, m_odds)
            pick, h_pick = self.find_best_value(sim, m_odds)
            
            if pick:
                status_txt = "✅ <b>PICK ACTIVO</b>" if pick['status'] == "VALID" else f"🚫 <b>NO BET</b> (Riesgo)"
                stake = self.get_kelly_stake(pick['prob'], pick['odd'], pick['market']) if pick['status'] == "VALID" else 0.0
                lineup_txt = sim.get('lineup_alert', '')
                
                msg = (f"🛡️ <b>ANÁLISIS</b> | {LEAGUE_CONFIG[div]['name']}\n"
                       f"⚽ {rh} vs {ra}\n{lineup_txt}"
                       f"───────────────\n"
                       f"{status_txt}\n"
                       f"🎯 PICK: <b>{pick['pick']}</b>\n"
                       f"⚖️ Cuota: <b>{pick['odd']:.2f}</b> | Prob: <b>{pick['prob']*100:.1f}%</b>\n"
                       f"⚔️ xG: {sim['lambdas'][0]:.2f} - {sim['lambdas'][1]:.2f}")
                self.send_msg(msg)
                
                if pick['status'] == "VALID": 
                    self.daily_picks_buffer.append(f"{rh} vs {ra}: {pick['pick']} @ {pick['odd']}")
                if h_pick: self.handicap_buffer.append(f"{rh} vs {ra}: {h_pick}")
                
                with open(HISTORY_FILE, 'a', newline='') as f:
                    csv.writer(f).writerow([today, div, rh, ra, pick['pick'], pick['market'], pick['prob'], pick['odd'], pick['ev'], pick['status'], stake, 0, "", ""])

        if self.daily_picks_buffer:
            prompt = f"Resume estos picks en HTML simple (Joya, Banker, Parlays):\n{self.daily_picks_buffer}\nSeguros:{self.handicap_buffer}"
            resp = self.call_gemini(prompt)
            self.send_msg(f"🧠 <b>DICTAMEN FINAL</b>\n\n{resp}")

if __name__ == "__main__":
    bot = OmniHybridBot()
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
