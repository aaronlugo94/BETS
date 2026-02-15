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
    print("⚠️ ADVERTENCIA: Librería 'google-genai' no encontrada. Modo IA desactivado.", flush=True)

# --- CONFIGURACIÓN EURO-SNIPER v48.0 (STABLE LOGIC + AI) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

RUN_TIME = "05:16" 

# AJUSTES DE MODELO (MODIFICADOS PARA ESTABILIDAD)
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.88          
WEIGHT_GOALS = 0.60         # 60% Peso Goles
WEIGHT_SOT = 0.40           # 40% Peso Tiros a Puerta
SEASON = '2526'             
HISTORY_FILE = "historial_value_bets.csv"

# GESTIÓN DE RIESGO
KELLY_FRACTION = 0.25       
MAX_STAKE_PCT = 0.03        
MIN_EV_THRESHOLD = 0.04     

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
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

class ValueSniperBot:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self._check_creds()
        self._init_history_file()
        
        # --- CLIENTE GEMINI ---
        self.ai_client = None
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                self.ai_client = genai.Client(api_key=GEMINI_API_KEY)
                print("🧠 Gemini AI: CONECTADO", flush=True)
            except Exception as e:
                print(f"⚠️ Gemini AI Error Conexión: {e}", flush=True)

    def _check_creds(self):
        print("--- VALUE HUNTER ENGINE v48 (STABLE LOGIC) STARTED ---", flush=True)

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Model_Prob', 'Market_Odd', 'EV_Percent', 'Stake_Rec', 'Result', 'Profit'])

    def send_msg(self, text):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
            print(f"[TELEGRAM LOG] {text}")
            return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        try: requests.post(url, json=payload, timeout=10)
        except Exception as e: print(f"Error Telegram: {e}")

    # --- UTILIDADES ---
    def dec_to_am(self, decimal_odd):
        if decimal_odd <= 1.01: return "-10000"
        if decimal_odd >= 2.00: return f"+{int((decimal_odd - 1) * 100)}"
        else: return f"{int(-100 / (decimal_odd - 1))}"

    # --- GEMINI AI ANALYSIS ---
    def get_ai_analysis(self, raw_data_text):
        if not self.ai_client: return ""
        
        prompt = f"""
        ERES UN ANALISTA DE APUESTAS DEPORTIVAS EXPERTO.
        
        Analiza esta Value Bet detectada por mi modelo matemático:
        {raw_data_text}
        
        TU TAREA:
        Provee un "AI INSIGHT" corto (máx 3 líneas) y crítico.
        1. Valida si el xG y el EV tienen sentido futbolístico.
        2. Menciona un factor externo (lesiones, motivación, estadio) que el modelo podría estar ignorando.
        
        FORMATO HTML OBLIGATORIO:
        🤖 <b>AI INSIGHT:</b> [Tu texto aquí]
        """
        
        try:
            response = self.ai_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            return f"\n───────────────\n{response.text}"
        except Exception as e:
            print(f"Error Generando AI Insight: {e}")
            return ""

    # --- MOTOR MATEMÁTICO (CORREGIDO) ---
    def calculate_form_exponential(self, df, team, metric_col_h, metric_col_a):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(6)
        if len(matches) < 3: return 1.0
        
        weighted_val = 0; total_weight = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            weight = pow(DECAY_ALPHA, 5 - i) 
            val = row[metric_col_h] if row['HomeTeam'] == team else row[metric_col_a]
            weighted_val += val * weight
            total_weight += weight
            
        if total_weight == 0: return 1.0
        avg_recent = weighted_val / total_weight
        
        league_avg = (df[metric_col_h].mean() + df[metric_col_a].mean()) / 2
        
        # --- FIX 1: FORM CLAMPING (Evita multiplicadores locos) ---
        raw_form = avg_recent / league_avg if league_avg > 0 else 1.0
        return max(0.70, min(1.30, raw_form)) # Limitamos la forma entre 0.7 y 1.3

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers={'User-Agent': USER_AGENTS[0]}, timeout=15)
            if r.status_code != 200: return None
            
            try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
            
            df = df.dropna(subset=['FTHG', 'FTAG'])
            has_sot = 'HST' in df.columns and 'AST' in df.columns
            
            team_stats = {}
            all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
            
            avg_hg = df['FTHG'].mean(); avg_ag = df['FTAG'].mean()
            avg_hst = df['HST'].mean() if has_sot else 1.0
            avg_ast = df['AST'].mean() if has_sot else 1.0
            
            for team in all_teams:
                form_g = self.calculate_form_exponential(df, team, 'FTHG', 'FTAG')
                form_sot = self.calculate_form_exponential(df, team, 'HST', 'AST') if has_sot else 1.0
                team_stats[team] = {'form_g': form_g, 'form_sot': form_sot}

            h_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
            a_stats = df.groupby('AwayTeam')[['FTAG', 'FTHG']].mean()
            stats = pd.concat([h_stats, a_stats], axis=1)
            stats.columns = ['HG','H_Conceded','AG','A_Conceded']
            
            if has_sot:
                h_sot = df.groupby('HomeTeam')[['HST', 'AST']].mean()
                a_sot = df.groupby('AwayTeam')[['AST', 'HST']].mean()
                sot_stats = pd.concat([h_sot, a_sot], axis=1)
                sot_stats.columns = ['HST','HST_Conceded','AST','AST_Conceded']
                stats = pd.concat([stats, sot_stats], axis=1)

            stats['Att_H_G'] = stats['HG'] / avg_hg
            stats['Def_H_G'] = stats['H_Conceded'] / avg_ag
            stats['Att_A_G'] = stats['AG'] / avg_ag
            stats['Def_A_G'] = stats['A_Conceded'] / avg_hg
            
            if has_sot:
                stats['Att_H_S'] = stats['HST'] / avg_hst
                stats['Def_H_S'] = stats['HST_Conceded'] / avg_ast
                stats['Att_A_S'] = stats['AST'] / avg_ast
                stats['Def_A_S'] = stats['AST_Conceded'] / avg_hst

            data_pack = {
                'stats': stats.fillna(1.0), 
                'avgs': {'hg': avg_hg, 'ag': avg_ag, 'hst': avg_hst, 'ast': avg_ast},
                'teams': stats.index.tolist(), 
                'details': team_stats, 
                'has_sot': has_sot,
                'raw_df': df
            }
            self.history_cache[div] = data_pack
            return data_pack
        except Exception as e:
            print(f"Error procesando liga {div}: {e}")
            return None

    def calculate_xg(self, home, away, data):
        s = data['stats']; avgs = data['avgs']; info = data['details']
        
        # 1. Componente Goles (Clásico)
        xg_h_goals = s.loc[home, 'Att_H_G'] * s.loc[away, 'Def_A_G'] * avgs['hg'] * info[home]['form_g']
        xg_a_goals = s.loc[away, 'Att_A_G'] * s.loc[home, 'Def_H_G'] * avgs['ag'] * info[away]['form_g']
        
        # 2. Componente SOT (Fixed Logic)
        if data['has_sot']:
            # Ratio de conversión DE LA LIGA (No del partido, para evitar loops)
            league_conv_h = avgs['hg'] / avgs['hst'] if avgs['hst'] > 0 else 0.1
            league_conv_a = avgs['ag'] / avgs['ast'] if avgs['ast'] > 0 else 0.1
            
            # xG basado en creación de ocasiones * efectividad media
            xSOT_h = s.loc[home, 'Att_H_S'] * s.loc[away, 'Def_A_S'] * avgs['hst'] * info[home]['form_sot']
            xSOT_a = s.loc[away, 'Att_A_S'] * s.loc[home, 'Def_H_S'] * avgs['ast'] * info[away]['form_sot']
            
            xg_from_sot_h = xSOT_h * league_conv_h
            xg_from_sot_a = xSOT_a * league_conv_a
            
            final_xg_h = (xg_h_goals * WEIGHT_GOALS) + (xg_from_sot_h * WEIGHT_SOT)
            final_xg_a = (xg_a_goals * WEIGHT_GOALS) + (xg_from_sot_a * WEIGHT_SOT)
        else:
            final_xg_h = xg_h_goals
            final_xg_a = xg_a_goals
        
        # --- FIX 2: REALISTIC CAP (Evita 99% probs) ---
        final_xg_h = min(3.85, final_xg_h)
        final_xg_a = min(3.85, final_xg_a)
            
        return final_xg_h, final_xg_a

    def simulate_match(self, xg_h, xg_a):
        h_sim = np.random.poisson(xg_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(xg_a, SIMULATION_RUNS)
        
        win_h = np.mean(h_sim > a_sim)
        win_a = np.mean(h_sim < a_sim)
        draw = np.mean(h_sim == a_sim)
        
        # Ajuste empate ligas under
        if (xg_h + xg_a) < 2.35:
            adj = 0.025
            draw += adj; win_h -= adj/2; win_a -= adj/2
            
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        over25 = np.mean((h_sim + a_sim) > 2.5)
        dc_1x = win_h + draw
        dc_x2 = win_a + draw
        
        total_win = win_h + win_a
        dnb_h = win_h / total_win if total_win > 0 else 0
        dnb_a = win_a / total_win if total_win > 0 else 0
        
        ah_h_minus15 = np.mean((h_sim - 1.5) > a_sim)
        ah_h_plus15 = np.mean((h_sim + 1.5) > a_sim)
        ah_a_minus15 = np.mean((a_sim - 1.5) > h_sim)
        ah_a_plus15 = np.mean((a_sim + 1.5) > h_sim)

        return {
            '1x2': (win_h, draw, win_a),
            'goals': (over25, btts),
            'dc': (dc_1x, dc_x2),
            'dnb': (dnb_h, dnb_a),
            'ah_h': (ah_h_minus15, ah_h_plus15),
            'ah_a': (ah_a_minus15, ah_a_plus15)
        }

    def get_kelly_stake(self, prob, odds):
        if odds <= 1.0: return 0.0
        q = 1 - prob
        b = odds - 1
        kelly_full = (b * prob - q) / b
        if kelly_full <= 0: return 0.0
        
        drawdown_factor = 1.0
        if os.path.exists(HISTORY_FILE):
            try:
                hist = pd.read_csv(HISTORY_FILE)
                if not hist.empty:
                    last_5 = hist.tail(5)
                    losses = last_5[last_5['Profit'] < 0].shape[0]
                    if losses >= 4: drawdown_factor = 0.5
            except: pass

        final_stake = kelly_full * KELLY_FRACTION * drawdown_factor
        return min(final_stake, MAX_STAKE_PCT)

    def audit_results(self):
        if not os.path.exists(HISTORY_FILE): return
        rows = []
        updated = False
        print("🔎 Auditando resultados...", flush=True)
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row['Result'] == 'PENDING':
                    div = None
                    for code, cfg in LEAGUE_CONFIG.items():
                        if cfg['name'] == row['League']: div = code; break
                    if div:
                        data = self.get_league_data(div)
                        if data:
                            raw = data['raw_df']
                            match_date = pd.to_datetime(row['Date'], dayfirst=True)
                            real_home = difflib.get_close_matches(row['Home'], data['teams'], n=1, cutoff=0.6)
                            if real_home:
                                rh = real_home[0]
                                mask = ((raw['Date'] >= match_date - timedelta(days=2)) & (raw['Date'] <= match_date + timedelta(days=2)) & (raw['HomeTeam'] == rh))
                                match = raw[mask]
                                if not match.empty:
                                    try:
                                        fthg = int(match.iloc[0]['FTHG']); ftag = int(match.iloc[0]['FTAG'])
                                        pick = row['Pick']; market_odd = float(row['Market_Odd']); stake_rec = float(row['Stake_Rec'])
                                        result = "LOSS"; profit = -stake_rec
                                        if "WIN HOME" in pick and fthg > ftag: result = "WIN"; profit = (stake_rec * market_odd) - stake_rec
                                        elif "WIN AWAY" in pick and ftag > fthg: result = "WIN"; profit = (stake_rec * market_odd) - stake_rec
                                        row['Result'] = result; row['Profit'] = round(profit, 4); updated = True
                                    except: pass
                rows.append(row)
        if updated:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print("✅ Auditoría completada.", flush=True)

    # --- EJECUCIÓN PRINCIPAL ---
    def run_analysis(self):
        self.audit_results()
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando Value Hunter Scan: {today}", flush=True)
        
        ts = int(time.time())
        url_fixt = f"https://www.football-data.co.uk/fixtures.csv?t={ts}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,text/plain;q=0.9,text/html;q=0.8',
            'Referer': 'https://www.football-data.co.uk/matches.php',
            'Connection': 'keep-alive'
        }

        try:
            print(f"📡 Descargando desde {url_fixt}...", flush=True)
            r = requests.get(url_fixt, headers=headers, timeout=25)
            if r.status_code != 200:
                self.send_msg(f"⚠️ Error HTTP descarga: {r.status_code}"); return
            
            try: content = r.content.decode('utf-8-sig')
            except: content = r.content.decode('latin-1')

            try: fixtures = pd.read_csv(io.StringIO(content), on_bad_lines='skip')
            except: fixtures = pd.read_csv(io.StringIO(content), sep=None, engine='python', on_bad_lines='skip')
            
            fixtures.columns = fixtures.columns.str.strip().str.replace('ï»¿', '')
            if 'Div' not in fixtures.columns or 'Date' not in fixtures.columns:
                print(f"❌ COLUMNAS: {fixtures.columns.tolist()}", flush=True)
                self.send_msg(f"⚠️ Error formato crítico."); return

            fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True, errors='coerce')
        
        except Exception as e:
            self.send_msg(f"⚠️ Excepción sistémica: {str(e)}"); return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = fixtures[(fixtures['Date'] >= target_date) & (fixtures['Date'] <= target_date + timedelta(days=1))]
        
        if daily.empty:
            self.send_msg(f"💤 Sin partidos detectados."); return

        bets_found = 0
        print(f"⚽ Analizando {len(daily)} partidos...", flush=True)
        
        for idx, row in daily.iterrows():
            if 'Div' not in row or pd.isna(row['Div']): continue
            div = row['Div']
            if div not in LEAGUE_CONFIG: continue
            
            home_team = row.get('HomeTeam'); away_team = row.get('AwayTeam')
            try:
                odd_h = float(row.get('B365H', row.get('AvgH', 0)) or 0)
                odd_a = float(row.get('B365A', row.get('AvgA', 0)) or 0)
            except: odd_h, odd_a = 0.0, 0.0

            if odd_h <= 1.01: continue
            data = self.get_league_data(div)
            if not data: continue
            
            real_h = difflib.get_close_matches(home_team, data['teams'], n=1, cutoff=0.6)
            real_a = difflib.get_close_matches(away_team, data['teams'], n=1, cutoff=0.6)
            if not real_h or not real_a: continue
            
            rh, ra = real_h[0], real_a[0]
            xg_h, xg_a = self.calculate_xg(rh, ra, data)
            sim_res = self.simulate_match(xg_h, xg_a)
            ph, pd_raw, pa = sim_res['1x2']
            
            ev_h = (ph * odd_h) - 1; ev_a = (pa * odd_a) - 1
            best_pick = None
            if ev_h > MIN_EV_THRESHOLD: best_pick = {'type': 'HOME', 'team': rh, 'prob': ph, 'odd': odd_h, 'ev': ev_h}
            elif ev_a > MIN_EV_THRESHOLD: best_pick = {'type': 'AWAY', 'team': ra, 'prob': pa, 'odd': odd_a, 'ev': ev_a}
            
            if best_pick:
                bets_found += 1
                stake_pct = self.get_kelly_stake(best_pick['prob'], best_pick['odd'])
                stake_blocks = int(stake_pct * 100 * 2)
                stake_bar = "🟩" * stake_blocks + "⬜" * (5 - stake_blocks)
                
                odd_am = self.dec_to_am(best_pick['odd'])
                fair_odd_am = self.dec_to_am(1/best_pick['prob'])
                
                ov25, btts = sim_res['goals']
                dc1x, dcx2 = sim_res['dc']
                dnb_h, dnb_a = sim_res['dnb']
                ah_h_n, ah_h_p = sim_res['ah_h']
                ah_a_n, ah_a_p = sim_res['ah_a']
                
                msg = (
                    f"💎 <b>VALUE DETECTADO</b> | {LEAGUE_CONFIG[div]['name']}\n"
                    f"⚽ <b>{rh}</b> vs {ra}\n"
                    f"───────────────\n"
                    f"🎯 PICK: <b>GANA {best_pick['type']} ({best_pick['team']})</b>\n"
                    f"⚖️ Cuota: <b>{odd_am}</b> ({best_pick['odd']})\n"
                    f"🧠 Prob: <b>{best_pick['prob']*100:.1f}%</b> (Fair: {fair_odd_am})\n"
                    f"📈 <b>EV: +{best_pick['ev']*100:.1f}%</b>\n"
                    f"🏦 Stake: {stake_bar} ({stake_pct*100:.2f}%)\n"
                    f"───────────────\n"
                    f"📊 <b>ANALÍTICA (X-RAY):</b>\n"
                    f"• 1X2: {ph*100:.0f}% | {pd_raw*100:.0f}% | {pa*100:.0f}%\n"
                    f"• D.Oport: 1X {dc1x*100:.0f}% | X2 {dcx2*100:.0f}%\n"
                    f"• BTTS: Sí {btts*100:.0f}% | No {(1-btts)*100:.0f}%\n"
                    f"• Goals 2.5: Ov {ov25*100:.0f}% | Un {(1-ov25)*100:.0f}%\n"
                    f"• DNB: H {dnb_h*100:.0f}% | A {dnb_a*100:.0f}%\n"
                    f"• AH -1.5: H {ah_h_n*100:.0f}% | A {ah_a_n*100:.0f}%\n"
                    f"• AH +1.5: H {ah_h_p*100:.0f}% | A {ah_a_p*100:.0f}%\n"
                    f"───────────────\n"
                    f"📊 xG: {rh} {xg_h:.2f} - {xg_a:.2f} {ra}"
                )
                
                # --- GEMINI INJECTION SAFE ---
                ai_insight = self.get_ai_analysis(msg)
                full_msg = msg + ai_insight
                self.send_msg(full_msg)
                
                with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([today, LEAGUE_CONFIG[div]['name'], rh, ra, f"WIN {best_pick['type']}", f"{best_pick['prob']:.3f}", best_pick['odd'], f"{best_pick['ev']:.3f}", f"{stake_pct:.4f}", "PENDING", 0])

        if bets_found == 0:
            self.send_msg(f"🧹 Barrido completado: Sin valor detectado hoy (> {MIN_EV_THRESHOLD*100}% EV).")

if __name__ == "__main__":
    bot = ValueSniperBot()
    print(f"🤖 BOT VALUE HUNTER v48. Hora target: {RUN_TIME}", flush=True)
    if os.getenv("SELF_TEST", "False") == "True": bot.run_analysis()
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
