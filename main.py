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

# --- CONFIGURACIÓN EURO-SNIPER v42.0 (ULTIMATE VALUE HUNTER) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RUN_TIME = "03:46" # Hora de ejecución diaria (UTC)

# AJUSTES DE MODELO Y GESTIÓN DE CAPITAL
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.88          # Memoria histórica (0.88 da peso a últimos 5-6 partidos)
WEIGHT_GOALS = 0.65         # Peso de Goles en cálculo de fuerza
WEIGHT_SOT = 0.35           # Peso de Tiros a Puerta (Shots on Target)
SEASON = '2526'             # Temporada 2025/2026 (Ajustado a fecha actual)
HISTORY_FILE = "historial_value_bets.csv"

# GESTIÓN DE RIESGO (KELLY CRITERION)
KELLY_FRACTION = 0.25       # 1/4 Kelly (Conservador)
MAX_STAKE_PCT = 0.03        # Máximo 3% del bankroll por apuesta
MIN_EV_THRESHOLD = 0.04     # Solo apostar si el valor esperado (EV) es > 4%

# USER AGENTS ROTATIVOS
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15'
]

# Configuración de Ligas Soportadas
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

    def _check_creds(self):
        print("--- VALUE HUNTER ENGINE v42 (STABLE) STARTED ---", flush=True)

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

    # --- MOTOR MATEMÁTICO ---
    
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
        return avg_recent / league_avg if league_avg > 0 else 1.0

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
        
        # Modelo 1: Basado en Goles
        xg_h_goals = s.loc[home, 'Att_H_G'] * s.loc[away, 'Def_A_G'] * avgs['hg'] * info[home]['form_g']
        xg_a_goals = s.loc[away, 'Att_A_G'] * s.loc[home, 'Def_H_G'] * avgs['ag'] * info[away]['form_g']
        
        # Modelo 2: Basado en Tiros a Puerta (Más estable)
        if data['has_sot']:
            xSOT_h = s.loc[home, 'Att_H_S'] * s.loc[away, 'Def_A_S'] * avgs['hst'] * info[home]['form_sot']
            xSOT_a = s.loc[away, 'Att_A_S'] * s.loc[home, 'Def_H_S'] * avgs['ast'] * info[away]['form_sot']
            
            conversion_h = xg_h_goals / xSOT_h if xSOT_h > 0 else 0.3
            conversion_a = xg_a_goals / xSOT_a if xSOT_a > 0 else 0.3
            
            final_xg_h = (xg_h_goals * WEIGHT_GOALS) + ((xSOT_h * conversion_h) * WEIGHT_SOT)
            final_xg_a = (xg_a_goals * WEIGHT_GOALS) + ((xSOT_a * conversion_a) * WEIGHT_SOT)
        else:
            final_xg_h = xg_h_goals
            final_xg_a = xg_a_goals
            
        return final_xg_h, final_xg_a

    def simulate_match(self, xg_h, xg_a):
        h_sim = np.random.poisson(xg_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(xg_a, SIMULATION_RUNS)
        
        win_h = np.mean(h_sim > a_sim)
        win_a = np.mean(h_sim < a_sim)
        draw = np.mean(h_sim == a_sim)
        
        # Corrección por sesgo de empate en ligas bajas
        if (xg_h + xg_a) < 2.35:
            adj = 0.025
            draw += adj; win_h -= adj/2; win_a -= adj/2
            
        return win_h, draw, win_a

    def get_kelly_stake(self, prob, odds):
        if odds <= 1.0: return 0.0
        q = 1 - prob
        b = odds - 1
        kelly_full = (b * prob - q) / b
        
        if kelly_full <= 0: return 0.0
        
        # Protección Dinámica de Drawdown
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
                                # Ventana de 48 horas para encontrar el partido
                                mask = (
                                    (raw['Date'] >= match_date - timedelta(days=2)) & 
                                    (raw['Date'] <= match_date + timedelta(days=2)) & 
                                    (raw['HomeTeam'] == rh)
                                )
                                match = raw[mask]
                                
                                if not match.empty:
                                    try:
                                        fthg = int(match.iloc[0]['FTHG'])
                                        ftag = int(match.iloc[0]['FTAG'])
                                        pick = row['Pick']
                                        market_odd = float(row['Market_Odd'])
                                        stake_rec = float(row['Stake_Rec'])
                                        
                                        result = "LOSS"; profit = -stake_rec
                                        
                                        if "WIN HOME" in pick and fthg > ftag:
                                            result = "WIN"; profit = (stake_rec * market_odd) - stake_rec
                                        elif "WIN AWAY" in pick and ftag > fthg:
                                            result = "WIN"; profit = (stake_rec * market_odd) - stake_rec
                                        
                                        row['Result'] = result
                                        row['Profit'] = round(profit, 4)
                                        updated = True
                                    except: pass
                rows.append(row)
        
        if updated:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print("✅ Historial actualizado.", flush=True)

    # --- EJECUCIÓN PRINCIPAL BLINDADA v42 ---
    def run_analysis(self):
        self.audit_results()

        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando Value Hunter Scan: {today}", flush=True)
        
        url_fixt = "https://www.football-data.co.uk/fixtures.csv"
        
        # Headers Anti-Bloqueo
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/csv,text/plain;q=0.9,text/html;q=0.8',
            'Referer': 'https://www.football-data.co.uk/matches.php',
            'Connection': 'keep-alive'
        }

        try:
            r = requests.get(url_fixt, headers=headers, timeout=20)
            
            if r.status_code != 200:
                self.send_msg(f"⚠️ Error HTTP descarga: {r.status_code}")
                return

            content = r.content.decode('latin-1')
            
            # Verificación de HTML (Bloqueo)
            if content.strip().startswith(('<html', '<!DOCTYPE', '<head')):
                print(f"❌ ERROR: Web devolvió HTML en lugar de CSV.", flush=True)
                self.send_msg("⚠️ Error: Bloqueo detectado (HTML Response).")
                return

            # Lectura Robusta de CSV
            try:
                fixtures = pd.read_csv(io.StringIO(content), on_bad_lines='skip')
            except:
                fixtures = pd.read_csv(io.StringIO(content), sep=None, engine='python', on_bad_lines='skip')
            
            # Limpieza de nombres de columna (Evita KeyError: 'Div')
            fixtures.columns = fixtures.columns.str.strip()
            
            if 'Div' not in fixtures.columns or 'Date' not in fixtures.columns:
                self.send_msg(f"⚠️ Error formato: No se encuentran columnas Div/Date.")
                return

            fixtures['Date'] = pd.to_datetime(fixtures['Date'], dayfirst=True, errors='coerce')
        
        except Exception as e:
            self.send_msg(f"⚠️ Excepción crítica descargando: {str(e)}")
            return

        target_date = pd.to_datetime(today, dayfirst=True)
        daily = fixtures[(fixtures['Date'] >= target_date) & (fixtures['Date'] <= target_date + timedelta(days=1))]
        
        if daily.empty:
            self.send_msg(f"💤 Sin partidos detectados para hoy ({today}).")
            return

        bets_found = 0
        
        for idx, row in daily.iterrows():
            if 'Div' not in row or pd.isna(row['Div']): continue

            div = row['Div']
            if div not in LEAGUE_CONFIG: continue
            
            home_team = row.get('HomeTeam')
            away_team = row.get('AwayTeam')
            
            # Extracción segura de cuotas
            try:
                odd_h = row.get('B365H', row.get('AvgH', 0))
                odd_d = row.get('B365D', row.get('AvgD', 0))
                odd_a = row.get('B365A', row.get('AvgA', 0))
                odd_h = float(odd_h) if odd_h else 0.0
                odd_a = float(odd_a) if odd_a else 0.0
            except: odd_h, odd_d, odd_a = 0.0, 0.0, 0.0

            if odd_h <= 1.01: continue

            data = self.get_league_data(div)
            if not data: continue
            
            real_h = difflib.get_close_matches(home_team, data['teams'], n=1, cutoff=0.6)
            real_a = difflib.get_close_matches(away_team, data['teams'], n=1, cutoff=0.6)
            if not real_h or not real_a: continue
            
            rh, ra = real_h[0], real_a[0]
            xg_h, xg_a = self.calculate_xg(rh, ra, data)
            ph, pd_raw, pa = self.simulate_match(xg_h, xg_a)
            
            ev_h = (ph * odd_h) - 1
            ev_a = (pa * odd_a) - 1
            
            best_pick = None
            if ev_h > MIN_EV_THRESHOLD:
                best_pick = {'type': 'HOME', 'team': rh, 'prob': ph, 'odd': odd_h, 'ev': ev_h}
            elif ev_a > MIN_EV_THRESHOLD:
                best_pick = {'type': 'AWAY', 'team': ra, 'prob': pa, 'odd': odd_a, 'ev': ev_a}
            
            if best_pick:
                bets_found += 1
                stake_pct = self.get_kelly_stake(best_pick['prob'], best_pick['odd'])
                
                stake_blocks = int(stake_pct * 100 * 2)
                stake_bar = "🟩" * stake_blocks + "⬜" * (5 - stake_blocks)
                
                msg = (
                    f"💎 <b>VALUE DETECTADO</b> | {LEAGUE_CONFIG[div]['name']}\n"
                    f"⚽ <b>{rh}</b> vs {ra}\n"
                    f"───────────────\n"
                    f"🎯 PICK: <b>GANA {best_pick['type']} ({best_pick['team']})</b>\n"
                    f"⚖️ Cuota: <b>{best_pick['odd']}</b>\n"
                    f"🧠 Prob: <b>{best_pick['prob']*100:.1f}%</b> (Fair: {1/best_pick['prob']:.2f})\n"
                    f"📈 <b>EV: +{best_pick['ev']*100:.1f}%</b>\n"
                    f"🏦 Stake: {stake_bar} ({stake_pct*100:.2f}%)\n"
                    f"───────────────\n"
                    f"📊 xG: {rh} {xg_h:.2f} - {xg_a:.2f} {ra}"
                )
                self.send_msg(msg)
                
                with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        today, LEAGUE_CONFIG[div]['name'], rh, ra, 
                        f"WIN {best_pick['type']}", 
                        f"{best_pick['prob']:.3f}", 
                        best_pick['odd'], 
                        f"{best_pick['ev']:.3f}", 
                        f"{stake_pct:.4f}", 
                        "PENDING", 0
                    ])

        if bets_found == 0:
            self.send_msg(f"🧹 Barrido completado: Sin valor detectado hoy (> {MIN_EV_THRESHOLD*100}% EV).")

if __name__ == "__main__":
    bot = ValueSniperBot()
    print(f"🤖 BOT VALUE HUNTER v42. Hora target: {RUN_TIME}", flush=True)
    
    # Ejecución inmediata si se define variable de entorno (para testing en deploy)
    if os.getenv("SELF_TEST", "False") == "True": 
        bot.run_analysis()
        
    schedule.every().day.at(RUN_TIME).do(bot.run_analysis)
    while True: schedule.run_pending(); time.sleep(60)
