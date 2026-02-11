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

# --- CONFIGURACIÓN EURO-SNIPER v33.0 (TITANIUM) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
RUN_TIME = "03:46" # UTC (20:00 Tucson)

# AJUSTES MATEMÁTICOS
SIMULATION_RUNS = 100000 
DECAY_ALPHA = 0.85 
SEASON = '2526'         
HISTORY_FILE = "historial_picks.csv"

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Wget/1.20.3 (linux-gnu)'
]

LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER', 'tier': 1, 'threshold': 0.60},
    'SP1': {'name': '🇪🇸 LA LIGA', 'tier': 1, 'threshold': 0.60},
    'I1':  {'name': '🇮🇹 SERIE A', 'tier': 1, 'threshold': 0.58},
    'D1':  {'name': '🇩🇪 BUNDES',  'tier': 1, 'threshold': 0.60},
    'F1':  {'name': '🇫🇷 LIGUE 1', 'tier': 1, 'threshold': 0.58},
    'P1':  {'name': '🇵🇹 PORTUGAL','tier': 2, 'threshold': 0.65},
    'N1':  {'name': '🇳🇱 HOLANDA', 'tier': 2, 'threshold': 0.65},
    'B1':  {'name': '🇧🇪 BELGICA', 'tier': 2, 'threshold': 0.65},
    'T1':  {'name': '🇹🇷 TURQUIA', 'tier': 2, 'threshold': 0.65}
}

class TelegramSniper:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self._check_creds()
        self._init_history_file()

    def _check_creds(self):
        print("--- TITANIUM ENGINE STARTED ---", flush=True)
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: 
            print("❌ ERROR: Credenciales faltantes", flush=True)

    def _init_history_file(self):
        if not os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Date', 'League', 'Home', 'Away', 'Pick', 'Confidence', 'Fair_Odd_US', 'Result', 'Profit'])

    def decimal_to_american(self, decimal):
        if decimal <= 1.01: return -10000
        if decimal >= 2.00: return f"+{int((decimal - 1) * 100)}"
        else: return f"{int(-100 / (decimal - 1))}"

    def send_msg(self, text):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        for i in range(3):
            try: requests.post(url, json=payload, timeout=10); return
            except: time.sleep(1)

    def send_file(self):
        if not os.path.exists(HISTORY_FILE): return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
        try:
            with open(HISTORY_FILE, 'rb') as f:
                requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"document": f})
        except Exception as e: print(f"Error enviando archivo: {e}", flush=True)

    def calculate_exponential_form(self, df, team):
        matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
        if len(matches) == 0: return 1.0
        weighted_points = 0; total_weight = 0
        for i, (_, row) in enumerate(matches.iterrows()):
            if row['HomeTeam'] == team: pts = 3 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 0)
            else: pts = 3 if row['FTAG'] > row['FTHG'] else (1 if row['FTAG'] == row['FTHG'] else 0)
            weight = pow(DECAY_ALPHA, 4 - i) 
            weighted_points += pts * weight
            total_weight += weight
        if total_weight == 0: return 1.0
        weighted_avg = weighted_points / total_weight
        return 1.0 + ((weighted_avg / 3.0 - 0.5) * 2 * 0.25)

    def load_fixtures(self):
        print(f"[SYSTEM] Updating Database...", flush=True)
        urls = ["https://www.football-data.co.uk/fixtures.csv", "http://www.football-data.co.uk/fixtures.csv"]
        for i, url in enumerate(urls):
            try:
                headers = {'User-Agent': USER_AGENTS[i % len(USER_AGENTS)]}
                r = requests.get(url, headers=headers, timeout=(5, 10))
                try: content = r.content.decode('utf-8-sig')
                except: content = r.content.decode('latin-1')
                self.fixtures = pd.read_csv(io.StringIO(content))
                if not self.fixtures.empty:
                    self.fixtures.rename(columns={self.fixtures.columns[0]: 'Div'}, inplace=True)
                    self.fixtures.columns = self.fixtures.columns.str.strip()
                    self.fixtures = self.fixtures.dropna(subset=['Div'])
                    self.fixtures['Date'] = pd.to_datetime(self.fixtures['Date'], dayfirst=True, errors='coerce')
                    return True
            except: time.sleep(2)
        return False

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        for i in range(2):
            try:
                headers = {'User-Agent': USER_AGENTS[0]}
                r = requests.get(url, headers=headers, timeout=(5, 10))
                try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
                except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
                df = df.dropna(subset=['FTHG', 'FTAG'])
                
                team_stats = {}
                all_teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
                for team in all_teams:
                    form_factor = self.calculate_exponential_form(df, team)
                    team_stats[team] = {'form': form_factor}

                avg_h = df['FTHG'].mean(); avg_a = df['FTAG'].mean()
                h_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
                a_stats = df.groupby('AwayTeam')[['FTAG', 'FTHG']].mean()
                stats = pd.concat([h_stats, a_stats], axis=1)
                stats.columns = ['HS','HC','AS','AC']
                stats['Att_H'] = stats['HS'] / avg_h; stats['Def_H'] = stats['HC'] / avg_a
                stats['Att_A'] = stats['AS'] / avg_a; stats['Def_A'] = stats['AC'] / avg_h
                
                data_pack = {'stats': stats.fillna(1.0), 'avgs': {'h': avg_h, 'a': avg_a}, 'teams': stats.index.tolist(), 'details': team_stats, 'raw_df': df}
                self.history_cache[div] = data_pack
                return data_pack
            except: time.sleep(1)
        return None

    def find_team(self, team, team_list):
        matches = difflib.get_close_matches(team, team_list, n=1, cutoff=0.5)
        return matches[0] if matches else None

    def audit_history(self):
        if not os.path.exists(HISTORY_FILE): return
        rows = []
        updated = False
        wins = 0; losses = 0; profit_units = 0.0
        
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
                            real_home = self.find_team(row['Home'], data['teams'])
                            mask = ((raw['Date'] >= match_date - timedelta(days=1)) & (raw['Date'] <= match_date + timedelta(days=1)) & (raw['HomeTeam'] == real_home))
                            match = raw[mask]
                            
                            if not match.empty:
                                updated = True
                                fthg = match.iloc[0]['FTHG']; ftag = match.iloc[0]['FTAG']
                                pick = row['Pick']
                                
                                odd_us = str(row['Fair_Odd_US'])
                                if odd_us.startswith("+"): odd_dec = (int(odd_us[1:]) / 100) + 1
                                else: odd_dec = (100 / abs(int(odd_us))) + 1
                                
                                result = "LOSS"; pnl = -1.0 
                                if "GANA LOCAL" in pick and fthg > ftag: result = "WIN"; pnl = odd_dec - 1
                                elif "GANA VISITA" in pick and ftag > fthg: result = "WIN"; pnl = odd_dec - 1
                                elif "1X" in pick and fthg >= ftag: result = "WIN"; pnl = (odd_dec - 1) * 0.5 
                                elif "X2" in pick and ftag >= fthg: result = "WIN"; pnl = (odd_dec - 1) * 0.5
                                elif "OVER" in pick and (fthg+ftag) > 2.5: result = "WIN"; pnl = 0.9
                                elif "BTTS" in pick and (fthg > 0 and ftag > 0): result = "WIN"; pnl = 0.9
                                
                                row['Result'] = result; row['Profit'] = round(pnl, 2)
                                if result == "WIN": wins += 1; profit_units += pnl
                                else: losses += 1; profit_units -= 1.0
                rows.append(row)
        
        if updated:
            with open(HISTORY_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            emoji = "🤑" if profit_units > 0 else "📉"
            self.send_msg(f"📝 <b>AUDITORÍA</b>\n✅ {wins} | ❌ {losses}\n{emoji} Balance: <b>{profit_units:+.2f} U</b>")
            self.send_file()

    def analyze_match(self, home, away, div):
        data = self.get_league_data(div)
        if not data: return None
        rh = self.find_team(home, data['teams']); ra = self.find_team(away, data['teams'])
        if not rh or not ra: return None
        
        s = data['stats']; avgs = data['avgs']; info = data['details']
        xg_h = (s.loc[rh, 'Att_H'] * info[rh]['form']) * s.loc[ra, 'Def_A'] * avgs['h']
        xg_a = (s.loc[ra, 'Att_A'] * info[ra]['form']) * s.loc[rh, 'Def_H'] * avgs['a']
        
        h_sim = np.random.poisson(xg_h, SIMULATION_RUNS)
        a_sim = np.random.poisson(xg_a, SIMULATION_RUNS)
        
        win_h = np.mean(h_sim > a_sim); win_a = np.mean(h_sim < a_sim)
        draw = np.mean(h_sim == a_sim)
        if (xg_h + xg_a) < 2.40: 
            boost = 0.03; draw += boost; win_h -= boost/2; win_a -= boost/2
        
        over25 = np.mean((h_sim + a_sim) > 2.5)
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        
        # HANDICAP CALCULATION
        # Prob de que Local gane por más de 1.5 goles (Goleada)
        ah_h = np.mean((h_sim - 1.5) > a_sim) 
        # Prob de que Visita gane por más de 1.5 goles
        ah_a = np.mean((a_sim - 1.5) > h_sim)
        
        return {
            'teams': (rh, ra),
            'xg': (xg_h, xg_a),
            'probs': (win_h, draw, win_a),
            'goals': (over25, btts),
            'dc': (win_h + draw, win_a + draw),
            'ah': (ah_h, ah_a),
            'form': (info[rh]['form'], info[ra]['form'])
        }

    def calculate_kelly_stake(self, prob, threshold):
        edge = prob - threshold
        if edge < 0: return "NO BET"
        if edge < 0.05: return "0.5% (Min)"
        if edge < 0.10: return "1.0% (Normal)"
        if edge < 0.15: return "1.5% (Fuerte)"
        return "2.0% (MAX)"

    def log_new_pick(self, date, league, home, away, pick, conf, us_odd):
        with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([date, league, home, away, pick, f"{conf:.2f}", us_odd, "PENDING", 0])

    def run_daily_scan(self):
        self.audit_history()

        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando análisis: {today}", flush=True)
        
        if not self.load_fixtures():
            self.send_msg(f"⚠️ Error descarga: {today}")
            return

        try: target = pd.to_datetime(today, dayfirst=True)
        except: return
        
        daily = self.fixtures[self.fixtures['Date'] == target]
        if daily.empty:
            self.send_msg(f"💤 <b>{today}:</b> Base de datos vacía.")
            return

        found_picks = 0
        header_sent = False
        rejected_log = []

        for idx, row in daily.iterrows():
            div = row['Div']
            if div in LEAGUE_CONFIG:
                res = self.analyze_match(row['HomeTeam'], row['AwayTeam'], div)
                if res:
                    ph, px, pa = res['probs']; po, pb = res['goals']; d1x, dx2 = res['dc']
                    ah_h, ah_a = res['ah']
                    threshold = LEAGUE_CONFIG[div]['threshold']
                    
                    pick = None; conf = 0.0; pick_type = ""
                    
                    if ph > threshold: pick = "GANA LOCAL"; conf = ph; pick_type="WIN"
                    elif pa > threshold: pick = "GANA VISITA"; conf = pa; pick_type="WIN"
                    elif d1x > 0.83: pick = "1X"; conf = d1x; pick_type="DC"
                    elif dx2 > 0.83: pick = "X2"; conf = dx2; pick_type="DC"
                    elif po > 0.63: pick = "OVER 2.5"; conf = po; pick_type="GOL"
                    elif pb > 0.63: pick = "BTTS (Ambos)"; conf = pb; pick_type="BTTS"
                    
                    if pick:
                        found_picks += 1
                        if not header_sent:
                            self.send_msg(f"🐺 <b>EURO-SNIPER v33</b>\n📅 {today} | 🧬 Titanium")
                            header_sent = True

                        fair_odd_dec = 1/conf
                        fair_odd_us = self.decimal_to_american(fair_odd_dec)
                        stake_reco = self.calculate_kelly_stake(conf, threshold)
                        self.log_new_pick(today, LEAGUE_CONFIG[div]['name'], res['teams'][0], res['teams'][1], pick, conf, fair_odd_us)

                        f_h = res['form'][0]; f_a = res['form'][1]
                        mom_h = "🔥" if f_h > 1.05 else ("🧊" if f_h < 0.95 else "➡️")
                        mom_a = "🔥" if f_a > 1.05 else ("🧊" if f_a < 0.95 else "➡️")
                        
                        edge = conf - threshold
                        stake_bar = "🟦⬜⬜ (Min)"
                        if edge > 0.10: stake_bar = "🟦🟦⬜ (Fuerte)"
                        if edge > 0.15: stake_bar = "🟦🟦🟦 (MAX)"

                        emoji_pick = "👉"
                        if pick_type == "WIN": emoji_pick = "💰"
                        if pick_type == "DC": emoji_pick = "🛡️"
                        if pick_type == "GOL": emoji_pick = "⚽"
                        if pick_type == "BTTS": emoji_pick = "🥊"

                        msg = (
                            f"🏆 <b>{LEAGUE_CONFIG[div]['name']}</b>\n"
                            f"<b>{res['teams'][0]}</b> {mom_h} vs {mom_a} <b>{res['teams'][1]}</b>\n"
                            f"───────────────\n"
                            f"📊 <b>X-RAY DATA:</b>\n"
                            f"• 1X2: {ph*100:.0f}% / {px*100:.0f}% / {pa*100:.0f}%\n"
                            f"• DC: 1X {d1x*100:.0f}% | X2 {dx2*100:.0f}%\n"
                            f"• GOALS: Ov {po*100:.0f}% | Un {(1-po)*100:.0f}%\n"
                            f"• HANDI: H-1.5 {ah_h*100:.0f}% | A+1.5 {(1-ah_h)*100:.0f}%\n"
                            f"───────────────\n"
                            f"🎯 <b>VEREDICTO:</b>\n"
                            f"{emoji_pick} <b>{pick}</b>\n"
                            f"⚖️ Fair Odd: <b>{fair_odd_us}</b>\n"
                            f"🏦 Stake: <b>{stake_bar}</b>"
                        )
                        self.send_msg(msg)
                        time.sleep(1.5)
                    else:
                        # LOGICA DE DESCARTES PRECISA (Debugging del Tottenham)
                        # Detectamos cuál fue el "mejor intento" y qué umbral falló
                        probs = [
                            (ph, threshold, "Gana"), 
                            (pa, threshold, "Gana"), 
                            (d1x, 0.83, "1X"), 
                            (dx2, 0.83, "X2"), 
                            (po, 0.63, "Over"), 
                            (pb, 0.63, "BTTS")
                        ]
                        # Ordenamos por probabilidad de mayor a menor
                        best_try = sorted(probs, key=lambda x: x[0], reverse=True)[0]
                        
                        prob_val = best_try[0]
                        req_val = best_try[1]
                        type_val = best_try[2]
                        
                        rejected_log.append(f"• {res['teams'][0]} vs {res['teams'][1]}: {type_val} {prob_val*100:.0f}% (Req {req_val*100:.0f}%)")

        if rejected_log:
            rej_msg = "\n".join(rejected_log[:10])
            self.send_msg(f"🗑️ <b>DESCARTES DEL DÍA:</b>\n{rej_msg}")
        
        if found_picks == 0 and not rejected_log:
            self.send_msg(f"⚠️ <b>{today}:</b> Sin partidos en la lista.")
        else:
            self.send_msg(f"🏁 <b>Fin del reporte.</b>")

if __name__ == "__main__":
    bot = TelegramSniper()
    print(f"🤖 BOT TITANIUM. Hora target: {RUN_TIME} UTC", flush=True)
    if os.getenv("SELF_TEST", "False") == "True": bot.run_daily_scan()
    schedule.every().day.at(RUN_TIME).do(bot.run_daily_scan)
    while True: schedule.run_pending(); time.sleep(60)
