import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
from datetime import datetime

# --- CONFIGURACIÓN EURO-SNIPER v25.0 (PRO VISUALS) ---

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# 03:00 UTC = 20:00 PM Tucson (Para recibirlo en la noche)
RUN_TIME = "04:15"

SIMULATION_RUNS = 50000 
FORM_WEIGHT = 0.20      
SEASON = '2526'         

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Wget/1.20.3 (linux-gnu)'
]

LEAGUE_CONFIG = {
    'E0':  {'name': '🇬🇧 PREMIER LEAGUE', 'tier': 1, 'threshold': 0.60},
    'SP1': {'name': '🇪🇸 LA LIGA',       'tier': 1, 'threshold': 0.60},
    'I1':  {'name': '🇮🇹 SERIE A',       'tier': 1, 'threshold': 0.60},
    'D1':  {'name': '🇩🇪 BUNDESLIGA',    'tier': 1, 'threshold': 0.60},
    'F1':  {'name': '🇫🇷 LIGUE 1',       'tier': 1, 'threshold': 0.60},
    'P1':  {'name': '🇵🇹 LIGA PORTUGAL', 'tier': 2, 'threshold': 0.66},
    'N1':  {'name': '🇳🇱 EREDIVISIE',    'tier': 2, 'threshold': 0.66},
    'B1':  {'name': '🇧🇪 PRO LEAGUE',    'tier': 2, 'threshold': 0.66},
    'T1':  {'name': '🇹🇷 SUPER LIG',     'tier': 2, 'threshold': 0.66}
}

class TelegramSniper:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 
        self._check_creds()

    def _check_creds(self):
        print("--- SYSTEM CHECK ---", flush=True)
        if not TELEGRAM_TOKEN: print("❌ ERROR: Falta TELEGRAM_TOKEN", flush=True)
        if not TELEGRAM_CHAT_ID: print("❌ ERROR: Falta TELEGRAM_CHAT_ID", flush=True)

    def send_msg(self, text):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID: return
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        
        for i in range(3):
            try:
                r = requests.post(url, json=payload, timeout=10)
                if r.status_code == 200: return
                if r.status_code in [400, 401, 403]: 
                    print(f"❌ Error Telegram: {r.status_code} - {r.text}", flush=True)
                    return
            except: time.sleep(2)

    def load_fixtures(self):
        print(f"[SYSTEM] Conectando a base de datos...", flush=True)
        urls = ["https://www.football-data.co.uk/fixtures.csv", "http://www.football-data.co.uk/fixtures.csv"]
        
        for i, url in enumerate(urls):
            try:
                headers = {'User-Agent': USER_AGENTS[i % len(USER_AGENTS)]}
                r = requests.get(url, headers=headers, timeout=(5, 10))
                r.raise_for_status()
                try: content = r.content.decode('utf-8-sig')
                except: content = r.content.decode('latin-1')
                self.fixtures = pd.read_csv(io.StringIO(content))
                if not self.fixtures.empty:
                    self.fixtures.rename(columns={self.fixtures.columns[0]: 'Div'}, inplace=True)
                    self.fixtures.columns = self.fixtures.columns.str.strip()
                    self.fixtures = self.fixtures.dropna(subset=['Div'])
                    self.fixtures['Date'] = pd.to_datetime(self.fixtures['Date'], dayfirst=True, errors='coerce')
                    print(f"✅ Calendario actualizado: {len(self.fixtures)} partidos.", flush=True)
                    return True
            except: time.sleep(2)
        print("⛔ ERROR: Fallo en descarga de datos.", flush=True)
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
                    matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
                    points = 0; max_pts = len(matches) * 3
                    if len(matches) > 0:
                        for _, row in matches.iterrows():
                            if row['HomeTeam'] == team: pts = 3 if row['FTHG'] > row['FTAG'] else (1 if row['FTHG'] == row['FTAG'] else 0)
                            else: pts = 3 if row['FTAG'] > row['FTHG'] else (1 if row['FTAG'] == row['FTHG'] else 0)
                            points += pts
                        form_factor = 1.0 + ((points / max_pts - 0.5) * 2 * FORM_WEIGHT)
                    else: form_factor = 1.0
                    team_stats[team] = {'form': form_factor}

                avg_h = df['FTHG'].mean(); avg_a = df['FTAG'].mean()
                h_stats = df.groupby('HomeTeam')[['FTHG', 'FTAG']].mean()
                a_stats = df.groupby('AwayTeam')[['FTAG', 'FTHG']].mean()
                stats = pd.concat([h_stats, a_stats], axis=1)
                stats.columns = ['HS','HC','AS','AC']
                stats['Att_H'] = stats['HS'] / avg_h; stats['Def_H'] = stats['HC'] / avg_a
                stats['Att_A'] = stats['AS'] / avg_a; stats['Def_A'] = stats['AC'] / avg_h
                
                data_pack = {'stats': stats.fillna(1.0), 'avgs': {'h': avg_h, 'a': avg_a}, 'teams': stats.index.tolist(), 'details': team_stats}
                self.history_cache[div] = data_pack
                return data_pack
            except: time.sleep(1)
        return None

    def find_team(self, team, team_list):
        matches = difflib.get_close_matches(team, team_list, n=1, cutoff=0.5)
        return matches[0] if matches else None

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
        over25 = np.mean((h_sim + a_sim) > 2.5)
        btts = np.mean((h_sim > 0) & (a_sim > 0))
        
        return {
            'teams': (rh, ra),
            'xg': (xg_h, xg_a),
            'probs': (win_h, draw, win_a),
            'goals': (over25, btts),
            'dc': (win_h + draw, win_a + draw),
            'form': (info[rh]['form'], info[ra]['form'])
        }

    def run_daily_scan(self):
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando análisis: {today}", flush=True)
        
        if not self.load_fixtures():
            self.send_msg(f"⚠️ Error descarga: {today}")
            return

        try: target = pd.to_datetime(today, dayfirst=True)
        except: return
        
        daily = self.fixtures[self.fixtures['Date'] == target]
        if daily.empty:
            self.send_msg(f"💤 <b>{today}:</b> Sin partidos Elite programados.")
            return

        found_picks = 0
        header_sent = False

        for idx, row in daily.iterrows():
            div = row['Div']
            if div in LEAGUE_CONFIG:
                res = self.analyze_match(row['HomeTeam'], row['AwayTeam'], div)
                if res:
                    ph, px, pa = res['probs']; po, pb = res['goals']; d1x, dx2 = res['dc']
                    
                    threshold = LEAGUE_CONFIG[div]['threshold']
                    pick = None; conf = 0.0
                    pick_type = ""
                    
                    # LOGICA X-RAY ESTRICTA
                    if ph > threshold: pick = "GANA LOCAL (1)"; conf = ph; pick_type = "WIN"
                    elif pa > threshold: pick = "GANA VISITA (2)"; conf = pa; pick_type = "WIN"
                    elif d1x > 0.83: pick = "LOCAL O EMPATE (1X)"; conf = d1x; pick_type = "SAFE"
                    elif dx2 > 0.83: pick = "VISITA O EMPATE (X2)"; conf = dx2; pick_type = "SAFE"
                    elif po > 0.63: pick = "OVER 2.5 GOLES"; conf = po; pick_type = "GOALS"
                    
                    if pick:
                        found_picks += 1
                        if not header_sent:
                            self.send_msg(f"🐺 <b>EURO-SNIPER PRO</b>\n📅 Reporte: {today}")
                            header_sent = True

                        fair_odd = 1/conf
                        
                        # Stake Visual
                        gap = conf - threshold
                        stake_bar = "🟦⬜⬜ (1/3)"
                        if gap > 0.10: stake_bar = "🟦🟦⬜ (2/3)"
                        if gap > 0.20: stake_bar = "🟦🟦🟦 (MAX)"

                        # Forma Visual
                        f_h = res['form'][0]; f_a = res['form'][1]
                        txt_h = "Excelente" if f_h > 1.1 else ("Bien" if f_h > 1.0 else ("Regular" if f_h > 0.9 else "Pobre"))
                        txt_a = "Excelente" if f_a > 1.1 else ("Bien" if f_a > 1.0 else ("Regular" if f_a > 0.9 else "Pobre"))
                        
                        # Emojis PICK
                        emoji_pick = "👉"
                        if pick_type == "WIN": emoji_pick = "🔥"
                        if pick_type == "SAFE": emoji_pick = "🛡️"
                        if pick_type == "GOALS": emoji_pick = "⚽"

                        # MENSAJE PREMIUM
                        msg = (
                            f"🏆 <b>{LEAGUE_CONFIG[div]['name']}</b>\n"
                            f"<b>{res['teams'][0].upper()}</b> vs <b>{res['teams'][1].upper()}</b>\n"
                            f"───────────────\n"
                            f"⚡ <i>Momentum:</i> {txt_h} vs {txt_a}\n"
                            f"📉 <i>xG Sim:</i> {res['xg'][0]:.2f} - {res['xg'][1]:.2f}\n"
                            f"───────────────\n"
                            f"📊 <b>DATOS DE MERCADO:</b>\n"
                            f"• 1X2: {ph*100:.0f}% / {px*100:.0f}% / {pa*100:.0f}%\n"
                            f"• DO: 1X {d1x*100:.0f}% | X2 {dx2*100:.0f}%\n"
                            f"• Gol: Ov {po*100:.0f}% | Un {(1-po)*100:.0f}%\n"
                            f"───────────────\n"
                            f"🎯 <b>VEREDICTO ALGORÍTMICO:</b>\n"
                            f"{emoji_pick} <b>{pick}</b>\n\n"
                            f"⚖️ Fair Odd: <b>@{fair_odd:.2f}</b>\n"
                            f"💰 Stake: <b>{stake_bar}</b>"
                        )
                        self.send_msg(msg)
                        time.sleep(1.5)

        if found_picks == 0:
            self.send_msg(f"⚠️ <b>{today}:</b> Filtro estricto activado. Ningún partido ofrece valor suficiente hoy.")
        else:
            self.send_msg(f"🏁 <b>Fin del reporte.</b>")

if __name__ == "__main__":
    bot = TelegramSniper()
    print(f"🤖 BOT PRO VISUALS. Hora target: {RUN_TIME} UTC", flush=True)
    
    if os.getenv("SELF_TEST", "False") == "True":
        bot.run_daily_scan()

    schedule.every().day.at(RUN_TIME).do(bot.run_daily_scan)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
