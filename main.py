import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
from datetime import datetime

# --- CONFIGURACIÓN DEL SERVIDOR ---
# En Railway, estas variables se configuran en la pestaña "Variables"
# Para probar en tu PC, pon tus datos aquí entre comillas.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "PON_TU_TOKEN_AQUI_SI_PRUEBAS_LOCAL")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "PON_TU_CHAT_ID_AQUI")

# HORA DE EJECUCIÓN (Hora del Servidor, suele ser UTC)
RUN_TIME = "03:00" 

# CONFIGURACIÓN SNIPER
SIMULATION_RUNS = 50000 
FORM_WEIGHT = 0.20      
SEASON = '2526'         

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Connection': 'keep-alive'
}

LEAGUE_ATLAS = {
    'E0': '🇬🇧 Premier', 'SP1': '🇪🇸 La Liga', 'I1': '🇮🇹 Serie A', 
    'D1': '🇩🇪 Bundes', 'F1': '🇫🇷 Ligue 1', 'P1': '🇵🇹 Primeira', 
    'N1': '🇳🇱 Eredivisie', 'B1': '🇧🇪 Pro League', 'T1': '🇹🇷 Süper Lig'
}

class TelegramSniper:
    def __init__(self):
        self.fixtures = None
        self.history_cache = {} 

    def send_msg(self, text):
        """Envía el mensaje a Telegram."""
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "HTML" # Usamos HTML para negritas
        }
        try:
            requests.post(url, json=payload)
        except Exception as e:
            print(f"Error enviando Telegram: {e}")

    def load_fixtures(self):
        print(f"[SYSTEM] Actualizando Base de Datos...")
        try:
            r = requests.get("https://www.football-data.co.uk/fixtures.csv", headers=HEADERS, timeout=15)
            r.raise_for_status()
            try: content = r.content.decode('utf-8-sig')
            except: content = r.content.decode('latin-1')
            
            self.fixtures = pd.read_csv(io.StringIO(content))
            self.fixtures.rename(columns={self.fixtures.columns[0]: 'Div'}, inplace=True)
            self.fixtures.columns = self.fixtures.columns.str.strip()
            self.fixtures = self.fixtures.dropna(subset=['Div'])
            self.fixtures['Date'] = pd.to_datetime(self.fixtures['Date'], dayfirst=True, errors='coerce')
            return True
        except Exception as e:
            print(f"Error cargando fixtures: {e}")
            return False

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        try:
            r = requests.get(url, headers=HEADERS)
            try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
            except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
            df = df.dropna(subset=['FTHG', 'FTAG'])
            
            # Form Analysis Simplificado
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
        except: return None

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
        
        return {
            'teams': f"{rh} vs {ra}",
            'xg': (xg_h, xg_a),
            'probs': (win_h, draw, win_a),
            'goals': over25,
            'dc': (win_h + draw, win_a + draw),
            'form': (info[rh]['form'], info[ra]['form'])
        }

    def run_daily_scan(self):
        # 1. Obtener fecha de hoy
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Iniciando escaneo para: {today}")
        
        if not self.load_fixtures():
            self.send_msg(f"⚠️ Error: No pude descargar el calendario para {today}")
            return

        # Convertir a datetime para filtrar
        try: target = pd.to_datetime(today, dayfirst=True)
        except: return
        
        daily = self.fixtures[self.fixtures['Date'] == target]
        
        if daily.empty:
            self.send_msg(f"💤 Hoy {today} no hay partidos programados en el calendario.")
            return

        found_picks = 0
        self.send_msg(f"🤖 <b>EURO-SNIPER REPORT</b>\n📅 Fecha: {today}\n🔍 Analizando mercado...")

        for idx, row in daily.iterrows():
            div = row['Div']
            if div in LEAGUE_ATLAS:
                res = self.analyze_match(row['HomeTeam'], row['AwayTeam'], div)
                if res:
                    ph, px, pa = res['probs']; po = res['goals']; d1x, dx2 = res['dc']
                    
                    # LOGICA DE SELECCIÓN (SOLO LO BUENO)
                    pick = None; conf = 0.0
                    
                    if ph > 0.60: pick = "GANA LOCAL"; conf = ph
                    elif pa > 0.60: pick = "GANA VISITA"; conf = pa
                    elif d1x > 0.83: pick = "1X (Local/Empate)"; conf = d1x
                    elif dx2 > 0.83: pick = "X2 (Visita/Empate)"; conf = dx2
                    elif po > 0.62: pick = "OVER 2.5 GOLES"; conf = po
                    
                    if pick:
                        found_picks += 1
                        fair_odd = 1/conf
                        
                        # Iconos Forma
                        ic_h = "🔥" if res['form'][0] > 1.05 else ("❄️" if res['form'][0] < 0.95 else "➖")
                        ic_a = "🔥" if res['form'][1] > 1.05 else ("❄️" if res['form'][1] < 0.95 else "➖")
                        icon_verdict = "💎" if conf > 0.65 else "✅"

                        # Construir Mensaje HTML
                        msg = (
                            f"<b>⚽ {res['teams'].upper()}</b>\n"
                            f"🏆 {LEAGUE_ATLAS[div]}\n"
                            f"📊 Forma: {ic_h} vs {ic_a}\n"
                            f"📈 xG: {res['xg'][0]:.2f} - {res['xg'][1]:.2f}\n"
                            f"------------------\n"
                            f"{icon_verdict} <b>{pick}</b> ({conf*100:.1f}%)\n"
                            f"🎯 Fair Odd: <b>@{fair_odd:.2f}</b>"
                        )
                        self.send_msg(msg)
                        time.sleep(1) # Pequeña pausa para no saturar Telegram

        if found_picks == 0:
            self.send_msg(f"⚠️ Hoy {today} no hay picks seguros (>60%) en las ligas Elite.")
        else:
            self.send_msg(f"🏁 <b>Fin del Reporte.</b> {found_picks} oportunidades detectadas.")

# --- BUCLE DE EJECUCIÓN (RAILWAY) ---
if __name__ == "__main__":
    bot = TelegramSniper()
    
    print(f"🤖 BOT INICIADO. Esperando a las {RUN_TIME} UTC para ejecutar...")
    self_test = os.getenv("SELF_TEST", "False")
    
    # Si quieres probarlo nada más subirlo, pon la variable SELF_TEST = True en Railway
    if self_test == "True":
        bot.run_daily_scan()

    # Programador
    schedule.every().day.at(RUN_TIME).do(bot.run_daily_scan)
    
    while True:
        schedule.run_pending()
        time.sleep(60) # Revisar cada minuto
