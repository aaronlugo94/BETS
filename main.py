import pandas as pd
import numpy as np
import requests
import io
import difflib
import time
import schedule
import os
from datetime import datetime

# --- CONFIGURACIÓN EURO-SNIPER v21.1 ---

# Credenciales (Configúralas en Railway -> Variables)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "PON_TU_TOKEN_AQUI_SI_ES_LOCAL")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "PON_TU_CHAT_ID_AQUI")

# HORA DE EJECUCIÓN (UTC)
# 03:00 UTC = 20:00 PM (8 PM) en Tucson (MST)
RUN_TIME = "03:28"

# AJUSTES DEL MODELO
SIMULATION_RUNS = 50000 
FORM_WEIGHT = 0.20      
SEASON = '2526'         

# HEADERS ROTATIVOS (Para evitar bloqueos de IP)
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Wget/1.20.3 (linux-gnu)'
]

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
        """Envía mensaje a Telegram con reintentos."""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            print("⚠️ Faltan credenciales de Telegram.", flush=True)
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
        
        for _ in range(3):
            try:
                r = requests.post(url, json=payload, timeout=10)
                if r.status_code == 200: break
            except Exception as e:
                print(f"Error enviando Telegram: {e}", flush=True)
                time.sleep(2)

    def load_fixtures(self):
        """
        Descarga datos con TIMEOUT ESTRICTO para evitar cuelgues de 7 minutos.
        """
        print(f"[SYSTEM] Iniciando descarga de calendario...", flush=True)
        
        # URLs alternativas (HTTP vs HTTPS a veces ayuda con bloqueos)
        urls = [
            "https://www.football-data.co.uk/fixtures.csv",
            "http://www.football-data.co.uk/fixtures.csv"
        ]
        
        for i, url in enumerate(urls):
            print(f"   ➡️ Intento {i+1}: Conectando a {url}...", flush=True)
            
            try:
                # TIMEOUT: (5 seg conectar, 15 seg leer). Si tarda más, corta.
                headers = {'User-Agent': USER_AGENTS[i % len(USER_AGENTS)]}
                r = requests.get(url, headers=headers, timeout=(5, 15))
                r.raise_for_status()
                
                print("   ✅ Conexión establecida. Procesando CSV...", flush=True)
                
                # Decodificación segura
                try: content = r.content.decode('utf-8-sig')
                except: content = r.content.decode('latin-1')
                
                self.fixtures = pd.read_csv(io.StringIO(content))
                
                # Limpieza de datos
                if not self.fixtures.empty:
                    self.fixtures.rename(columns={self.fixtures.columns[0]: 'Div'}, inplace=True)
                    self.fixtures.columns = self.fixtures.columns.str.strip()
                    self.fixtures = self.fixtures.dropna(subset=['Div'])
                    self.fixtures['Date'] = pd.to_datetime(self.fixtures['Date'], dayfirst=True, errors='coerce')
                    print(f"   ✅ Datos cargados: {len(self.fixtures)} partidos.", flush=True)
                    return True
                    
            except requests.exceptions.Timeout:
                print(f"   ❌ Timeout: El servidor tardó demasiado. Reintentando...", flush=True)
            except Exception as e:
                print(f"   ❌ Error en intento {i+1}: {e}", flush=True)
            
            time.sleep(3) # Espera antes de reintentar
        
        print("⛔ ERROR CRÍTICO: No se pudo descargar el calendario.", flush=True)
        return False

    def get_league_data(self, div):
        if div in self.history_cache: return self.history_cache[div]
        url = f"https://www.football-data.co.uk/mmz4281/{SEASON}/{div}.csv"
        
        # Reintentos para datos históricos
        for i in range(2):
            try:
                headers = {'User-Agent': USER_AGENTS[0]}
                r = requests.get(url, headers=headers, timeout=(5, 10))
                
                try: df = pd.read_csv(io.StringIO(r.content.decode('utf-8-sig')))
                except: df = pd.read_csv(io.StringIO(r.content.decode('latin-1')))
                df = df.dropna(subset=['FTHG', 'FTAG'])
                
                # Form Analysis
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
            except:
                time.sleep(1)
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
            'teams': f"{rh} vs {ra}",
            'xg': (xg_h, xg_a),
            'probs': (win_h, draw, win_a),
            'goals': (over25, btts),
            'dc': (win_h + draw, win_a + draw),
            'form': (info[rh]['form'], info[ra]['form'])
        }

    def run_daily_scan(self):
        # Fecha del SERVIDOR (Cuando son las 20:00 en Tucson, en el server ya es mañana o madrugada)
        today = datetime.now().strftime('%d/%m/%Y')
        print(f"🚀 Ejecutando escaneo para fecha: {today}", flush=True)
        
        if not self.load_fixtures():
            self.send_msg(f"⚠️ <b>Error Crítico:</b> El servidor de datos no responde para la fecha {today}.")
            return

        try: target = pd.to_datetime(today, dayfirst=True)
        except: return
        
        daily = self.fixtures[self.fixtures['Date'] == target]
        
        if daily.empty:
            self.send_msg(f"💤 <b>Reporte {today}:</b> No hay partidos programados hoy. El bot sigue activo.")
            print("No hay partidos. Aviso enviado.", flush=True)
            return

        found_picks = 0
        header_sent = False

        for idx, row in daily.iterrows():
            div = row['Div']
            if div in LEAGUE_ATLAS:
                res = self.analyze_match(row['HomeTeam'], row['AwayTeam'], div)
                if res:
                    ph, px, pa = res['probs']; po, pb = res['goals']; d1x, dx2 = res['dc']
                    
                    pick = None; conf = 0.0
                    
                    # LOGICA DE SELECCIÓN DE VALOR
                    if ph > 0.60: pick = "GANA LOCAL"; conf = ph
                    elif pa > 0.60: pick = "GANA VISITA"; conf = pa
                    elif d1x > 0.83: pick = "1X (Local/Empate)"; conf = d1x
                    elif dx2 > 0.83: pick = "X2 (Visita/Empate)"; conf = dx2
                    elif po > 0.62: pick = "OVER 2.5 GOLES"; conf = po
                    elif pb > 0.62: pick = "AMBOS MARCAN (BTTS)"; conf = pb
                    
                    if pick:
                        found_picks += 1
                        if not header_sent:
                            self.send_msg(f"🐺 <b>EURO-SNIPER NIGHT OPS</b>\n📅 Picks para: {today}\n🔎 Analizando mercado...")
                            header_sent = True

                        fair_odd = 1/conf
                        ic_h = "🔥" if res['form'][0] > 1.05 else ("❄️" if res['form'][0] < 0.95 else "➖")
                        ic_a = "🔥" if res['form'][1] > 1.05 else ("❄️" if res['form'][1] < 0.95 else "➖")
                        icon_verdict = "💎" if conf > 0.65 else "✅"

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
                        time.sleep(1) # Pausa anti-spam

        if found_picks == 0:
            self.send_msg(f"⚠️ <b>Reporte {today}:</b> Hay partidos, pero NINGUNO cumple los criterios de seguridad (>60%).")
            print("Escaneo finalizado sin picks.", flush=True)
        else:
            self.send_msg(f"🏁 <b>Fin del Reporte.</b> {found_picks} oportunidades detectadas.\n⚠️ <i>Nota: Revisa alineaciones.</i>")
            print(f"Escaneo finalizado con {found_picks} picks.", flush=True)

# --- BUCLE DE EJECUCIÓN (RAILWAY) ---
if __name__ == "__main__":
    bot = TelegramSniper()
    
    print(f"🤖 BOT NIGHT OPS INICIADO.", flush=True)
    print(f"🕒 Hora objetivo: {RUN_TIME} UTC (20:00 Tucson)", flush=True)
    
    # Auto-test inmediato al desplegar (Opcional, para que veas si funciona YA)
    # Si quieres que se ejecute una vez al subirlo, descomenta la siguiente línea:
    bot.run_daily_scan()

    schedule.every().day.at(RUN_TIME).do(bot.run_daily_scan)
    
    while True:
        schedule.run_pending()
        time.sleep(60)
