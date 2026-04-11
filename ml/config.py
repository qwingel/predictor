from datetime import datetime, timedelta

# База данных
DB_PATH = "data/cs2_data.db"

# Период сбора данных (последние 12 месяцев)
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=365)

START_DATE_STR = "2025-03-31"
END_DATE_STR = "2026-03-31"

# Настройки парсинга
REQUEST_DELAY = 3          # секунды между запросами к страницам матчей
PAGE_LOAD_TIMEOUT = 30     # таймаут загрузки страницы
MAX_RETRIES = 3            # количество попыток при ошибке

# Топ-30 команд HLTV (по состоянию на март 2025)
# Можно обновить позже через парсинг
TOP_30_TEAMS = [
    "Spirit", "G2", "Vitality", "FaZe", "MOUZ", "NAVI", "NRG",
    "The MongolZ", "Liquid", "FURIA", "Falcons", "paiN", "3DMAX",
    "Aurora", "FUT", "Astralis", "HEROIC", "Legacy", "Monte",
    "Ninjas in Pyjamas", "SINNERS", "9z", "BetBoom", "HOTU", "Gentle Mates", "TYLOO",
    "B8", "PARIVISION", "Passion UA", "GamerLegion"
]

# Словарь нормализации названий (HLTV -> каноническое имя)
TEAM_ALIASES = {
    "Spirit": ["Spirit", "Team Spirit", "spirit", "ts"],
    "G2": ["G2", "G2 Esports", "g2"],
    "Vitality": ["Vitality", "Team Vitality", "vitality"],
    "FaZe": ["FaZe", "FaZe Clan", "faze"],
    "MOUZ": ["MOUZ", "Mouz", "Mousesports", "mouz", "mouse"],
    "NAVI": ["NAVI", "Natus Vincere", "NaVi", "navi", "natus vincere"],
    "NRG": ["NRG", "nrg", "NRG"],
    "The MongolZ": ["The MongolZ", "MongolZ", "mongolz"],
    "Liquid": ["Liquid", "Team Liquid", "liquid"],
    "FURIA": ["FURIA", "Furia", "furia", "furias"],
    "Falcons": ["Falcons", "FALCONS", "falcons"],
    "paiN": ["paiN", "paiN Gaming", "pain", "pain gaming"],
    "3DMAX": ["3DMAX", "3dmax"],
    "Aurora": ["AURORA", "aurora", "Aurora"],
    "FUT": ["fut", "Fut"],
    "Astralis": ["Astralis", "astralis"],
    "HEROIC": ["HEROIC", "Heroic", "heroic"],
    "Legacy": ["legacy", "LEGACY"],
    "Monte": ["monte", "MONTE", "monte"],
    "Ninjas in Pyjamas": ["NIP", "nip"],
    "SINNERS": ["sinners", "Sinners", "SINNERS"],
    "9z": ["9z", "9z Team"],
    "BetBoom": ["BetBoom", "betboom", "bb"],
    "HOTU": ["hotu", "HOTU"],
    "Gentle Mates": ["gentle mates", "GM"],
    "TYLOO": ["tyloo", "TYLOO"],
    "B8": ["B8", "b8"],
    "PARIVISION": ["PARIVISION", "parivision", "pari"],
    "Passion UA": ["Passion UA", "passion ua", "passion"],
    "GamerLegion": ["GamerLegion", "gamerlegion", "gl"]
}
