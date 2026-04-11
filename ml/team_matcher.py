"""
Модуль для сопоставления названий команд с каноническими именами топ-30.
"""
import difflib
import re
from ml.config import TEAM_ALIASES, TOP_30_TEAMS


# Расширенный список алиасов для популярных команд
EXTENDED_ALIASES = {
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


def normalize_name(name):
    """Нормализует название команды"""
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9\s]', '', name)
    name = re.sub(r'\s+', ' ', name)
    return name


def get_canonical_name(name):
    """Возвращает каноническое имя команды из топ-30"""
    if not name:
        return None

    name_lower = normalize_name(name)

    # Проверяем расширенные алиасы
    for canonical, aliases in EXTENDED_ALIASES.items():
        for alias in aliases:
            if normalize_name(alias) == name_lower:
                return canonical

    # Проверяем стандартные алиасы из конфига
    for canonical, aliases in TEAM_ALIASES.items():
        for alias in aliases:
            if normalize_name(alias) == name_lower:
                return canonical

    # Fuzzy matching для топ команд
    best_match = None
    best_ratio = 0

    for team in TOP_30_TEAMS:
        ratio = difflib.SequenceMatcher(None, name_lower, normalize_name(team)).ratio()
        if ratio > best_ratio and ratio > 0.75:  # Высокий порог для точности
            best_ratio = ratio
            best_match = team

    return best_match


def is_top_team(team_name):
    """Проверяет, входит ли команда в топ-30"""
    canonical = get_canonical_name(team_name)
    return canonical is not None


def match_team_names(name1, name2):
    """Сравнивает два названия команд"""
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)

    if norm1 == norm2:
        return True

    # Проверка через алиасы
    all_aliases = {**TEAM_ALIASES, **EXTENDED_ALIASES}
    for canonical, aliases in all_aliases.items():
        canon_norm = normalize_name(canonical)
        if norm1 == canon_norm or norm2 == canon_norm:
            return True
        for alias in aliases:
            alias_norm = normalize_name(alias)
            if norm1 == alias_norm or norm2 == alias_norm:
                return True

    return False
