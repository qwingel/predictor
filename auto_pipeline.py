"""
Автоматизированный пайплайн: парсинг hltv.org/matches → прогноз → Telegram.
"""
import sys
import os
import time
import json
from datetime import datetime
from typing import List, Dict

if sys.stdout.isatty() or hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
# HLTV парсер (локальный путь для разработки, на сервере не используется)
_HLTV_PARSE_PATH = r"D:\pycharm_projects\hltv_parse\cs2_data_collector"
if os.path.exists(_HLTV_PARSE_PATH):
    sys.path.insert(0, _HLTV_PARSE_PATH)
else:
    # На сервере — путь к команде парсинга (если скопирована)
    _SERVER_PARSE_PATH = os.path.join(PROJECT_DIR, "cs2_data_collector")
    if os.path.exists(_SERVER_PARSE_PATH):
        sys.path.insert(0, _SERVER_PARSE_PATH)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from predict import load_model, predict_for_all_maps
from telegram_publisher import post_to_telegram
from ml.utils import normalize_team_name, load_team_ratings
from ml.team_matcher import get_canonical_name
from parse_winline import get_winline_odds
from bet_recommendation import evaluate_bet

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RATINGS_PATH = os.path.join(PROJECT_DIR, "data", "top_teams.txt")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model_lgbm_final.pkl")
TELEGRAM_CHANNEL_PUBLIC = "-1003681541532"
TELEGRAM_VIP_CHANNEL = "-1003764539642"
PREDICTIONS_LOG = os.path.join(PROJECT_DIR, "data", "predictions_log.json")
ALL_MAPS = ["Inferno", "Mirage", "Dust2", "Ancient", "Anubis", "Nuke", "overpass"]
MAP_EMOJI = {
    "Inferno": "🔥",
    "Mirage": "🏜️",
    "Dust2": "💀",
    "Ancient": "🏛️",
    "Anubis": "🐺",
    "Nuke": "☢️",
    "overpass": "🌉",
}


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------
def _create_driver():
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    driver = uc.Chrome(options=options)
    return driver


def _wait_for_page(driver, timeout=30):
    try:
        for _ in range(3):
            source = driver.page_source.lower()
            if "checking your browser" in source or "cf-challenge" in source:
                time.sleep(10)
                driver.refresh()
                time.sleep(5)
            else:
                break
        WebDriverWait(driver, timeout).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )
        return True
    except TimeoutException:
        return False


def _is_top30(team_name: str) -> bool:
    normalized = normalize_team_name(team_name)
    ratings = load_team_ratings(RATINGS_PATH)
    if normalized in ratings:
        return True
    canonical = get_canonical_name(team_name)
    return canonical is not None


# ---------------------------------------------------------------------------
# 1. Scrape HLTV
# ---------------------------------------------------------------------------
def scrape_matches_today() -> List[Dict]:
    today_str = datetime.now().strftime("%A - %Y-%m-%d")
    date_iso = datetime.now().strftime("%Y-%m-%d")
    print(f"Parsing hltv.org/matches - today: {today_str}")

    driver = _create_driver()
    matches: List[Dict] = []

    try:
        driver.get("https://www.hltv.org/matches")
        if not _wait_for_page(driver):
            return []

        # Kill cookie banner
        driver.execute_script("""
            ['CybotCookiebotDialog','CybotCookiebotDialogBodyUnderlay','CybotCookiebotDialogBodyContentWrapper']
            .forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.remove();
            });
            document.querySelectorAll('[id*="Cybot"], [id*="Cookiebot"]').forEach(function(el) { el.remove(); });
        """)
        time.sleep(2)
        _wait_for_page(driver)
        time.sleep(2)

        # Switch to "Time" sorting
        try:
            driver.find_element(By.CSS_SELECTOR, ".matches-sort-by-toggle-time").click()
            time.sleep(2)
        except Exception:
            pass

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".matches-list-section"))
            )
        except TimeoutException:
            return []

        # Find today's section
        today_section = None
        sections = driver.find_elements(By.CSS_SELECTOR, ".matches-list-section")
        for section in sections:
            try:
                headline_el = section.find_element(By.CSS_SELECTOR, ".matches-list-headline")
                headline = headline_el.get_attribute("innerHTML") or ""
                if today_str in headline or date_iso in headline:
                    today_section = section
                    break
            except NoSuchElementException:
                continue

        if not today_section and sections:
            today_section = sections[0]

        if not today_section:
            return []

        containers = today_section.find_elements(By.CSS_SELECTOR, ".live-match-container")
        if not containers:
            containers = today_section.find_elements(By.CSS_SELECTOR, "[data-match-wrapper]")

        def get_text(el):
            return driver.execute_script("return arguments[0].textContent;", el).strip()

        for idx, container in enumerate(containers, 1):
            try:
                teamname_els = container.find_elements(By.CSS_SELECTOR, ".match-teamname")
                team1 = get_text(teamname_els[0]) if len(teamname_els) >= 1 else ""
                team2 = get_text(teamname_els[1]) if len(teamname_els) >= 2 else ""

                if not team1 or not team2:
                    continue
                if not _is_top30(team1) or not _is_top30(team2):
                    continue

                # Match time
                try:
                    time_el = container.find_element(By.CSS_SELECTOR, ".match-time")
                    match_time = get_text(time_el)
                except NoSuchElementException:
                    match_time = ""

                # Tournament
                try:
                    tournament = container.find_element(By.CSS_SELECTOR, ".match-event").text.strip()
                except NoSuchElementException:
                    tournament = "Unknown"

                # LAN
                is_lan = 1 if container.get_attribute("lan") == "true" else 0

                # Format (BO1 / BO3 / BO5)
                try:
                    format_text = container.find_element(By.CSS_SELECTOR, ".match-meta").text.strip()
                except NoSuchElementException:
                    format_text = ""

                # HLTV match link — <a> находится внутри .match-wrapper
                try:
                    link_el = container.find_element(By.CSS_SELECTOR, ".match a")
                    match_url = link_el.get_attribute("href")
                    if match_url and not match_url.startswith("http"):
                        match_url = "https://www.hltv.org" + match_url
                except NoSuchElementException:
                    match_url = ""

                print(f"  [{idx}] OK  {team1} vs {team2}  |  {match_time}  |  {tournament}  |  {format_text}")
                matches.append({
                    "team1_name": team1,
                    "team2_name": team2,
                    "match_time": match_time,
                    "tournament": tournament,
                    "is_lan": is_lan,
                    "format_text": format_text,
                    "match_url": match_url,
                })
            except Exception as e:
                print(f"  [{idx}] error: {e}")
                continue

        print(f"  => {len(matches)} top-30 matches found")
        return matches

    except Exception as e:
        import traceback
        traceback.print_exc()
        return []
    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 2. Predict
# ---------------------------------------------------------------------------
def _run_predictions(matches: List[Dict], model):
    predictions = {}
    for m in matches:
        t1 = m["team1_name"]
        t2 = m["team2_name"]
        is_lan = m["is_lan"]
        print(f"  {t1} vs {t2} (lan={is_lan})")

        try:
            result = predict_for_all_maps(model, t1, t2, is_lan)
        except Exception as e:
            print(f"  Prediction error: {e}")
            continue

        maps_data = {}
        for map_name in ALL_MAPS:
            if map_name not in result:
                continue
            md = result[map_name]
            if md[1] < 0:
                maps_data[map_name] = {"winner": t1, "prob_winner": md[2], "prob_loser": md[3]}
            else:
                maps_data[map_name] = {"winner": t2, "prob_winner": md[3], "prob_loser": md[2]}

        gen = result["general"]
        if gen[1] < 0:
            gen_data = {"winner": t1, "prob_winner": gen[2], "prob_loser": gen[3]}
        else:
            gen_data = {"winner": t2, "prob_winner": gen[3], "prob_loser": gen[2]}

        # Winline odds + bet recommendation for the predicted winner
        bet_recommendation = None
        odds = get_winline_odds(t1, t2)
        if odds:
            gen_winner = gen_data["winner"]
            model_prob = gen_data["prob_winner"]
            bookmaker_odds = odds["team1"] if gen_winner == t1 else odds["team2"]
            bet_recommendation = evaluate_bet(model_prob, bookmaker_odds)

        key = f"{t1} vs {t2}"
        predictions[key] = {
            "maps": maps_data,
            "general": gen_data,
            "odds": odds,
            "bet_recommendation": bet_recommendation,
        }

    return predictions

# ---------------------------------------------------------------------------
# 3. Generate report
# ---------------------------------------------------------------------------
def generate_report(matches, predictions) -> str:
    lines = [
        f"🎯 CS2 Predictions — {datetime.now().strftime('%d.%m.%Y')}",
        ""
    ]

    for m in matches:
        key = f"{m['team1_name']} vs {m['team2_name']}"
        if key not in predictions:
            continue

        pred = predictions[key]
        t1 = m["team1_name"]
        t2 = m["team2_name"]
        tournament = m.get("tournament", "")
        format_text = m.get("format_text", "")
        match_time = m.get("match_time", "")

        lines += [
            f"🏟️ {tournament}{' | ' + format_text if format_text else ''}{' ⏰ ' + match_time if match_time else ''}",
            f"⚔️ {t1} vs {t2}",
            "",
            "📊"
        ]

        for map_name in ALL_MAPS:
            md = pred["maps"].get(map_name)
            if md:
                emoji = MAP_EMOJI.get(map_name, "🗺️")
                prob_w = md["prob_winner"]
                if prob_w >= 0.65:
                    marker = "🟢"
                elif prob_w >= 0.55:
                    marker = "🟡"
                else:
                    marker = "🔴"
                lines.append(f"  {emoji} {map_name:<12} {marker} {md['winner']} — {prob_w:.0%}")

        gen = pred["general"]
        gen_prob = gen["prob_winner"]
        gen_winner = gen["winner"]

        lines += [
            "",
            f"🏆 Победитель: {gen_winner} ({gen_prob:.0%})",
        ]

        bet = pred.get("bet_recommendation")
        if bet and gen_winner == t1:
            odds = pred["odds"]
            bookmaker = odds["team1"]
        elif bet:
            odds = pred["odds"]
            bookmaker = odds["team2"]
        else:
            bookmaker = None

        if bet and bookmaker:
            ev = bet["ev"]
            ev_str = f'+{ev:.1%}' if ev >= 0 else f'{ev:.1%}'
            emoji_map = {"СТАВИТЬ": "🟢", "РИСК": "🟡"}
            bet_emoji = emoji_map.get(bet["recommendation"], "🔴")
            lines.append(f"💰 Бет-совет: {bet_emoji} {bet['recommendation']} | Модель {bet['model_prob']:.0%} | Winline {bookmaker} | EV {ev_str}")

        lines += [
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            ""
        ]

    lines.append("🤖 predictor bot v3.0")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3b. Generate simple report (public channel — only favorites) TODO: сделать 1 матч
# ---------------------------------------------------------------------------
def generate_simple_report(matches, predictions, match_index=0) -> str:
    """
    Формирует краткий отчёт для публичного канала ТОЛЬКО по одному матчу.
    :param matches: список всех матчей дня
    :param predictions: словарь прогнозов
    :param match_index: индекс матча в списке (0 = первый)
    """
    if not matches or match_index >= len(matches):
        return "⚠️ Нет матчей для отчёта."

    lines = [
        f"🎯 CS2 Прогнозы — {datetime.now().strftime('%d.%m.%Y')}",
        ""
    ]

    m = matches[match_index]
    key = f"{m['team1_name']} vs {m['team2_name']}"
    if key not in predictions:
        return f"⚠️ Прогноз для матча {key} не найден."

    pred = predictions[key]
    t1 = m["team1_name"]
    t2 = m["team2_name"]
    tournament = m.get("tournament", "")
    format_text = m.get("format_text", "")
    match_time = m.get("match_time", "")

    gen = pred["general"]
    gen_winner = gen["winner"]
    gen_prob = gen["prob_winner"]

    if gen_prob >= 0.65:
        marker = "🟢"
    elif gen_prob >= 0.55:
        marker = "🟡"
    else:
        marker = "🔴"

    lines += [
        f"🏟️ {tournament}{' | ' + format_text if format_text else ''}{' ⏰ ' + match_time if match_time else ''}",
        f"⚔️ {t1} vs {t2}",
        f"🏆 Фаворит: {gen_winner} {marker}",
        "━━━━━━━━━━━━━━━━━━━━━━━━",
        "Больше матчей и информации по ним @cyberpredictorbot",
        ""
    ]

    lines.append("🤖 predictor bot v3.0")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Save predictions to log
# ---------------------------------------------------------------------------
def save_predictions_log(matches: List[Dict], predictions: Dict):
    """Сохраняет прогнозы в JSON-файл для последующего сравнения с результатами."""
    os.makedirs(os.path.dirname(PREDICTIONS_LOG), exist_ok=True)

    date_key = datetime.now().strftime("%Y-%m-%d")

    # Load existing log
    if os.path.exists(PREDICTIONS_LOG):
        with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
            log = json.load(f)
    else:
        log = {}

    entries = []
    for m in matches:
        key = f"{m['team1_name']} vs {m['team2_name']}"
        if key not in predictions:
            continue
        pred = predictions[key]
        entries.append({
            "team1": m["team1_name"],
            "team2": m["team2_name"],
            "match_time": m.get("match_time", ""),
            "tournament": m.get("tournament", ""),
            "format": m.get("format_text", ""),
            "match_url": m.get("match_url", ""),
            "predicted_winner": pred["general"]["winner"],
            "predicted_prob": round(pred["general"]["prob_winner"], 4),
            "maps": {
                map_name: {
                    "winner": md["winner"],
                    "prob": round(md["prob_winner"], 4),
                }
                for map_name, md in pred["maps"].items()
            },
        })

    log[date_key] = entries
    with open(PREDICTIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"  Predictions saved to {PREDICTIONS_LOG} (date: {date_key})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 60)
    print("HLTV -> Predict -> Telegram")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    print("Step 1: Scrape hltv.org/matches")
    matches = scrape_matches_today()
    if not matches:
        print("No top-30 matches found. Exiting.")
        return

    print("\nStep 2: Predictions")
    print("  Loading model...")
    model = load_model(MODEL_PATH)
    predictions = _run_predictions(matches, model)
    if not predictions:
        print("  No predictions produced")
        return

    print("\nStep 3: Report")
    report = generate_report(matches, predictions)
    print("\n" + report)

    print("\nStep 4: Telegram")
    try:
        post_to_telegram(TELEGRAM_VIP_CHANNEL, report)
        print("  VIP channel posted successfully!")
    except Exception as e:
        print(f"  VIP Telegram error: {e}")

    try:
        simple_report = generate_simple_report(matches, predictions)
        post_to_telegram(TELEGRAM_CHANNEL_PUBLIC, simple_report)
        print("  Public channel posted successfully!")
    except Exception as e:
        print(f"  Public Telegram error: {e}")

    print("\nStep 5: Save predictions log")
    save_predictions_log(matches, predictions)


if __name__ == "__main__":
    run_pipeline()
