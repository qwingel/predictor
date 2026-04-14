"""
Скрипт для сравнения прогнозов модели с реальными результатами вчерашних матчей.
Читает predictions_log.json → парсит HLTV результаты → постит отчёт в Telegram.
"""
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

if sys.stdout.isatty() or hasattr(sys.stdout, 'encoding') and sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, r"D:\pycharm_projects\hltv_parse\cs2_data_collector")

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from telegram_publisher import post_to_telegram

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PREDICTIONS_LOG = os.path.join(PROJECT_DIR, "data", "predictions_log.json")
ACCURACY_TRACKER = os.path.join(PROJECT_DIR, "data", "accuracy_tracker.json")
TELEGRAM_CHANNEL = "-1003764539642"
TELEGRAM_CHANNEL_PUBLIC = "-1003681541532"
CHROME_BINARY = None  # undetected-chromedriver найдёт сам
MAP_EMOJI = {
    "Inferno": "🔥",
    "Mirage": "🏜️",
    "Dust2": "💀",
    "Ancient": "🏛️",
    "Anubis": "🐺",
    "Nuke": "☢️",
    "overpass": "🌉",
}
ALL_MAPS = ["Inferno", "Mirage", "Dust2", "Ancient", "Anubis", "Nuke", "overpass"]


def _create_driver():
    options = uc.ChromeOptions()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    kwargs = {"options": options}
    if CHROME_BINARY:
        kwargs["browser_executable_path"] = CHROME_BINARY
    driver = uc.Chrome(**kwargs)
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


# ---------------------------------------------------------------------------
# Scrape match results from HLTV
# ---------------------------------------------------------------------------
def scrape_match_result(match_url: str) -> Optional[Dict]:
    """Парсит страницу результата матча и возвращает скоры.

    Returns:
        {
            "score_team1": int,    # общий счёт, e.g. 2
            "score_team2": int,    # e.g. 0
            "maps": [
                {"name": "Inferno", "score1": 13, "score2": 10},
                ...
            ]
        } or None
    """
    driver = _create_driver()
    try:
        driver.get(match_url)
        if not _wait_for_page(driver):
            return None

        # Kill cookie banner
        driver.execute_script("""
            ['CybotCookiebotDialog','CybotCookiebotDialogBodyUnderlay','CybotCookiebotDialogBodyContentWrapper']
            .forEach(function(id) {
                var el = document.getElementById(id);
                if (el) el.remove();
            });
        """)
        time.sleep(5)

        # Explicit wait for mapholder elements
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".mapholder"))
            )
        except TimeoutException:
            print("  No .mapholder elements found after wait")
            pass

        def get_text(el):
            return driver.execute_script("return arguments[0].textContent;", el).strip()

        # Overall match score — from map results (count wins)
        # We'll compute this after parsing maps, since mapholder elements contain the actual data
        overall_t1 = None
        overall_t2 = None

        # Individual map results — using selectors from parse_match_details.py
        map_results = []
        try:
            map_holders = driver.find_elements(By.CSS_SELECTOR, ".mapholder")
            print(f"  Found {len(map_holders)} mapholder elements")
            for map_holder in map_holders:
                try:
                    # Map name
                    try:
                        map_name_el = map_holder.find_element(By.CSS_SELECTOR, ".mapname")
                    except Exception:
                        try:
                            map_name_el = map_holder.find_element(By.CSS_SELECTOR, ".map-name-holder .mapname")
                        except Exception:
                            continue
                    map_name = get_text(map_name_el)

                    # Scores — try .results-left / .results-right
                    s1, s2 = None, None
                    try:
                        left_block = map_holder.find_element(By.CSS_SELECTOR, ".results-left")
                        score_text = get_text(left_block.find_element(By.CSS_SELECTOR, ".results-team-score"))
                        if score_text != "-":
                            s1 = int(score_text)
                    except Exception:
                        pass
                    try:
                        right_block = map_holder.find_element(By.CSS_SELECTOR, ".results-right")
                        score_text = get_text(right_block.find_element(By.CSS_SELECTOR, ".results-team-score"))
                        if score_text != "-":
                            s2 = int(score_text)
                    except Exception:
                        pass

                    # Fallback: direct .results-team-score
                    if s1 is None or s2 is None:
                        scores = map_holder.find_elements(By.CSS_SELECTOR, ".results-team-score")
                        if len(scores) >= 2:
                            t1 = get_text(scores[0])
                            t2 = get_text(scores[1])
                            if t1 != "-":
                                s1 = int(t1)
                            if t2 != "-":
                                s2 = int(t2)

                    if s1 is not None and s2 is not None:
                        print(f"  Map: {map_name} — {s1}:{s2}")
                        map_results.append({"name": map_name, "score1": s1, "score2": s2})
                except Exception as e:
                    print(f"  Map parse error: {e}")
                    continue
        except Exception as e:
            print(f"  No mapholder elements found: {e}")
            pass

        if overall_t1 is None and not map_results:
            return None

        # Always compute overall score from map wins
        if map_results:
            wins1 = sum(1 for m in map_results if m["score1"] > m["score2"])
            wins2 = sum(1 for m in map_results if m["score2"] > m["score1"])
            overall_t1 = wins1
            overall_t2 = wins2

        return {
            "score_team1": overall_t1,
            "score_team2": overall_t2,
            "maps": map_results,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  Error scraping {match_url}: {e}")
        return None
    finally:
        try:
            driver.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Load yesterday's predictions
# ---------------------------------------------------------------------------
def load_yesterday_predictions() -> List[Dict]:
    """Читает predictions_log.json и возвращает прогнозы за вчера."""
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if not os.path.exists(PREDICTIONS_LOG):
        print(f"  No predictions log found at {PREDICTIONS_LOG}")
        return []

    with open(PREDICTIONS_LOG, "r", encoding="utf-8") as f:
        log = json.load(f)

    entries = log.get(yesterday_str, [])
    if not entries:
        # Also try looking for any recent date keys
        available_dates = sorted(log.keys(), reverse=True)
        print(f"  No predictions for {yesterday_str}. Available dates: {available_dates[:5]}")
        if available_dates:
            use_date = available_dates[0]
            print(f"  Using latest date: {use_date}")
            entries = log[use_date]

    print(f"  Loaded {len(entries)} predictions")
    return entries


# ---------------------------------------------------------------------------
# Generate report
# ---------------------------------------------------------------------------
def generate_results_report(results: List[Dict]) -> tuple[str, int, int, int, int]:
    """Формирует отчёт сравнения прогнозов и результатов."""
    date_str = (datetime.now() - timedelta(days=1)).strftime('%d.%m.%Y')

    # Заголовок с рамкой
    lines = [
        "╔════════════════════════════════════════╗",
        f"║     📊 ОТЧЁТ ПРОГНОЗОВ за {date_str}     ║",
        "╚════════════════════════════════════════╝",
        ""
    ]

    total = 0
    correct_general = 0
    correct_maps = 0
    total_maps = 0

    # Статистика по картам для группировки
    maps_stats = []

    for idx, r in enumerate(results):
        t1 = r["team1"]
        t2 = r["team2"]
        pred_winner = r["predicted_winner"]
        pred_prob = r["predicted_prob"]

        # Шапка матча
        lines += [
            f"🎯 МАТЧ #{idx + 1}",
            f"┌───────────── {t1}  VS  {t2} ─────────────┐",
            f"│ Прогноз: {pred_winner} (вероятность {pred_prob:.0%})",
        ]

        # Общий результат
        if r.get("actual_score") is not None:
            actual_t1, actual_t2 = r["actual_score"]
            actual_winner = t1 if actual_t1 > actual_t2 else (t2 if actual_t2 > actual_t1 else "Ничья")
            total += 1

            if actual_winner == pred_winner:
                correct_general += 1
                marker = "✅"
                status = "ВЕРНО"
            else:
                marker = "❌"
                status = "НЕВЕРНО"

            # Цветовые индикаторы (работает в Telegram)
            result_line = f"│ {marker} Результат: {status} | {actual_winner} ({actual_t1}:{actual_t2})"
            if actual_winner == pred_winner:
                result_line = f"│ 🟢 Результат: {status} | {actual_winner} ({actual_t1}:{actual_t2})"
            else:
                result_line = f"│ 🔴 Результат: {status} | {actual_winner} ({actual_t1}:{actual_t2})"

            lines.append(result_line)
        else:
            lines.append("│ ⏳ Результат ещё не получен")
            lines.append("└────────────────────────────────────────┘")
            lines.append("")
            continue

        # Карты
        if r.get("actual_maps"):
            lines.append("│")
            lines.append("│ 🗺️  ДЕТАЛИ ПО КАРТАМ:")

            map_correct = 0
            map_total = 0

            for am in r["actual_maps"]:
                map_name = am["name"]
                emoji = MAP_EMOJI.get(map_name, "🎮")

                # Актуальный результат карты
                actual_map_winner = t1 if am["score1"] > am["score2"] else (t2 if am["score2"] > am["score1"] else "?")
                actual_score_str = f"{am['score1']}:{am['score2']}"

                # Прогноз карты
                pred_map = r.get("maps", {}).get(map_name)
                if pred_map:
                    pred_map_winner = pred_map["winner"]
                    pred_map_prob = pred_map["winner_prob"]
                    is_correct = pred_map_winner == actual_map_winner

                    if is_correct:
                        correct_maps += 1
                        map_correct += 1
                        map_marker = "✅"
                        color_marker = "🟢"
                    else:
                        map_marker = "❌"
                        color_marker = "🔴"

                    map_total += 1
                    total_maps += 1

                    lines.append(
                        f"│   {emoji} {map_name}: {color_marker} {map_marker} {pred_map_winner} ({pred_map_prob:.0%}) → {actual_map_winner} ({actual_score_str})")
                else:
                    lines.append(f"│   {emoji} {map_name}: ⚪️ нет прогноза → {actual_map_winner} ({actual_score_str})")

            # Статистика по картам матча
            if map_total > 0:
                map_acc = map_correct / map_total
                bar = "█" * int(map_acc * 10) + "░" * (10 - int(map_acc * 10))
                lines.append(f"│   └─ 📊 Точность по картам: {map_correct}/{map_total} ({map_acc:.0%}) {bar}")

            maps_stats.append((map_correct, map_total))

        lines.append("└────────────────────────────────────────┘")
        lines.append("")

    # Общая статистика с прогресс-барами
    lines.append("╔════════════════════════════════════════╗")
    lines.append("║          📈 ОБЩАЯ СТАТИСТИКА           ║")
    lines.append("╚════════════════════════════════════════╝")
    lines.append("")

    if total > 0:
        acc = correct_general / total
        bar = "█" * int(acc * 10) + "░" * (10 - int(acc * 10))
        lines.append(f"🏆 ОБЩИЙ РЕЗУЛЬТАТ:")
        lines.append(f"   {correct_general}/{total} ({acc:.0%}) {bar}")
        lines.append("")

    if total_maps > 0:
        map_acc = correct_maps / total_maps
        bar = "█" * int(map_acc * 10) + "░" * (10 - int(map_acc * 10))
        lines.append(f"🗺️  ПРОГНОЗЫ ПО КАРТАМ:")
        lines.append(f"   {correct_maps}/{total_maps} ({map_acc:.0%}) {bar}")
        lines.append("")

    # Дополнительная статистика
    if maps_stats:
        total_match_maps = sum(count for _, count in maps_stats)
        avg_match_acc = sum(correct / count if count > 0 else 0 for correct, count in maps_stats) / len(maps_stats)
        lines.append(f"📊 СРЕДНЯЯ ТОЧНОСТЬ ПО МАТЧАМ:")
        lines.append(f"   {avg_match_acc:.0%} (в среднем за матч)")
        lines.append("")

    lines.append("")
    lines.append("🤖 *predictor bot v3.0*")
    lines.append("_Данные обновлены автоматически_")

    return "\n".join(lines), total, correct_general, total_maps, correct_maps


# ---------------------------------------------------------------------------
# Simple report (public channel — only match results, no probabilities)
# ---------------------------------------------------------------------------
def generate_simple_results_report(results: List[Dict]) -> str:
    lines = [
        f"📊 Результаты — {(datetime.now() - timedelta(days=1)).strftime('%d.%m.%Y')}",
        ""
    ]

    total = 0
    correct = 0

    for r in results:
        t1 = r["team1"]
        t2 = r["team2"]

        if r.get("actual_score") is None:
            lines += [
                f"⚔️ {t1} vs {t2}",
                f"  ⏳ Результат ещё не получен",
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━",
                ""
            ]
            continue

        actual_t1, actual_t2 = r["actual_score"]
        actual_winner = t1 if actual_t1 > actual_t2 else (t2 if actual_t2 > actual_t1 else "Ничья")
        pred_winner = r["predicted_winner"]
        total += 1

        if actual_winner == pred_winner:
            correct += 1
            marker = "✅"
        else:
            marker = "❌"

        lines += [
            f"⚔️ {t1} vs {t2}",
            f"  {marker} {actual_winner} ({actual_t1}:{actual_t2})",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            ""
        ]

    lines.append("📈 ИТОГО: ")
    if total > 0:
        acc = correct / total
        lines.append(f"  {correct}/{total} ({acc:.0%})")

    lines.append("")
    lines.append("🤖 predictor bot v3.0")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Accuracy tracker
# ---------------------------------------------------------------------------
def load_accuracy() -> Dict:
    if os.path.exists(ACCURACY_TRACKER):
        with open(ACCURACY_TRACKER, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total_matches": 0, "correct_general": 0, "total_maps": 0, "correct_maps": 0}


def save_accuracy(data: Dict):
    os.makedirs(os.path.dirname(ACCURACY_TRACKER), exist_ok=True)
    with open(ACCURACY_TRACKER, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def update_accuracy(total: int, correct_general: int, total_maps: int, correct_maps: int):
    acc = load_accuracy()
    acc["total_matches"] += total
    acc["correct_general"] += correct_general
    acc["total_maps"] += total_maps
    acc["correct_maps"] += correct_maps
    save_accuracy(acc)

    gen_acc = acc["correct_general"] / acc["total_matches"] if acc["total_matches"] > 0 else 0
    map_acc = acc["correct_maps"] / acc["total_maps"] if acc["total_maps"] > 0 else 0
    print(f"  Accuracy updated: matches {acc['correct_general']}/{acc['total_matches']} ({gen_acc:.0%}), "
          f"maps {acc['correct_maps']}/{acc['total_maps']} ({map_acc:.0%})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_results_report():
    print("=" * 60)
    print("Results Report — comparing predictions vs actual results")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()

    # Load predictions
    print("Step 1: Load yesterday's predictions")
    entries = load_yesterday_predictions()
    if not entries:
        print("  No predictions found. Exiting.")
        return

    # Scrape results
    print("\nStep 2: Scrape actual results from HLTV")
    results = []
    for entry in entries:
        match_url = entry.get("match_url", "")
        if not match_url:
            print(f"  Skipping {entry['team1']} vs {entry['team2']} — no HLTV URL")
            results.append(entry)
            continue

        print(f"  Scraping: {entry['team1']} vs {entry['team2']} — {match_url}")
        match_result = scrape_match_result(match_url)
        time.sleep(3)

        enriched = dict(entry)
        if match_result:
            enriched["actual_score"] = (match_result["score_team1"], match_result["score_team2"])
            enriched["actual_maps"] = match_result["maps"]
        else:
            enriched["actual_score"] = None
            enriched["actual_maps"] = []

        results.append(enriched)

    # Generate report
    print("\nStep 3: Generate report")
    report, total, correct_general, total_maps, correct_maps = generate_results_report(results)
    print("\n" + report)

    # Update accuracy tracker
    print("\nStep 4: Update accuracy")
    update_accuracy(total, correct_general, total_maps, correct_maps)

    # Post to Telegram
    print("\nStep 5: Telegram")
    try:
        post_to_telegram(TELEGRAM_CHANNEL, report)
        print("  VIP channel posted successfully!")
    except Exception as e:
        print(f"  VIP Telegram error: {e}")

    try:
        simple_report = generate_simple_results_report(results)
        post_to_telegram(TELEGRAM_CHANNEL_PUBLIC, simple_report)
        print("  Public channel posted successfully!")
    except Exception as e:
        print(f"  Public Telegram error: {e}")


if __name__ == "__main__":
    run_results_report()
