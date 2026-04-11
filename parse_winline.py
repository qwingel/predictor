"""
Парсинг коэффициентов Winline на конкретный матч Counter-Strike.
"""
import sys
import io
import time
import difflib
from typing import Dict, Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait

CHROME_BINARY = r"C:\Program Files\Google\Chrome Beta\Application\chrome.exe"


def _create_driver():
    options = uc.ChromeOptions()
    options.binary_location = CHROME_BINARY
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    return uc.Chrome(options=options)


def get_winline_odds(team1: str, team2: str) -> Optional[Dict[str, float]]:
    """Находит коэффициенты Winline на матч двух команд.

    Args:
        team1: имя первой команды (как в HLTV / top-30).
        team2: имя второй команды.

    Returns:
        {"team1": coeff1, "team2": coeff2} или None если матч не найден.
    """
    driver = _create_driver()
    driver.set_window_size(1920, 1080)

    t1_norm = team1.strip().upper()
    t2_norm = team2.strip().upper()

    try:
        driver.get("https://winline.ru/stavki/sport/kibersport/counter-strike")

        for _ in range(3):
            src = driver.page_source.lower()
            if "checking your browser" in src or "cf-challenge" in src:
                time.sleep(15)
                driver.refresh()
                time.sleep(5)
            else:
                break

        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        for _ in range(8):
            time.sleep(3)
            body_text = driver.execute_script("return document.body.textContent || '';")
            if len(body_text) > 100_000:
                break

        # Собираем все матчи с коэффициентами
        matches = driver.execute_script("""
            var results = [];
            var cards = document.querySelectorAll('.card.ng-star-inserted');

            for (var i = 0; i < cards.length; i++) {
                var card = cards[i];
                var cardText = card.textContent || '';

                var nameDivs = card.querySelectorAll('div.name.ng-star-inserted');
                var names = [];
                for (var j = 0; j < nameDivs.length && names.length < 2; j++) {
                    var t = nameDivs[j].textContent.trim();
                    if (t) names.push(t);
                }
                if (names.length < 2) continue;

                if (!cardText.includes('Матч')) continue;

                var markets = card.getElementsByTagName('ww-feature-event-market-dsk');
                if (!markets || markets.length === 0) continue;

                var market = markets[0];
                var spans = market.querySelectorAll('span');
                var coefs = [];
                for (var k = 0; k < spans.length && coefs.length < 2; k++) {
                    var v = parseFloat(spans[k].textContent.trim().replace(',', '.'));
                    if (!isNaN(v) && v > 1.01 && v < 15.0) coefs.push(v);
                }
                if (coefs.length < 2) continue;

                results.push({
                    team1: names[0],
                    team2: names[1],
                    odds1: coefs[0],
                    odds2: coefs[1]
                });
            }
            return results;
        """)

        def match_name(hltv_name: str, winline_name: str) -> bool:
            """Точное совпадение, содержит или fuzzy >= 0.8."""
            h = hltv_name.upper().strip()
            w = winline_name.upper().strip()
            if h == w:
                return True
            # Одно имя целиком содержится в другом (e.g. "FUT" vs "FUT ESPORTS")
            if h in w or w in h:
                return True
            ratio = difflib.SequenceMatcher(None, h, w).ratio()
            return ratio >= 0.8

        for m in matches:
            m1 = m["team1"].upper().strip()
            m2 = m["team2"].upper().strip()

            # Проверяем оба направления (HLTV team1 может быть на любой позиции)
            if (match_name(t1_norm, m1) and match_name(t2_norm, m2)):
                print(f"  Found: {m['team1']} ({m['odds1']}) vs {m['team2']} ({m['odds2']})")
                return {"team1": float(m["odds1"]), "team2": float(m["odds2"])}
            elif (match_name(t1_norm, m2) and match_name(t2_norm, m1)):
                print(f"  Found (reversed): {m['team1']} ({m['odds1']}) vs {m['team2']} ({m['odds2']})")
                return {"team1": float(m["odds2"]), "team2": float(m["odds1"])}

        print(f"  Match {team1} vs {team2} not found on Winline")
        return None

    except Exception as e:
        print(f"  Error: {e}")
        return None
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def scrape_all_odds() -> list[dict]:
    """Scrape all Winline CS:2 odds in one pass.

    Returns:
        List of dicts: [{"team1": name1, "team2": name2, "odds1": float, "odds2": float}, ...]
    """
    driver = _create_driver()
    driver.set_window_size(1920, 1080)

    try:
        driver.get("https://winline.ru/stavki/sport/kibersport/counter-strike")

        # Cloudflare bypass
        for _ in range(3):
            src = driver.page_source.lower()
            if "checking your browser" in src or "cf-challenge" in src:
                time.sleep(15)
                driver.refresh()
                time.sleep(5)
            else:
                break

        WebDriverWait(driver, 30).until(
            lambda d: d.execute_script("return document.readyState") == "complete"
        )

        for _ in range(8):
            time.sleep(3)
            body_text = driver.execute_script("return document.body.textContent || '';")
            if len(body_text) > 100_000:
                break

        matches = driver.execute_script("""
            var results = [];
            var cards = document.querySelectorAll('.card.ng-star-inserted');

            for (var i = 0; i < cards.length; i++) {
                var card = cards[i];
                var cardText = card.textContent || '';

                var nameDivs = card.querySelectorAll('div.name.ng-star-inserted');
                var names = [];
                for (var j = 0; j < nameDivs.length && names.length < 2; j++) {
                    var t = nameDivs[j].textContent.trim();
                    if (t) names.push(t);
                }
                if (names.length < 2) continue;

                if (!cardText.includes('Матч')) continue;

                var markets = card.getElementsByTagName('ww-feature-event-market-dsk');
                if (!markets || markets.length === 0) continue;

                var market = markets[0];
                var spans = market.querySelectorAll('span');
                var coefs = [];
                for (var k = 0; k < spans.length && coefs.length < 2; k++) {
                    var v = parseFloat(spans[k].textContent.trim().replace(',', '.'));
                    if (!isNaN(v) && v > 1.01 && v < 15.0) coefs.push(v);
                }
                if (coefs.length < 2) continue;

                results.push({
                    team1: names[0],
                    team2: names[1],
                    odds1: coefs[0],
                    odds2: coefs[1]
                });
            }
            return results;
        """)

        return [{"team1": m["team1"], "team2": m["team2"], "odds1": float(m["odds1"]), "odds2": float(m["odds2"])} for m in matches]

    except Exception as e:
        print(f"  Error scraping Winline: {e}")
        return []
    finally:
        try:
            driver.quit()
        except Exception:
            pass


def find_odds(team1: str, team2: str, all_odds: list[dict]) -> Optional[Dict[str, float]]:
    """Find odds for a specific match within pre-scraped odds list.

    Args:
        team1: HLTV team name.
        team2: HLTV team name.
        all_odds: Result from scrape_all_odds().

    Returns:
        {"team1": coeff1, "team2": coeff2} or None.
    """

    def match_name(hltv_name: str, winline_name: str) -> bool:
        h = hltv_name.upper().strip()
        w = winline_name.upper().strip()
        if h == w:
            return True
        if h in w or w in h:
            return True
        ratio = difflib.SequenceMatcher(None, h, w).ratio()
        return ratio >= 0.8

    t1_norm = team1.strip().upper()
    t2_norm = team2.strip().upper()

    for m in all_odds:
        m1 = m["team1"].upper().strip()
        m2 = m["team2"].upper().strip()
        if match_name(t1_norm, m1) and match_name(t2_norm, m2):
            return {"team1": m["odds1"], "team2": m["odds2"]}
        elif match_name(t1_norm, m2) and match_name(t2_norm, m1):
            return {"team1": m["odds2"], "team2": m["odds1"]}

    return None


if __name__ == "__main__":
    odds = get_winline_odds("PARIVISION", "FUT")
    if odds:
        print(f"  PARIVISION: {odds['team1']}")
        print(f"  FUT: {odds['team2']}")
    else:
        print("  Not found")
