def calculate_ev(model_prob, bookmaker_odds):
    """
    Рассчитывает математическое ожидание ставки.

    Формула: EV = P(win) * (odds - 1) - P(lose) * 1
           = P(win) * odds - 1

    Args:
        model_prob: вероятность победы от модели (0.0 - 1.0)
        bookmaker_odds: коэффициент букмекера

    Returns:
        float: EV в долямах (например, 0.05 = 5% прибыли на единицу ставки)
    """
    # Правильная формула EV для ставок
    ev = model_prob * (bookmaker_odds - 1) - (1 - model_prob)
    # Упрощённо: ev = model_prob * bookmaker_odds - 1
    return model_prob * bookmaker_odds - 1


def calculate_kelly_size(model_prob, bookmaker_odds, kelly_fraction=0.25, max_bet=0.05, min_bet=0.01):
    """
    Рассчитывает оптимальный размер ставки по критерию Келли.

    Формула Келли: f* = (p * b - q) / b
    где:
        p = model_prob (вероятность выигрыша)
        q = 1 - p (вероятность проигрыша)
        b = bookmaker_odds - 1 (коэффициент выплаты)

    Args:
        model_prob: вероятность от модели (0.0 - 1.0)
        bookmaker_odds: коэффициент букмекера
        kelly_fraction: дробный Келли (0.25 = 25% от полного Келли)
        max_bet: максимальная ставка в процентах от банка (0.05 = 5%)
        min_bet: минимальная ставка в процентах от банка (0.01 = 1%)

    Returns:
        float: процент от банка для ставки (0.0 если ставка не рекомендуется)
    """
    # Проверка на положительное EV
    ev = calculate_ev(model_prob, bookmaker_odds)
    if ev <= 0:
        return 0.0

    # Полная формула Келли: f* = (p * (odds - 1) - q) / (odds - 1)
    # Упрощённо: f* = p - (1 - p) / (odds - 1)
    b = bookmaker_odds - 1
    q = 1 - model_prob

    if b <= 0:
        return 0.0

    full_kelly = (model_prob * b - q) / b

    # Если Келли отрицательный — не ставим
    if full_kelly <= 0:
        return 0.0

    # Дробный Келли
    bet_size = full_kelly * kelly_fraction

    # Ограничиваем минимальной и максимальной ставкой
    bet_size = max(min_bet, min(bet_size, max_bet))

    return bet_size


def calculate_confidence_interval(model_prob, uncertainty_base=0.05):
    """
    Рассчитывает доверительный интервал для вероятности.

    Args:
        model_prob: вероятность от модели (0.0 - 1.0)
        uncertainty_base: базовая неопределённость (0.05 = 5%)

    Returns:
        tuple: (prob_lower, prob_upper)
    """
    # Чем больше карт, тем меньше неопределённость
    sample_factor = 0.3
    uncertainty = uncertainty_base * (1 - sample_factor * 0.5)

    prob_lower = max(0.0, model_prob - uncertainty)
    prob_upper = min(1.0, model_prob + uncertainty)

    return prob_lower, prob_upper


def evaluate_bet(model_prob, bookmaker_odds, kelly_fraction=0.25, min_edge=0.03, max_bet=0.05):
    """
    Полная оценка ставки: стоит ли ставить и какой размер банка.

    Логика принятия решений:
    1. Если EV < 0 — математически убыточная ставка, НЕ СТАВИТЬ
    2. Если EV > 0, но edge < min_edge — преимущество слишком мало, риск
    3. Если EV > 0 и edge >= min_edge — валуйная ставка, рассчитываем размер по Келли

    Args:
        model_prob: вероятность от модели (0.0 - 1.0)
        bookmaker_odds: коэффициент букмекера
        kelly_fraction: дробный Келли (0.25 = 25% от полного Келли, более консервативно)
        min_edge: минимальное преимущество над букмекером (0.03 = 3%)
        max_bet: максимальная ставка в процентах от банка (0.05 = 5%)

    Returns:
        dict: решение и все расчёты
    """
    # 1. Подразумеваемая вероятность букмекера
    bookmaker_implied = 1 / bookmaker_odds

    # 2. Расчёт EV (математическое ожидание)
    ev = calculate_ev(model_prob, bookmaker_odds)

    # 3. Преимущество над букмекером (edge)
    edge = model_prob - bookmaker_implied

    # 4. "Справедливый" коэффициент модели
    fair_odds = 1 / model_prob if model_prob > 0 else float('inf')

    # 5. Доверительный интервал для оценки риска
    prob_lower, prob_upper = calculate_confidence_interval(model_prob)
    ev_lower = calculate_ev(prob_lower, bookmaker_odds)
    ev_upper = calculate_ev(prob_upper, bookmaker_odds)

    # 6. Размер ставки по Келли
    bet_size = calculate_kelly_size(model_prob, bookmaker_odds, kelly_fraction, max_bet)

    # 7. Формируем рекомендацию
    if ev <= 0:
        recommendation = "НЕ СТАВИТЬ"
        reason = "Отрицательное математическое ожидание (EV <= 0)"
        confidence = "Низкая"
    elif edge < min_edge:
        recommendation = "НЕ СТАВИТЬ"
        reason = f"Преимущество {edge*100:.2f}% ниже порога {min_edge*100:.1f}%"
        confidence = "Низкая"
    elif ev_lower < 0:
        recommendation = "РИСК"
        reason = f"EV положительно, но нижняя граница интервала отрицательная (EV={ev*100:.2f}%, EV_lower={ev_lower*100:.2f}%)"
        confidence = "Средняя"
        bet_size = bet_size * 0.5  # Уменьшаем ставку на 50% из-за неопределённости
    else:
        recommendation = "СТАВИТЬ"
        reason = f"Валуйная ставка: EV={ev*100:.2f}%, edge={edge*100:.2f}%"
        confidence = "Высокая"

    # 8. Оценка "силы" ставки (0-100)
    # Чем больше EV и edge, тем выше сила
    bet_strength = min(100, max(0, (ev * 100 + edge * 100) * 5))

    return {
        'recommendation': recommendation,
        'reason': reason,
        'confidence': confidence,
        'bet_strength': bet_strength,
        'model_prob': model_prob,
        'bookmaker_implied_prob': bookmaker_implied,
        'bookmaker_odds': bookmaker_odds,
        'fair_odds': fair_odds,
        'edge': edge,
        'ev': ev,
        'ev_lower_bound': ev_lower,
        'bet_size_percent': bet_size,
        'confidence_interval': (prob_lower, prob_upper)
    }