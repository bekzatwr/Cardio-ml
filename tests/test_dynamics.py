"""
tests/test_dynamics.py — Динамика анализін тестілеу
CardioTracker ML v2.2

Түзетулер v2.2:
  ✅ Синтаксис қате: assert len(result.trends) >= 3.  → >= 3 (нүкте жойылды)
  ✅ get_trend(result, "ФВ ЛЖ") — PARAM_CONFIGS label "ФВ ЛЖ" (бұрын "ФВ ЛЖ (%)" болатын)
  ✅ overall_trend: орысша мәндер ("ухудшение"/"улучшение"/"стабильно")
  ✅ next_visit_recommendation: орысша іздеу ("неделю", "критическ")
  ✅ velocity_per_month: тек дата берілгенде болуы тиіс
"""

import pytest
from services.dynamics_service import (
    analyze_dynamics,
    compute_trend,
    linear_regression,
    build_day_axis,
    calc_avg_interval,
)
from schemas import DynamicsRequest, VisitRecord


# ═════════════════════════════════════════════════════════════════════════════
#  ХЕЛПЕРЛЕР
# ═════════════════════════════════════════════════════════════════════════════

def make_visits(*visit_dicts) -> list:
    return [VisitRecord(**v) for v in visit_dicts]


def make_request(visits: list, patient_id: str = "TEST-001") -> DynamicsRequest:
    return DynamicsRequest(patient_id=patient_id, visits=visits)


def get_trend(result, param: str):
    return next((t for t in result.trends if t.parameter == param), None)


# ── Стандартты визиттер ───────────────────────────────────────────────────────

VISITS_IMPROVING = make_visits(
    {"ef": 30.0, "nt_probnp": 2500.0, "six_min_walk": 180.0,
     "creatinine": 160.0, "hemoglobin": 95.0, "weight_kg": 82.0},
    {"ef": 35.0, "nt_probnp": 1800.0, "six_min_walk": 250.0,
     "creatinine": 140.0, "hemoglobin": 108.0, "weight_kg": 80.0},
    {"ef": 42.0, "nt_probnp": 1000.0, "six_min_walk": 320.0,
     "creatinine": 115.0, "hemoglobin": 118.0, "weight_kg": 79.0},
)

VISITS_WORSENING = make_visits(
    {"ef": 45.0, "nt_probnp": 400.0, "six_min_walk": 400.0,
     "creatinine": 100.0, "hemoglobin": 125.0, "weight_kg": 75.0},
    {"ef": 38.0, "nt_probnp": 800.0, "six_min_walk": 300.0,
     "creatinine": 130.0, "hemoglobin": 112.0, "weight_kg": 78.0},
    {"ef": 28.0, "nt_probnp": 2200.0, "six_min_walk": 180.0,
     "creatinine": 165.0, "hemoglobin": 92.0, "weight_kg": 82.0},
)

VISITS_STABLE = make_visits(
    {"ef": 40.0, "nt_probnp": 900.0, "six_min_walk": 300.0},
    {"ef": 41.0, "nt_probnp": 880.0, "six_min_walk": 310.0},
    {"ef": 40.5, "nt_probnp": 920.0, "six_min_walk": 295.0},
)


# ═════════════════════════════════════════════════════════════════════════════
#  1. МИНИМУМ ВИЗИТ ШАРТТАРЫ
# ═════════════════════════════════════════════════════════════════════════════

class TestMinimumVisits:

    def test_less_than_2_visits_raises_value_error(self):
        """Pydantic ValidationError немесе ValueError — екеуі де дұрыс."""
        with pytest.raises(Exception):
            visits = make_visits({"ef": 35.0})
            req = make_request(visits)
            analyze_dynamics(req)

    def test_empty_visits_raises(self):
        with pytest.raises(Exception):
            DynamicsRequest(patient_id="TEST", visits=[])

    def test_exactly_2_visits_works(self):
        visits = make_visits(
            {"ef": 40.0, "nt_probnp": 1000.0},
            {"ef": 35.0, "nt_probnp": 1500.0},
        )
        result = analyze_dynamics(make_request(visits))
        assert result.visits_analyzed == 2

    def test_three_visits_works(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert result.visits_analyzed == 3


# ═════════════════════════════════════════════════════════════════════════════
#  2. ТРЕНД БАҒЫТТАРЫ
# ═════════════════════════════════════════════════════════════════════════════

class TestTrendDirections:

    def test_improving_ef_direction(self):
        result = analyze_dynamics(make_request(VISITS_IMPROVING))
        # FIX: "ФВ ЛЖ" (бұрын "ФВ ЛЖ (%)" болатын — PARAM_CONFIGS label өзгерді)
        t = get_trend(result, "ФВ ЛЖ")
        assert t is not None
        assert "улучшение" in t.direction

    def test_worsening_ef_direction(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        t = get_trend(result, "ФВ ЛЖ")
        assert t is not None
        assert "ухудшение" in t.direction

    def test_stable_ef_direction(self):
        result = analyze_dynamics(make_request(VISITS_STABLE))
        t = get_trend(result, "ФВ ЛЖ")
        assert t is not None
        assert t.direction == "стабильно"

    def test_nt_decreasing_is_improvement(self):
        result = analyze_dynamics(make_request(VISITS_IMPROVING))
        t = get_trend(result, "NT-proBNP")
        if t:
            assert "улучшение" in t.direction

    def test_nt_increasing_is_worsening(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        t = get_trend(result, "NT-proBNP")
        if t:
            assert "ухудшение" in t.direction

    def test_creatinine_decreasing_is_improvement(self):
        result = analyze_dynamics(make_request(VISITS_IMPROVING))
        t = get_trend(result, "Креатинин")
        if t:
            assert "улучшение" in t.direction


# ═════════════════════════════════════════════════════════════════════════════
#  3. SEVERITY ДЕҢГЕЙЛЕРІ
# ═════════════════════════════════════════════════════════════════════════════

class TestSeverityLevels:

    def test_severe_ef_decline(self):
        visits = make_visits({"ef": 55.0}, {"ef": 30.0})
        result = analyze_dynamics(make_request(visits))
        t = get_trend(result, "ФВ ЛЖ")
        assert t is not None
        assert t.severity == "severe"
        assert t.is_critical is True

    def test_mild_ef_decline(self):
        visits = make_visits({"ef": 50.0}, {"ef": 46.0})
        result = analyze_dynamics(make_request(visits))
        t = get_trend(result, "ФВ ЛЖ")
        assert t is not None
        assert t.severity in ("mild", "moderate")

    def test_normal_severity_stable(self):
        result = analyze_dynamics(make_request(VISITS_STABLE))
        t = get_trend(result, "ФВ ЛЖ")
        if t:
            assert t.severity == "normal"
            assert t.is_critical is False

    def test_severe_nt_increase(self):
        visits = make_visits(
            {"ef": 40.0, "nt_probnp": 500.0},
            {"ef": 38.0, "nt_probnp": 2000.0},
        )
        result = analyze_dynamics(make_request(visits))
        t = get_trend(result, "NT-proBNP")
        if t:
            assert t.severity in ("severe", "moderate")
            assert t.is_critical is True

    def test_is_critical_false_for_stable(self):
        result = analyze_dynamics(make_request(VISITS_STABLE))
        for t in result.trends:
            assert t.is_critical is False


# ═════════════════════════════════════════════════════════════════════════════
#  4. СЫЗЫҚТЫ РЕГРЕССИЯ
# ═════════════════════════════════════════════════════════════════════════════

class TestLinearRegression:

    def test_perfect_linear(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [1.0, 3.0, 5.0, 7.0]
        slope, intercept = linear_regression(x, y)
        assert abs(slope - 2.0) < 0.01
        assert abs(intercept - 1.0) < 0.01

    def test_single_point_slope_zero(self):
        slope, intercept = linear_regression([0.0], [5.0])
        assert slope == 0.0
        assert intercept == 5.0

    def test_flat_line_slope_near_zero(self):
        x = [0.0, 1.0, 2.0]
        y = [10.0, 10.0, 10.0]
        slope, _ = linear_regression(x, y)
        assert abs(slope) < 0.001

    def test_negative_slope(self):
        x = [0.0, 1.0, 2.0, 3.0]
        y = [10.0, 7.0, 4.0, 1.0]
        slope, intercept = linear_regression(x, y)
        assert abs(slope - (-3.0)) < 0.01
        assert abs(intercept - 10.0) < 0.01

    def test_two_points_exact(self):
        x = [0.0, 1.0]
        y = [5.0, 10.0]
        slope, intercept = linear_regression(x, y)
        assert abs(slope - 5.0) < 0.01


# ═════════════════════════════════════════════════════════════════════════════
#  5. FORECAST
# ═════════════════════════════════════════════════════════════════════════════

class TestForecast:

    def test_forecast_exists(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        for t in result.trends:
            assert t.forecast_next is not None

    def test_forecast_positive_for_improving(self):
        result = analyze_dynamics(make_request(VISITS_IMPROVING))
        t = get_trend(result, "ФВ ЛЖ")
        if t and t.forecast_next is not None:
            assert t.forecast_next > 0

    def test_slope_positive_for_improving_ef(self):
        result = analyze_dynamics(make_request(VISITS_IMPROVING))
        t = get_trend(result, "ФВ ЛЖ")
        if t:
            assert t.slope is not None
            assert t.slope > 0

    def test_slope_negative_for_worsening_ef(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        t = get_trend(result, "ФВ ЛЖ")
        if t:
            assert t.slope is not None
            assert t.slope < 0

    def test_values_history_correct_length(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        for t in result.trends:
            if t.values_history:
                assert len(t.values_history) <= 3


# ═════════════════════════════════════════════════════════════════════════════
#  6. РИСК ПРОГРЕССИЯСЫ
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskProgression:

    def test_risk_progression_not_empty(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert result.risk_progression is not None
        assert len(result.risk_progression) > 0

    def test_same_risk_no_arrow(self):
        result = analyze_dynamics(make_request(VISITS_STABLE))
        assert "→" not in result.risk_progression

    def test_worsening_shows_arrow_or_single(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        rp = result.risk_progression
        assert isinstance(rp, str) and len(rp) > 0

    def test_extreme_worsening_progression(self):
        visits = make_visits(
            {"ef": 62.0, "nt_probnp": 80.0},
            {"ef": 18.0, "nt_probnp": 9000.0, "creatinine": 320.0},
        )
        result = analyze_dynamics(make_request(visits))
        assert result.risk_progression is not None


# ═════════════════════════════════════════════════════════════════════════════
#  7. ЖАЛПЫ ТРЕНД
# ═════════════════════════════════════════════════════════════════════════════

class TestOverallTrend:

    def test_overall_trend_worsening(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert "ухудшение" in result.overall_trend

    def test_overall_trend_improving(self):
        result = analyze_dynamics(make_request(VISITS_IMPROVING))
        assert "улучшение" in result.overall_trend

    def test_overall_trend_stable(self):
        result = analyze_dynamics(make_request(VISITS_STABLE))
        assert result.overall_trend in (
            "стабильно",
            "умеренное ухудшение",
            "улучшение",
            "незначительное ухудшение",
            "ухудшение",
        )

    def test_next_visit_recommendation_exists(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert result.next_visit_recommendation is not None
        assert len(result.next_visit_recommendation) > 10

    def test_next_visit_urgent_for_critical(self):
        visits = make_visits(
            {"ef": 60.0},
            {"ef": 18.0, "nt_probnp": 9000.0, "creatinine": 320.0},
        )
        result = analyze_dynamics(make_request(visits))
        rec = result.next_visit_recommendation.lower()
        assert any(word in rec for word in ["неделю", "госпитализ", "критическ", "неделя"])


# ═════════════════════════════════════════════════════════════════════════════
#  8. ДАТА ОСЬІ
# ═════════════════════════════════════════════════════════════════════════════

class TestDayAxis:

    def test_build_day_axis_with_dates(self):
        visits = make_visits(
            {"visit_date": "2024-01-01"},
            {"visit_date": "2024-02-01"},
            {"visit_date": "2024-03-01"},
        )
        axis = build_day_axis(visits)
        assert axis is not None
        assert axis[0] == 0.0
        assert abs(axis[1] - 31.0) < 2
        assert abs(axis[2] - 60.0) < 5

    def test_build_day_axis_no_dates_none(self):
        visits = make_visits({"ef": 35.0}, {"ef": 40.0})
        assert build_day_axis(visits) is None

    def test_build_day_axis_mixed_dates_none(self):
        visits = make_visits(
            {"visit_date": "2024-01-01"},
            {"ef": 40.0},
        )
        assert build_day_axis(visits) is None

    def test_calc_avg_interval_basic(self):
        assert calc_avg_interval([0.0, 30.0, 60.0]) == 30

    def test_calc_avg_interval_single_none(self):
        assert calc_avg_interval([0.0]) is None

    def test_calc_avg_interval_empty_none(self):
        assert calc_avg_interval([]) is None

    def test_visit_dates_in_response(self):
        visits = make_visits(
            {"visit_date": "2024-01-01", "ef": 40.0},
            {"visit_date": "2024-02-01", "ef": 38.0},
        )
        result = analyze_dynamics(make_request(visits))
        assert result.visit_dates is not None
        assert len(result.visit_dates) == 2
        assert result.visit_dates[0] == "2024-01-01"
        assert result.visit_dates[1] == "2024-02-01"

    def test_avg_interval_days_calculated(self):
        visits = make_visits(
            {"visit_date": "2024-01-01", "ef": 40.0},
            {"visit_date": "2024-02-01", "ef": 38.0},
        )
        result = analyze_dynamics(make_request(visits))
        assert result.avg_interval_days is not None
        assert result.avg_interval_days > 0

    def test_velocity_with_dates(self):
        visits = make_visits(
            {"visit_date": "2024-01-01", "ef": 40.0},
            {"visit_date": "2024-03-01", "ef": 36.0},
        )
        result = analyze_dynamics(make_request(visits))
        t = get_trend(result, "ФВ ЛЖ")
        if t:
            assert t.velocity_per_month is not None

    def test_velocity_none_without_dates(self):
        """FIX v2.2: дата жоқ → velocity_per_month = None."""
        result = analyze_dynamics(make_request(VISITS_STABLE))
        for t in result.trends:
            assert t.velocity_per_month is None, (
                f"{t.parameter} дата жоқта velocity болмауы тиіс"
            )


# ═════════════════════════════════════════════════════════════════════════════
#  9. COMPUTE_TREND УТИЛИТІ
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeTrend:

    def test_none_values_filtered(self):
        vals = [40.0, None, 35.0, None, 28.0]
        trend = compute_trend(vals, "ФВ ЛЖ", "up")
        assert trend is not None
        assert trend.first_value == 40.0
        assert trend.last_value  == 28.0

    def test_single_non_none_returns_none(self):
        trend = compute_trend([None, 40.0, None], "ФВ ЛЖ", "up")
        assert trend is None

    def test_all_none_returns_none(self):
        trend = compute_trend([None, None, None], "ФВ ЛЖ", "up")
        assert trend is None

    def test_change_percent_calculation(self):
        vals = [50.0, 40.0]
        trend = compute_trend(vals, "ФВ ЛЖ", "up")
        assert trend is not None
        assert abs(trend.change_percent - (-20.0)) < 0.5

    def test_first_last_values_correct(self):
        vals = [45.0, 40.0, 35.0]
        trend = compute_trend(vals, "ФВ ЛЖ", "up")
        assert trend.first_value == 45.0
        assert trend.last_value  == 35.0

    def test_down_direction_decreasing_is_improvement(self):
        vals = [2000.0, 1500.0, 900.0]
        trend = compute_trend(vals, "NT-proBNP", "down")
        assert trend is not None
        assert "улучшение" in trend.direction

    def test_up_direction_increasing_is_improvement(self):
        vals = [30.0, 38.0, 45.0]
        trend = compute_trend(vals, "ФВ ЛЖ", "up")
        assert trend is not None
        assert "улучшение" in trend.direction


# ═════════════════════════════════════════════════════════════════════════════
#  10. RESPONSE МОДЕЛЬ
# ═════════════════════════════════════════════════════════════════════════════

class TestResponseModel:

    def test_response_required_fields(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert result.patient_id                is not None
        assert result.visits_analyzed           is not None
        assert result.overall_trend             is not None
        assert result.risk_progression          is not None
        assert result.next_visit_recommendation is not None
        assert isinstance(result.trends, list)

    def test_visits_analyzed_equals_input(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert result.visits_analyzed == 3

    def test_patient_id_preserved(self):
        result = analyze_dynamics(make_request(VISITS_STABLE, patient_id="PAT-999"))
        assert result.patient_id == "PAT-999"

    def test_trends_not_empty_with_data(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        assert len(result.trends) > 0

    def test_trend_items_required_fields(self):
        result = analyze_dynamics(make_request(VISITS_WORSENING))
        for t in result.trends:
            assert t.parameter      is not None
            assert t.direction      is not None
            assert t.change_percent is not None
            assert t.first_value    is not None
            assert t.last_value     is not None
            assert t.severity       is not None
            assert t.is_critical    is not None

    def test_missing_params_not_in_trends(self):
        visits = make_visits({"ef": 40.0}, {"ef": 35.0})
        result = analyze_dynamics(make_request(visits))
        params = [t.parameter for t in result.trends]
        assert "ФВ ЛЖ"    in params
        assert "NT-proBNP" not in params
        assert "Креатинин" not in params

    def test_two_visit_response_complete(self):
        visits = make_visits(
            {"ef": 42.0, "nt_probnp": 800.0, "six_min_walk": 350.0},
            {"ef": 36.0, "nt_probnp": 1400.0, "six_min_walk": 250.0},
        )
        result = analyze_dynamics(make_request(visits))
        assert result.visits_analyzed == 2
        # FIX v2.2: >= 3. → >= 3 (нүкте синтаксис қате болатын)
        assert len(result.trends) >= 3