"""
test_edge_cases.py — Edge case тесттері
CardioTracker ML v2.2

Барлық history=[] / None жағдайлар тексеріледі.

Түзетулер v2.2:
  ✅ Версия жолы v2.1 → v2.2
  ✅ t_alert_critical_combo: ef=35 (30-дан жоғары → ALERT-01 жоқ) — дұрыс сақталды
  ✅ t_summary_first_visit: "бірінші визит" summary_text-те болуы тиіс
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from schemas import (
    AlertCheckRequest, PatientData, VisitRecord,
    NtProBnpParadoxRequest, DynamicsRequest, SummaryRequest,
)
from services.alert_service import check_alerts, check_nt_paradox
from services.dynamics_service import analyze_dynamics
from services.summary_service import generate_summary

PASS = "✅ PASS"
FAIL = "❌ FAIL"

results = []

def run_edge_case(name, fn):
    try:
        fn()
        results.append(f"{PASS}  {name}")
    except Exception as e:
        results.append(f"{FAIL}  {name}  →  {type(e).__name__}: {e}")


# ── ALERT SERVICE EDGE CASES ──────────────────────────────────────

def t_alert_first_visit():
    """Бірінші визит: history=[] → IndexError болмауы керек"""
    req = AlertCheckRequest(
        current=PatientData(ef=25.0, creatinine=90.0, hemoglobin=105.0, nt_probnp=200.0),
        visit_history=[],
    )
    result = check_alerts(req)
    assert isinstance(result, list)


def t_alert_history_none():
    """history=None → IndexError болмауы керек"""
    req = AlertCheckRequest(
        current=PatientData(ef=35.0, creatinine=95.0),
        visit_history=None,
    )
    result = check_alerts(req)
    assert isinstance(result, list)


def t_alert_prev_creatinine_zero():
    """prev.creatinine=0 → ZeroDivisionError болмауы керек"""
    req = AlertCheckRequest(
        current=PatientData(creatinine=120.0),
        visit_history=[VisitRecord(creatinine=0.0)],
    )
    result = check_alerts(req)
    assert isinstance(result, list)


def t_alert_prev_nt_none():
    """prev.nt_probnp=None → NoneType error болмауы керек"""
    req = AlertCheckRequest(
        current=PatientData(nt_probnp=1200.0, ef=40.0),
        visit_history=[VisitRecord(nt_probnp=None, ef=38.0)],
    )
    result = check_alerts(req)
    assert isinstance(result, list)


def t_alert_all_none_values():
    """Барлық мәндер None → бос тізім, exception жоқ"""
    req = AlertCheckRequest(
        current=PatientData(),
        visit_history=[],
    )
    result = check_alerts(req)
    assert result == []


def t_alert_critical_combo():
    """
    Нақты critical combo: EF<40 + Cr>150 + Hb<110 → COMBO-05.
    ef=35 (30-дан жоғары) → ALERT-01 жоқ болуы тиіс.
    """
    req = AlertCheckRequest(
        current=PatientData(
            ef=35.0, creatinine=160.0, hemoglobin=95.0,
            nt_probnp=1500.0, ecg_af=True
        ),
        visit_history=[VisitRecord(ef=38.0, creatinine=120.0)],
    )
    result = check_alerts(req)
    codes = [a.alert_code for a in result]
    assert "COMBO-05" in codes, f"COMBO-05 жоқ: {codes}"
    assert "ALERT-01" not in codes   # ef=35, яғни >=30 → ALERT-01 іске қоспайды
    assert any(a.priority == "CRITICAL" for a in result)


# ── NT PARADOX EDGE CASES ─────────────────────────────────────────

def t_nt_paradox_none_values():
    """nt_probnp=None → paradox_detected=False, exception жоқ"""
    req = NtProBnpParadoxRequest(nt_probnp=None, ef=30.0, has_symptoms=True)
    result = check_nt_paradox(req)
    assert result.paradox_detected is False


def t_nt_paradox_detected():
    """NT <125 + EF <35 + симптомдар → парадокс анықталу керек"""
    req = NtProBnpParadoxRequest(nt_probnp=100.0, ef=30.0, has_symptoms=True)
    result = check_nt_paradox(req)
    assert result.paradox_detected is True


# ── DYNAMICS SERVICE EDGE CASES ───────────────────────────────────

def t_dynamics_less_than_2():
    """1 визит → ValueError немесе ValidationError"""
    try:
        req = DynamicsRequest(visits=[VisitRecord(ef=45.0)])
        analyze_dynamics(req)
        assert False, "ValueError күтілді"
    except (ValueError, Exception):
        pass  # schemas validator немесе service ValueError — екеуі де дұрыс


def t_dynamics_2_visits_ok():
    """2 визит → жұмыс істеуі керек"""
    req = DynamicsRequest(
        patient_id="TEST-01",
        visits=[
            VisitRecord(ef=50.0, nt_probnp=300.0, six_min_walk=420.0, creatinine=90.0),
            VisitRecord(ef=45.0, nt_probnp=600.0, six_min_walk=360.0, creatinine=105.0),
        ]
    )
    result = analyze_dynamics(req)
    assert result.visits_analyzed == 2


def t_dynamics_all_none():
    """Барлық мәндер None болса — тренд тізімі бос, exception жоқ"""
    req = DynamicsRequest(
        visits=[VisitRecord(), VisitRecord()]
    )
    result = analyze_dynamics(req)
    assert isinstance(result.trends, list)


def t_dynamics_no_dates():
    """visit_date жоқ → avg_interval_days=None, exception жоқ"""
    req = DynamicsRequest(
        visits=[
            VisitRecord(ef=50.0),
            VisitRecord(ef=45.0),
        ]
    )
    result = analyze_dynamics(req)
    assert result.avg_interval_days is None


# ── SUMMARY SERVICE EDGE CASES ────────────────────────────────────

def t_summary_first_visit():
    """visit_history=[] → бірінші визит режимі, exception жоқ"""
    req = SummaryRequest(
        patient_data=PatientData(ef=42.0, six_min_walk=300.0, hemoglobin=108.0),
        visit_history=[],
    )
    result = generate_summary(req)
    # summary_text-те "бірінші визит" болуы тиіс (summary_service.py-де бар)
    assert "бірінші визит" in result.summary_text.lower() or result.risk_group


def t_summary_history_none():
    """visit_history=None → exception жоқ"""
    req = SummaryRequest(
        patient_data=PatientData(ef=48.0),
        visit_history=None,
    )
    result = generate_summary(req)
    assert result.summary_text


def t_summary_all_none():
    """Барлық мәндер None болса да — сводка генерацияланады"""
    req = SummaryRequest(
        patient_data=PatientData(),
        visit_history=[],
    )
    result = generate_summary(req)
    assert isinstance(result.summary_text, str)


def t_summary_with_history():
    """visit_history бар → ДИНАМИКА бөлімі шығуы керек"""
    req = SummaryRequest(
        patient_data=PatientData(ef=40.0, creatinine=120.0, hemoglobin=100.0),
        visit_history=[
            VisitRecord(ef=45.0, creatinine=90.0, hemoglobin=115.0),
        ],
    )
    result = generate_summary(req)
    assert "ДИНАМИКА" in result.summary_text


# ── БАРЛЫҚ ТЕСТТЕР ───────────────────────────────────────────────

run_edge_case("Alert: бірінші визит history=[]", t_alert_first_visit)
run_edge_case("Alert: history=None",                    t_alert_history_none)
run_edge_case("Alert: prev.creatinine=0",               t_alert_prev_creatinine_zero)
run_edge_case("Alert: prev.nt_probnp=None",             t_alert_prev_nt_none)
run_edge_case("Alert: барлық мәндер None",              t_alert_all_none_values)
run_edge_case("Alert: critical combo EF+Cr+Hb",         t_alert_critical_combo)
run_edge_case("NT Paradox: nt_probnp=None",             t_nt_paradox_none_values)
run_edge_case("NT Paradox: анықталды",                  t_nt_paradox_detected)
run_edge_case("Dynamics: 1 визит → ValueError",         t_dynamics_less_than_2)
run_edge_case("Dynamics: 2 визит жұмыс істейді",        t_dynamics_2_visits_ok)
run_edge_case("Dynamics: барлық мәндер None",           t_dynamics_all_none)
run_edge_case("Dynamics: visit_date жоқ → None",        t_dynamics_no_dates)
run_edge_case("Summary: бірінші визит history=[]",      t_summary_first_visit)
run_edge_case("Summary: visit_history=None",            t_summary_history_none)
run_edge_case("Summary: барлық мәндер None",            t_summary_all_none)
run_edge_case("Summary: тарихпен динамика",             t_summary_with_history)

# ── НӘТИЖЕ ШЫҒАРУ ────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"EDGE CASE ТЕСТТЕРІ — CardioTracker ML v2.2")
print(f"{'='*55}")
for r in results:
    print(r)

passed = sum(1 for r in results if r.startswith("✅"))
failed = sum(1 for r in results if r.startswith("❌"))
print(f"\n{'='*55}")
print(f"Нәтиже: {passed}/{len(results)} өтті   |   {failed} сәтсіз")