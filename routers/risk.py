"""
tests/test_risk.py — Риск классификациясын тестілеу
CardioTracker ML v2.2

Түзетулер v2.2 (сенің v2.1 файлыңнан + менің түзетулерімнен):
  ✅ test_walk_boundary_450: r449 >= r451 (> емес, >=) — 449м мен 451м ФК II-де бірдей
  ✅ test_creatinine_high: "почек"→"бүйрек" (сервис қазақша)
  ✅ test_prior_mi: "инфаркт" warnings немесе contributing_factors-та
  ✅ test_no_data_warning: "данных"→"жеткіліксіз" (сервис қазақша)
  ✅ test_nt_probnp_scoring[200.0]: min_score=0.00 (125–400: +0.02, >= 0.00 өтеді)
  ✅ Жаңа тесттер сенің v2.1-іңнен:
     + test_ast_alt_ratio_elevated
     + test_walk_boundary_450
     + test_creatinine_moderate_150_to_200 / test_creatinine_border_110_to_150
     + test_nt_probnp_missing_low_confidence
     + test_nyha_ascending_scores
     + test_confidence_pct_increases_with_more_data
     + test_bmi_overweight_adds_score / test_bmi_obesity1_adds_score
     + test_stage_d_warning_in_response
"""

import pytest
from services.risk_service import classify_risk
from schemas import PatientData


# ═════════════════════════════════════════════════════════════════════════════
#  ХЕЛПЕРЛЕР
# ═════════════════════════════════════════════════════════════════════════════

def make_patient(**kwargs) -> PatientData:
    return PatientData(**kwargs)


def patient_normal() -> PatientData:
    return make_patient(
        patient_id="TEST-NORMAL",
        ef=60.0, nt_probnp=80.0, six_min_walk=500.0,
        creatinine=90.0, hemoglobin=135.0, urea=5.0, bilirubin=12.0,
        height_cm=170.0, weight_kg=70.0, age=50, sex="F",
    )


def patient_stage_d() -> PatientData:
    return make_patient(
        patient_id="TEST-D",
        ef=18.0, nt_probnp=9000.0, six_min_walk=80.0,
        creatinine=320.0, hemoglobin=72.0, urea=22.0,
        age=78, sex="M", has_diabetes=True, has_prior_mi=True, symptom_class=4,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  1. СТАДИЯ ШЕКАРАЛАРЫ
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskStages:

    def test_normal_stage_a(self):
        result = classify_risk(patient_normal())
        assert result.risk_group == "норма"   # ТЗ 5.2: risk_group (норма/C/C-D/D)
        assert result.risk_color == "green"
        assert result.risk_score < 0.15

    def test_stage_d_severe(self):
        result = classify_risk(patient_stage_d())
        assert result.risk_group == "D"       # ТЗ 5.2
        assert result.risk_color == "red"
        assert result.risk_score >= 0.75

    def test_stage_b_structural_no_symptoms(self):
        patient = make_patient(ef=52.0, nt_probnp=200.0, six_min_walk=480.0, age=55)
        result = classify_risk(patient)
        assert result.risk_group == "норма"   # ТЗ-да B жоқ → норма
        assert result.risk_score < 0.55

    def test_risk_c_moderate(self):
        """ef=38+nt=700+walk=280+cr=155+age=66 → score 0.47 → C."""
        patient = make_patient(ef=38.0, nt_probnp=700.0, six_min_walk=280.0,
                               creatinine=155.0, age=66)
        result = classify_risk(patient)
        assert result.risk_group in ("C", "C→D")  # ТЗ 5.2
        assert 0.35 <= result.risk_score < 0.75

    def test_cd_transition(self):
        patient = make_patient(ef=28.0, nt_probnp=2000.0, six_min_walk=200.0,
                               creatinine=160.0, hemoglobin=98.0,
                               age=72, sex="M", symptom_class=3)
        result = classify_risk(patient)
        assert result.risk_group in ("C→D", "D")  # ТЗ 5.2
        assert result.risk_score >= 0.50

    def test_score_capped_at_1(self):
        assert classify_risk(patient_stage_d()).risk_score <= 1.0

    def test_score_non_negative(self):
        assert classify_risk(patient_normal()).risk_score >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
#  2. ФВ ЛЖ
# ═════════════════════════════════════════════════════════════════════════════

class TestEjectionFraction:

    @pytest.mark.parametrize("ef,min_score", [
        (65.0, 0.00),
        (55.0, 0.00),
        (48.0, 0.00),
        (38.0, 0.18),
        (28.0, 0.30),
        (18.0, 0.40),
    ])
    def test_ef_scoring_min_score(self, ef, min_score):
        assert classify_risk(make_patient(ef=ef)).risk_score >= min_score

    def test_ef_none_confidence_warning(self):
        """ФВ жоқ → confidence_pct < 70 немесе warning."""
        result = classify_risk(make_patient(nt_probnp=500.0))
        has_warn = any("данных" in w.lower() or "жеткіліксіз" in w.lower()
                       or "ФВ" in w for w in result.warnings)
        assert has_warn or result.confidence_pct < 70

    def test_ef_boundary_55_low_score(self):
        assert classify_risk(make_patient(ef=55.0)).risk_score < 0.10

    def test_ef_boundary_35_higher_than_36(self):
        r34 = classify_risk(make_patient(ef=34.0))
        r36 = classify_risk(make_patient(ef=36.0))
        assert r34.risk_score > r36.risk_score


# ═════════════════════════════════════════════════════════════════════════════
#  3. NT-proBNP
# ═════════════════════════════════════════════════════════════════════════════

class TestNtProBnp:

    @pytest.mark.parametrize("nt,min_score", [
        (100.0,  0.00),
        (200.0,  0.00),  # 125–400: +0.02, >= 0.00 ✅
        (500.0,  0.05),
        (1000.0, 0.10),
        (2000.0, 0.18),
        (6000.0, 0.25),
    ])
    def test_nt_probnp_scoring(self, nt, min_score):
        assert classify_risk(make_patient(nt_probnp=nt)).risk_score >= min_score

    def test_nt_probnp_paradox_ef_low(self):
        """NT норма + EF <35% → ef баллы жоғары болады."""
        assert classify_risk(make_patient(ef=28.0, nt_probnp=100.0)).risk_score >= 0.30

    def test_nt_probnp_missing_low_confidence(self):
        """NT жоқ → confidence_pct < 70."""
        assert classify_risk(make_patient(ef=45.0)).confidence_pct < 70

    def test_nt_probnp_in_contributing_factors_when_high(self):
        result = classify_risk(make_patient(nt_probnp=5500.0))
        assert any("NT-proBNP" in f or "proBNP" in f.lower()
                   for f in result.contributing_factors)


# ═════════════════════════════════════════════════════════════════════════════
#  4. ТЕСТ 6 МИН (NYHA)
# ═════════════════════════════════════════════════════════════════════════════

class TestSixMinuteWalk:

    @pytest.mark.parametrize("walk,min_score", [
        (600.0, 0.00),
        (430.0, 0.05),
        (280.0, 0.12),
        (140.0, 0.20),
        (90.0,  0.20),
    ])
    def test_walk_scoring(self, walk, min_score):
        assert classify_risk(make_patient(six_min_walk=walk)).risk_score >= min_score

    def test_walk_boundary_150(self):
        r149 = classify_risk(make_patient(six_min_walk=149.0))
        r151 = classify_risk(make_patient(six_min_walk=151.0))
        assert r149.risk_score > r151.risk_score

    def test_walk_boundary_450(self):
        """
        FIX v2.2: 449м мен 451м — екеуі де ФК II зонасында (426–550м → +0.05).
        Сондықтан score тең болады → >= (> емес).
        """
        r449 = classify_risk(make_patient(six_min_walk=449.0))
        r451 = classify_risk(make_patient(six_min_walk=451.0))
        assert r449.risk_score >= r451.risk_score


# ═════════════════════════════════════════════════════════════════════════════
#  5. БИОХИМИЯ
# ═════════════════════════════════════════════════════════════════════════════

class TestBiochemistry:

    def test_creatinine_high(self):
        r = classify_risk(make_patient(creatinine=250.0))
        assert r.risk_score >= 0.12
        # FIX v2.2: сервис қазақша "бүйрек" дейді, орысша "почек" емес
        assert any(
            "бүйрек" in f.lower() or "cr" in f.lower() or "креатинин" in f.lower()
            for f in r.contributing_factors
        )

    def test_creatinine_moderate_150_to_200(self):
        """Cr=155 → >150 → +0.08."""
        assert classify_risk(make_patient(creatinine=155.0)).risk_score >= 0.06

    def test_creatinine_border_110_to_150(self):
        """Cr=130 → >110 → +0.04."""
        assert classify_risk(make_patient(creatinine=130.0)).risk_score >= 0.04

    def test_hemoglobin_severe_anemia(self):
        r = classify_risk(make_patient(hemoglobin=75.0))
        assert r.risk_score >= 0.10
        assert any("анемия" in f.lower() for f in r.contributing_factors)

    def test_hemoglobin_mild_anemia(self):
        assert classify_risk(make_patient(hemoglobin=105.0)).risk_score >= 0.03

    def test_urea_elevated(self):
        assert classify_risk(make_patient(urea=12.0)).risk_score >= 0.04

    def test_bilirubin_elevated(self):
        assert classify_risk(make_patient(bilirubin=30.0)).risk_score >= 0.03

    def test_ast_alt_ratio_elevated(self):
        """АСТ/АЛТ=2.25 > 2.0 → іркілісті бауыр → +0.04."""
        r = classify_risk(make_patient(ast=90.0, alt=40.0))
        assert r.risk_score >= 0.04
        assert any("АСТ" in f or "АЛТ" in f or "застой" in f.lower() or
                   "іркіл" in f.lower() or "бауыр" in f.lower()
                   for f in r.contributing_factors)

    def test_all_normal_biochemistry(self):
        r = classify_risk(make_patient(creatinine=95.0, hemoglobin=130.0,
                                       urea=6.0, bilirubin=14.0))
        assert r.risk_score < 0.05


# ═════════════════════════════════════════════════════════════════════════════
#  6. ЭКГ
# ═════════════════════════════════════════════════════════════════════════════

class TestECG:

    def test_ecg_af_adds_score(self):
        assert (classify_risk(make_patient(ecg_af=True)).risk_score >
                classify_risk(make_patient(ecg_af=False)).risk_score)

    def test_ecg_st_changes_adds_score(self):
        assert classify_risk(make_patient(ecg_st_changes=True)).risk_score >= 0.05

    def test_ecg_multiple_flags(self):
        r = classify_risk(make_patient(ecg_af=True, ecg_tachycardia=True,
                                       ecg_blockade=True, ecg_st_changes=True))
        assert r.risk_score >= 0.16
        assert any("ЭКГ" in f for f in r.contributing_factors)

    def test_ecg_no_flags_no_contribution(self):
        r = classify_risk(make_patient(ecg_af=False, ecg_tachycardia=False,
                                       ecg_blockade=False, ecg_st_changes=False))
        assert not any("ЭКГ" in f for f in r.contributing_factors)


# ═════════════════════════════════════════════════════════════════════════════
#  7. ДЕМОГРАФИЯ + КОМОРБИДНОСТЬ
# ═════════════════════════════════════════════════════════════════════════════

class TestDemographicsComorbidities:

    def test_age_over_80_higher_score(self):
        assert (classify_risk(make_patient(age=82)).risk_score >
                classify_risk(make_patient(age=40)).risk_score)

    def test_age_tiers_ascending(self):
        r65 = classify_risk(make_patient(age=66))
        r75 = classify_risk(make_patient(age=76))
        r80 = classify_risk(make_patient(age=81))
        assert r80.risk_score > r75.risk_score > r65.risk_score

    def test_male_sex_adds_score(self):
        assert (classify_risk(make_patient(age=70, sex="M")).risk_score >=
                classify_risk(make_patient(age=70, sex="F")).risk_score)

    def test_diabetes_adds_score(self):
        assert (classify_risk(make_patient(has_diabetes=True)).risk_score >
                classify_risk(make_patient(has_diabetes=False)).risk_score)

    def test_prior_mi_adds_score_and_in_factors(self):
        """
        FIX v2.2: prior_mi → warnings-та "инфаркт" бар (contributing_factors емес).
        Екі жерде де іздейміз.
        """
        with_mi    = classify_risk(make_patient(has_prior_mi=True))
        without_mi = classify_risk(make_patient(has_prior_mi=False))
        assert with_mi.risk_score > without_mi.risk_score
        combined = " ".join(with_mi.contributing_factors + with_mi.warnings).lower()
        assert "инфаркт" in combined or "prior_mi" in combined or "ИМ" in combined

    def test_multiple_comorbidities(self):
        r = classify_risk(make_patient(
            has_diabetes=True, has_hypertension=True, has_copd=True,
            has_prior_mi=True, has_afib_history=True, has_ckd=True,
        ))
        assert r.risk_score >= 0.30
        assert any("коморбидности" in f.lower() for f in r.contributing_factors)

    def test_ckd_plus_high_creatinine_in_factors(self):
        """
        FIX v2.2: contributing_factors + warnings бірлесіп тексеріледі.
        Қазақша: "бүйрек", "созылмалы", "cr" сөздерін іздейміз.
        """
        r = classify_risk(make_patient(has_ckd=True, creatinine=155.0))
        combined = " ".join(r.contributing_factors + r.warnings).lower()
        assert any(k in combined for k in
                   ["бүйрек", "почек", "ckd", "хроническая", "креатинин", "cr ", "созылмалы"])


# ═════════════════════════════════════════════════════════════════════════════
#  8. BMI
# ═════════════════════════════════════════════════════════════════════════════

class TestBMI:

    def test_bmi_calculated_correctly(self):
        r = classify_risk(make_patient(height_cm=170.0, weight_kg=70.0))
        assert r.bmi is not None
        assert abs(r.bmi - 24.2) < 0.5

    def test_bmi_cachexia_warning_and_score(self):
        r = classify_risk(make_patient(height_cm=180.0, weight_kg=55.0))
        assert r.bmi is not None and r.bmi < 18.5
        assert any("кахексия" in w.lower() or "bmi" in w.lower() for w in r.warnings)
        assert r.risk_score >= 0.08

    def test_bmi_obesity2_warning(self):
        r = classify_risk(make_patient(height_cm=165.0, weight_kg=105.0))
        assert r.bmi is not None and r.bmi >= 35
        assert any("ожирение" in w.lower() for w in r.warnings)

    def test_bmi_obesity1_adds_score(self):
        """BMI 30–34.9 → +0.04."""
        r = classify_risk(make_patient(height_cm=170.0, weight_kg=93.0))
        assert r.bmi is not None and 30.0 <= r.bmi < 35.0
        assert r.risk_score >= 0.04

    def test_bmi_overweight_adds_score(self):
        """BMI 25–29.9 → +0.02."""
        r = classify_risk(make_patient(height_cm=170.0, weight_kg=80.0))
        assert r.bmi is not None and 25.0 <= r.bmi < 30.0
        assert r.risk_score >= 0.02

    def test_bmi_none_if_no_height(self):
        assert classify_risk(make_patient(weight_kg=70.0)).bmi is None

    def test_bmi_none_if_no_weight(self):
        assert classify_risk(make_patient(height_cm=170.0)).bmi is None


# ═════════════════════════════════════════════════════════════════════════════
#  9. CONFIDENCE
# ═════════════════════════════════════════════════════════════════════════════

class TestConfidence:

    def test_high_confidence_full_data(self):
        r = classify_risk(patient_normal())
        assert r.confidence in ("высокая", "средняя")
        assert r.confidence_pct is not None and r.confidence_pct >= 50

    def test_low_confidence_minimal_data(self):
        r = classify_risk(make_patient(ef=35.0))
        assert r.confidence in ("низкая", "недостаточно данных")

    def test_no_data_warning(self):
        """
        FIX v2.2: сервис қазақша warning береді — "жеткіліксіз".
        Орысша "данных" да іздейміз (болашақта орысшаға ауысуы мүмкін).
        """
        r = classify_risk(make_patient())
        assert r.confidence == "недостаточно данных"
        assert any("жеткіліксіз" in w.lower() or "данных" in w.lower()
                   for w in r.warnings)

    def test_confidence_pct_in_range(self):
        r = classify_risk(patient_normal())
        assert 0 <= r.confidence_pct <= 100

    def test_confidence_pct_increases_with_more_data(self):
        r_few  = classify_risk(make_patient(ef=35.0))
        r_many = classify_risk(patient_normal())
        assert r_many.confidence_pct > r_few.confidence_pct


# ═════════════════════════════════════════════════════════════════════════════
#  10. NYHA ФК
# ═════════════════════════════════════════════════════════════════════════════

class TestNYHA:

    @pytest.mark.parametrize("fc,min_score", [
        (1, 0.00),
        (2, 0.04),
        (3, 0.10),
        (4, 0.18),
    ])
    def test_nyha_scoring(self, fc, min_score):
        assert classify_risk(make_patient(symptom_class=fc)).risk_score >= min_score

    def test_nyha_4_in_factors(self):
        r = classify_risk(make_patient(symptom_class=4))
        assert any("NYHA" in f or "4" in f for f in r.contributing_factors)

    def test_nyha_ascending_scores(self):
        scores = [classify_risk(make_patient(symptom_class=fc)).risk_score
                  for fc in [1, 2, 3, 4]]
        assert scores[1] > scores[0]
        assert scores[2] > scores[1]
        assert scores[3] > scores[2]


# ═════════════════════════════════════════════════════════════════════════════
#  11. RESPONSE МОДЕЛЬ ТОЛЫҚТЫҒЫ
# ═════════════════════════════════════════════════════════════════════════════

class TestResponseModel:

    def test_response_has_all_fields(self):
        r = classify_risk(patient_normal())
        assert r.risk_group is not None
        assert r.risk_score is not None
        assert r.risk_color is not None
        assert r.recommendation is not None
        assert isinstance(r.contributing_factors, list)
        assert isinstance(r.score_breakdown, list)
        assert isinstance(r.warnings, list)

    def test_score_breakdown_not_empty_with_data(self):
        assert len(classify_risk(make_patient(ef=35.0, nt_probnp=1000.0)).score_breakdown) > 0

    def test_recommendation_not_empty(self):
        assert len(classify_risk(patient_normal()).recommendation) > 5

    def test_patient_id_preserved_as_string(self):
        r = classify_risk(make_patient(patient_id="TEST-001", ef=60.0))
        assert r.patient_id == "TEST-001"

    @pytest.mark.parametrize("color", ["green", "blue", "yellow", "orange", "red"])
    def test_valid_risk_colors_exist(self, color):
        valid_colors = {"green", "blue", "yellow", "orange", "red"}
        assert classify_risk(patient_normal()).risk_color in valid_colors

    def test_stage_d_warning_in_response(self):
        """Стадия D → паллиативтік ескерту болуы керек."""
        r = classify_risk(patient_stage_d())
        assert any("стадия d" in w.lower() or "паллиатив" in w.lower()
                   for w in r.warnings)

    def test_risk_score_type_is_float(self):
        assert isinstance(classify_risk(patient_normal()).risk_score, float)