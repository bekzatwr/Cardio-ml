"""
tests/test_ml_prep.py — ML Validator тесттері
CardioTracker ML v2.2

Тест топтары:
  1. Деректерді flatten жасау (_flatten_records)
  2. Corrupted жазбаларды анықтау (_detect_corrupted)
  3. Пропуск статистикасы (_compute_missing_stats)
  4. Класс үлестірімі (_compute_class_distribution)
  5. RandomForest оқыту (_train_baseline_model)
  6. analyze_collected_data — интеграциялық тест
  7. Edge cases — бос, None, толық пустой жазбалар
"""

import pytest
from services.ml_validator_service import (
    analyze_collected_data,
    _flatten_records,
    _detect_corrupted,
    _compute_missing_stats,
    _compute_class_distribution,
    _train_baseline_model,
    CRITICAL_FEATURES,
    V3_TARGET_RECORDS,
)


# ═════════════════════════════════════════════════════════════════
# MOCK ДЕРЕКТЕР ЖАСАЙТЫН ХЕЛПЕРЛЕР
# ═════════════════════════════════════════════════════════════════

def make_record(
    record_id: str = "td_1",
    ef_lv: float = 38.0,
    nt_probnp: float = 1200.0,
    six_min_walk: float = 280.0,
    hemoglobin: float = 108.0,
    creatinine: float = 95.0,
    urea: float = 6.5,
    risk_group: str = "C",
    label_type: str = "gold_label",
    doctor_id: int = 2,
) -> dict:
    """Идеальды толық жазба."""
    return {
        "record_id":     record_id,
        "visit_id":      next((int(p) for p in record_id.split("_") if p.isdigit()), 1),
        "patient_id":    42,
        "doctor_id":     doctor_id,
        "label_type":    label_type,
        "created_at":    "2024-02-15T09:30:00",
        "input_features": {
            "ef_lv":        ef_lv,
            "nt_probnp":    nt_probnp,
            "six_min_walk": six_min_walk,
            "hemoglobin":   hemoglobin,
            "creatinine":   creatinine,
            "urea":         urea,
            "risk_group":   risk_group,
            "risk_score":   0.48,
        },
        "doctor_decision": {
            "diagnosis":      "ХСН ФК III",
            "risk_group":     risk_group,
            "medications":    [{"name": "Бисопролол", "dose": "5мг"}],
            "next_visit_days": 14,
            "change_reason":  "lab_deterioration",
        },
    }


def make_missing_ef_record(record_id: str = "td_miss_1") -> dict:
    """ФВ/ЛЖ жоқ жазба (бірақ nt_probnp бар → corrupted емес)."""
    rec = make_record(record_id)
    rec["input_features"]["ef_lv"] = None
    return rec


def make_corrupted_record(record_id: str = "td_corrupt_1") -> dict:
    """Барлық маңызды параметрлер жоқ → corrupted."""
    return {
        "record_id":     record_id,
        "visit_id":      999,
        "patient_id":    99,
        "doctor_id":     2,
        "label_type":    "secondary_label",
        "created_at":    "2024-02-15T09:30:00",
        "input_features": {
            "ef_lv":        None,
            "nt_probnp":    None,
            "six_min_walk": None,
            "hemoglobin":   105.0,
        },
        "doctor_decision": {
            "risk_group": "C",
            "medications": [],
        },
    }


def make_fully_empty_record(record_id: str = "td_empty_1") -> dict:
    """Толық бос жазба."""
    return {
        "record_id":      record_id,
        "visit_id":       0,
        "patient_id":     None,
        "doctor_id":      None,
        "label_type":     None,
        "input_features":  {},
        "doctor_decision": {},
    }


def make_dataset(n_good: int = 15, n_missing_ef: int = 2, n_corrupted: int = 1, n_empty: int = 1) -> list:
    """
    Тест датасеті:
      - n_good: толық идеальды жазбалар (риск тобы әртүрлі)
      - n_missing_ef: ФВ жоқ бірақ corrupted емес
      - n_corrupted: барлық маңызды параметрлер жоқ
      - n_empty: толық бос
    """
    records = []

    risk_groups = ["норма", "C", "C", "C", "C→D", "C→D", "D"]
    for i in range(n_good):
        rg = risk_groups[i % len(risk_groups)]
        records.append(make_record(
            record_id=f"td_{i+1}",
            ef_lv=40.0 + i,
            nt_probnp=500.0 + i * 50,
            six_min_walk=300.0 + i * 10,
            risk_group=rg,
            label_type="gold_label" if i % 3 == 0 else "secondary_label",
        ))

    for i in range(n_missing_ef):
        records.append(make_missing_ef_record(f"td_miss_{i+1}"))

    for i in range(n_corrupted):
        records.append(make_corrupted_record(f"td_corrupt_{i+1}"))

    for i in range(n_empty):
        records.append(make_fully_empty_record(f"td_empty_{i+1}"))

    return records


# ═════════════════════════════════════════════════════════════════
# 1. FLATTEN ТЕСТТЕРІ
# ═════════════════════════════════════════════════════════════════

class TestFlattenRecords:

    def test_single_record_flattened(self):
        """Бір жазба дұрыс flatten болады."""
        rec = make_record("td_1")
        df  = _flatten_records([rec])
        assert len(df) == 1
        assert "feat_ef_lv"     in df.columns
        assert "feat_nt_probnp" in df.columns
        assert "risk_group"     in df.columns
        assert df["feat_ef_lv"].iloc[0] == 38.0

    def test_empty_list_returns_empty_df(self):
        """Бос тізім → бос DataFrame."""
        df = _flatten_records([])
        assert df.empty

    def test_nested_features_extracted(self):
        """input_features ішіндегі барлық өрістер flatten болады."""
        df = _flatten_records([make_record("td_1")])
        for feat in ["ef_lv", "nt_probnp", "six_min_walk", "hemoglobin"]:
            assert f"feat_{feat}" in df.columns

    def test_doctor_decision_risk_group_extracted(self):
        """doctor_decision.risk_group → risk_group бағаны."""
        df = _flatten_records([make_record("td_1", risk_group="C→D")])
        assert df["risk_group"].iloc[0] == "C→D"

    def test_numeric_conversion(self):
        """Сандық бағандар float типінде."""
        df = _flatten_records([make_record("td_1")])
        assert df["feat_ef_lv"].dtype in ["float64", "float32"]

    def test_none_values_become_nan(self):
        """None мәндер NaN болады."""
        import pandas as pd
        rec = make_missing_ef_record("td_miss")
        df  = _flatten_records([rec])
        assert pd.isna(df["feat_ef_lv"].iloc[0])

    def test_label_type_preserved(self):
        """label_type сақталады."""
        df = _flatten_records([make_record("td_1", label_type="gold_label")])
        assert df["label_type"].iloc[0] == "gold_label"


# ═════════════════════════════════════════════════════════════════
# 2. CORRUPTED АНЫҚТАУ ТЕСТТЕРІ
# ═════════════════════════════════════════════════════════════════

class TestDetectCorrupted:

    def test_good_record_not_corrupted(self):
        """Толық жазба corrupted емес."""
        df = _flatten_records([make_record("td_1")])
        valid, corrupted = _detect_corrupted(df)
        assert len(valid)     == 1
        assert len(corrupted) == 0

    def test_missing_ef_not_corrupted(self):
        """ФВ жоқ бірақ nt_probnp бар → corrupted емес."""
        df = _flatten_records([make_missing_ef_record()])
        valid, corrupted = _detect_corrupted(df)
        assert len(valid)     == 1
        assert len(corrupted) == 0

    def test_all_critical_missing_is_corrupted(self):
        """ef, nt_probnp, six_min_walk барлығы None → corrupted."""
        df = _flatten_records([make_corrupted_record()])
        valid, corrupted = _detect_corrupted(df)
        assert len(valid)     == 0
        assert len(corrupted) == 1

    def test_empty_record_is_corrupted(self):
        """Толық бос жазба → corrupted."""
        df = _flatten_records([make_fully_empty_record()])
        valid, corrupted = _detect_corrupted(df)
        assert len(corrupted) == 1

    def test_mixed_dataset(self):
        """Аралас датасет — corrupted дұрыс бөлінеді."""
        records = [
            make_record("td_1"),
            make_record("td_2"),
            make_missing_ef_record("td_miss"),
            make_corrupted_record("td_corrupt"),
            make_fully_empty_record("td_empty"),
        ]
        df = _flatten_records(records)
        valid, corrupted = _detect_corrupted(df)
        assert len(valid)     == 3  # td_1, td_2, td_miss
        assert len(corrupted) == 2  # td_corrupt, td_empty

    def test_empty_df_returns_empty(self):
        """Бос DataFrame → екі бос DataFrame."""
        import pandas as pd
        valid, corrupted = _detect_corrupted(pd.DataFrame())
        assert valid.empty
        assert corrupted.empty


# ═════════════════════════════════════════════════════════════════
# 3. ПРОПУСК СТАТИСТИКАСЫ ТЕСТТЕРІ
# ═════════════════════════════════════════════════════════════════

class TestMissingStats:

    def test_no_missing_in_perfect_data(self):
        """Толық деректерде пропуск жоқ."""
        records = [make_record(f"td_{i}") for i in range(5)]
        df      = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        stats   = _compute_missing_stats(valid)
        ef_stat = next((s for s in stats if s["feature"] == "ef_lv"), None)
        assert ef_stat is not None
        assert ef_stat["missing_pct"] == 0.0

    def test_missing_ef_counted(self):
        """ФВ жоқ жазбалар % дұрыс есептеледі."""
        records = [make_record("td_1"), make_missing_ef_record("td_miss")]
        df      = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        stats   = _compute_missing_stats(valid)
        ef_stat = next((s for s in stats if s["feature"] == "ef_lv"), None)
        assert ef_stat is not None
        assert ef_stat["missing_count"] == 1
        assert ef_stat["missing_pct"]   == 50.0

    def test_critical_features_marked(self):
        """CRITICAL_FEATURES is_critical=True деп белгіленеді."""
        df      = _flatten_records([make_record("td_1")])
        valid, _ = _detect_corrupted(df)
        stats   = _compute_missing_stats(valid)
        for s in stats:
            if s["feature"] in CRITICAL_FEATURES:
                assert s["is_critical"] is True

    def test_all_features_present_in_stats(self):
        """Барлық 8 клиникалық параметр статистикада бар."""
        from services.ml_validator_service import CLINICAL_FEATURES
        df      = _flatten_records([make_record("td_1")])
        valid, _ = _detect_corrupted(df)
        stats   = _compute_missing_stats(valid)
        stat_features = {s["feature"] for s in stats}
        for feat in CLINICAL_FEATURES:
            assert feat in stat_features

    def test_empty_df_returns_empty_stats(self):
        """Бос DataFrame → бос тізім."""
        import pandas as pd
        stats = _compute_missing_stats(pd.DataFrame())
        assert stats == []


# ═════════════════════════════════════════════════════════════════
# 4. КЛАСС ҮЛЕСТІРІМІ ТЕСТТЕРІ
# ═════════════════════════════════════════════════════════════════

class TestClassDistribution:

    def test_distribution_correct(self):
        """Класс саны дұрыс есептеледі."""
        records = [
            make_record("td_1", risk_group="норма"),
            make_record("td_2", risk_group="C"),
            make_record("td_3", risk_group="C"),
            make_record("td_4", risk_group="C→D"),
        ]
        df      = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        dist, _ = _compute_class_distribution(valid)
        assert dist.get("C")     == 2
        assert dist.get("норма") == 1
        assert dist.get("C→D")   == 1

    def test_no_warning_balanced(self):
        """Теңдестірілген деректерде ескерту жоқ."""
        records = [make_record(f"td_{i}", risk_group=["норма","C","C→D","D"][i%4])
                   for i in range(20)]
        df      = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        _, warning = _compute_class_distribution(valid)
        assert warning is None

    def test_warning_on_rare_class(self):
        """Сирек класс (<5%) ескерту шығарады."""
        records = [make_record(f"td_{i}", risk_group="C") for i in range(20)]
        records.append(make_record("td_rare", risk_group="D"))  # тек 1 жазба
        df      = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        _, warning = _compute_class_distribution(valid)
        assert warning is not None
        assert "D" in warning

    def test_empty_df_returns_empty(self):
        """Бос DataFrame → бос dict."""
        import pandas as pd
        dist, _ = _compute_class_distribution(pd.DataFrame())
        assert dist == {}


# ═════════════════════════════════════════════════════════════════
# 5. RANDOMFOREST ТЕСТТЕРІ
# ═════════════════════════════════════════════════════════════════

class TestTrainBaselineModel:

    def test_no_nan_error_with_missing_values(self):
        """
        НЕ КЕРЕК: ValueError: Input contains NaN.
        Медиана imputation жасалып, модель crash болмайды.
        """
        records = make_dataset(n_good=15, n_missing_ef=3)
        df      = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        # Бұл crash болмауы керек
        feat_imp, accuracy, warning = _train_baseline_model(valid, top_n=3)
        # Модель оқытылды немесе ескерту берілді — екеуі де дұрыс
        assert isinstance(feat_imp, list)

    def test_feature_importance_sum_near_1(self):
        """Feature importance жиынтығы ~1.0 болады."""
        records  = make_dataset(n_good=20)
        df       = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        feat_imp, _, _ = _train_baseline_model(valid, top_n=8)
        if feat_imp:
            total = sum(f["importance"] for f in feat_imp)
            assert 0.9 <= total <= 1.0 + 1e-6

    def test_top_n_respected(self):
        """top_n параметрі сақталады."""
        records  = make_dataset(n_good=20)
        df       = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        feat_imp, _, _ = _train_baseline_model(valid, top_n=3)
        if feat_imp:
            assert len(feat_imp) <= 3

    def test_rank_ascending(self):
        """Рейтинг 1-ден өседі."""
        records  = make_dataset(n_good=20)
        df       = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        feat_imp, _, _ = _train_baseline_model(valid, top_n=5)
        if feat_imp:
            ranks = [f["rank"] for f in feat_imp]
            assert ranks == list(range(1, len(feat_imp) + 1))

    def test_too_few_records_no_model(self):
        """< 10 жазба → модель оқытылмайды, warning қайтарылады."""
        records  = [make_record(f"td_{i}") for i in range(5)]
        df       = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        feat_imp, accuracy, warning = _train_baseline_model(valid, top_n=3)
        assert feat_imp  == []
        assert accuracy  is None
        assert warning   is not None

    def test_single_class_no_model(self):
        """1 ғана класс болса → модель оқытылмайды."""
        records  = [make_record(f"td_{i}", risk_group="C") for i in range(15)]
        df       = _flatten_records(records)
        valid, _ = _detect_corrupted(df)
        feat_imp, accuracy, warning = _train_baseline_model(valid, top_n=3)
        assert feat_imp == []
        assert warning  is not None


# ═════════════════════════════════════════════════════════════════
# 6. ИНТЕГРАЦИЯЛЫҚ ТЕСТТЕР (analyze_collected_data)
# ═════════════════════════════════════════════════════════════════

class TestAnalyzeCollectedData:

    def test_empty_input_returns_zero_report(self):
        """Бос тізім → нөлдік есеп, crash жоқ."""
        result = analyze_collected_data([])
        assert result["total_records"]    == 0
        assert result["valid_records"]    == 0
        assert result["model_trained"]    is False
        assert result["readiness_score"]  == 0.0
        assert result["readiness_label"]  == "недостаточно"

    def test_corrupted_filtered_out(self):
        """Corrupted жазбалар valid_records-тан шығарылады."""
        records = [
            make_record("td_1"),
            make_corrupted_record("td_c"),
            make_fully_empty_record("td_e"),
        ]
        result = analyze_collected_data(records)
        assert result["total_records"]    == 3
        assert result["valid_records"]    == 1
        assert result["corrupted_records"] == 2

    def test_corrupted_pct_correct(self):
        """Corrupted % дұрыс есептеледі."""
        records = [make_record(f"td_{i}") for i in range(8)]
        records += [make_corrupted_record(f"td_c_{i}") for i in range(2)]
        result = analyze_collected_data(records)
        assert result["corrupted_pct"] == 20.0

    def test_missing_pct_correct(self):
        """Пропуск % дұрыс есептеледі."""
        records = [make_record("td_1"), make_missing_ef_record("td_miss")]
        result  = analyze_collected_data(records)
        ef_stat = next(
            (s for s in result["missing_values"] if s["feature"] == "ef_lv"),
            None
        )
        assert ef_stat is not None
        assert ef_stat["missing_pct"] == 50.0

    def test_class_distribution_present(self):
        """Класс үлестірімі есепте бар."""
        records = make_dataset(n_good=15)
        result  = analyze_collected_data(records)
        assert isinstance(result["class_distribution"], dict)
        assert len(result["class_distribution"]) > 0

    def test_model_trained_with_enough_data(self):
        """≥10 жарамды жазба болса модель оқытылады."""
        records = make_dataset(n_good=15)
        result  = analyze_collected_data(records)
        assert result["model_trained"]      is True
        assert len(result["feature_importance"]) > 0

    def test_model_not_trained_with_few_data(self):
        """< 10 жарамды жазба → model_trained=False."""
        records = [make_record(f"td_{i}") for i in range(5)]
        result  = analyze_collected_data(records)
        assert result["model_trained"] is False

    def test_readiness_score_increases_with_data(self):
        """Жазба саны өскен сайын readiness_score өседі."""
        r_small = analyze_collected_data(make_dataset(n_good=10))
        r_large = analyze_collected_data(make_dataset(n_good=50))
        assert r_large["readiness_score"] > r_small["readiness_score"]

    def test_gold_label_count(self):
        """Gold label саны дұрыс есептеледі."""
        records = [
            make_record("td_1", label_type="gold_label"),
            make_record("td_2", label_type="gold_label"),
            make_record("td_3", label_type="secondary_label"),
        ]
        result = analyze_collected_data(records)
        assert result["gold_label_count"] == 2
        assert round(result["gold_label_pct"]) == 67

    def test_generated_at_present(self):
        """generated_at ISO format болады."""
        result = analyze_collected_data([make_record("td_1")])
        assert "T" in result["generated_at"]

    def test_records_needed_decreases(self):
        """records_needed_for_v3 жазба санымен кемиді."""
        r10  = analyze_collected_data(make_dataset(n_good=10))
        r50  = analyze_collected_data(make_dataset(n_good=50))
        assert r10["records_needed_for_v3"] > r50["records_needed_for_v3"]

    def test_full_dataset_no_crash(self):
        """Толық датасет (15 + 2 miss + 1 corrupt + 1 empty) crash болмайды."""
        records = make_dataset(n_good=15, n_missing_ef=2, n_corrupted=1, n_empty=1)
        result  = analyze_collected_data(records)
        assert result["total_records"]    == 19
        assert result["valid_records"]    == 17  # 15 + 2 miss (corrupt емес)
        assert result["corrupted_records"] == 2   # corrupt + empty


# ═════════════════════════════════════════════════════════════════
# 7. EDGE CASES
# ═════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_all_corrupted_no_crash(self):
        """Барлығы corrupted болса crash болмайды."""
        records = [make_corrupted_record(f"td_c_{i}") for i in range(5)]
        result  = analyze_collected_data(records)
        assert result["valid_records"]  == 0
        assert result["model_trained"]  is False

    def test_string_json_input_features(self):
        """input_features string болса → json.loads() жасалады."""
        import json
        rec = make_record("td_str")
        rec["input_features"]  = json.dumps(rec["input_features"])
        rec["doctor_decision"] = json.dumps(rec["doctor_decision"])
        result = analyze_collected_data([rec])
        assert result["valid_records"] == 1

    def test_no_value_error_nan_in_model(self):
        """
        КРИТИКАЛЫҚ ТЕСТ:
        NaN мәндері болса да RandomForest crash болмайды.
        ValueError: Input contains NaN — болмауы керек.
        """
        records = make_dataset(n_good=15, n_missing_ef=5)
        # Бұл жол crash болмауы тиіс
        result  = analyze_collected_data(records)
        assert "model_warning" in result  # warning болуы мүмкін, crash емес

    def test_single_record(self):
        """1 жазба — crash жоқ, есеп қайтарылады."""
        result = analyze_collected_data([make_record("td_1")])
        assert result["total_records"] == 1
        assert result["model_trained"] is False  # <10 жазба

    def test_very_large_dataset_performance(self):
        """500 жазба — crash жоқ, орынды уақытта аяқталады."""
        import time
        records = make_dataset(n_good=500)
        t0     = time.time()
        result  = analyze_collected_data(records, top_n_features=5)
        elapsed = time.time() - t0
        records = make_dataset(n_good=500, n_missing_ef=0, n_corrupted=0, n_empty=0)
        assert elapsed < 30.0  # 30 секундтан асса — тым баяу