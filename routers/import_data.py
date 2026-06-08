"""
routers/import_data.py — Excel датасетін импорттау, batch-классификация
CardioTracker ML v2.2

Архитектура: thin router + _process_excel синхронды функциясы.

Түзетулер v2.2:
  ✅ _to_patient: data_parser v2.2 parse_row() нәтижесімен синхрон
       — comorbidities (has_hypertension, has_prior_mi т.б.) қосылды
       — symptom_class (NYHA) қосылды
  ✅ ImportPatientResult: nyha_class, confidence, ef_lv өрістері қосылды
       (schemas.py v2.2-де қосылды)
  ✅ errors: List[str] клиентке ImportResponse ішінде қайтарылады
  ✅ total_patients = барлық пациенттер (сәтті + қателі)
  ✅ finally блогында temp файл МІНДЕТТІ жойылады
"""

import os
import sys
import tempfile
from typing import Dict, List, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool

from schemas import (
    ImportResponse,
    ImportPatientResult,
    PatientData,
    AlertCheckRequest,
)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
#  SYS.PATH ХЕЛПЕРІ
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_base_dir() -> str:
    """cardio-ml/ папкасын sys.path-қа қосады (бір рет)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    return base_dir


# ─────────────────────────────────────────────────────────────────────────────
#  КОНВЕРТЕР: data_parser dict → PatientData
# ─────────────────────────────────────────────────────────────────────────────

def _to_patient(p: dict) -> PatientData:
    """
    data_parser.parse_row() нәтижесін PatientData-ға айналдырады.

    Айырмашылық:
      data_parser → ecg_flags: {atrial_fibrillation: bool, ...}  (dict)
      PatientData → ecg_af: bool, ecg_tachycardia: bool, ...     (жеке өрістер)

    FIX v2.2: comorbidities + symptom_class — data_parser v2.2-де қосылды,
    енді parse_row() нәтижесінен тікелей алынады.
    """
    ecg: dict = p.get("ecg_flags") or {}

    return PatientData(
        patient_id=p.get("name") or None,

        # Кардиология
        ef=p.get("ef_lv"),
        six_min_walk=p.get("six_min_walk"),
        nt_probnp=p.get("nt_pro_bnp"),       # датасетте жоқ → None

        # Биохимия
        creatinine=p.get("creatinine"),
        hemoglobin=p.get("hemoglobin"),
        urea=p.get("urea"),
        bilirubin=p.get("bilirubin"),
        ast=None,
        alt=None,

        # Антропометрия
        height_cm=p.get("height_cm"),
        weight_kg=p.get("weight_kg"),

        # ЭКГ: dict → жеке bool өрістер
        ecg_af=bool(ecg.get("atrial_fibrillation", False)),
        ecg_tachycardia=bool(ecg.get("tachycardia", False)),
        ecg_blockade=bool(ecg.get("bundle_branch_block", False)),
        ecg_st_changes=bool(ecg.get("st_changes", False)),

        # FIX v2.2: симптомдар + NYHA — data_parser v2.2-де бар
        symptom_class=p.get("symptom_class"),
        has_symptoms=bool(p.get("has_symptoms", False)),

        # FIX v2.2: коморбидность — data_parser v2.2-де бар
        has_diabetes=bool(p.get("has_diabetes", False)),
        has_hypertension=bool(p.get("has_hypertension", False)),
        has_copd=bool(p.get("has_copd", False)),
        has_prior_mi=bool(p.get("has_prior_mi", False)),
        has_afib_history=bool(p.get("has_afib_history", False)),
        has_ckd=bool(p.get("has_ckd", False)),

        age=None,
        sex=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  СИНХРОНДЫ ӨҢДЕУ (run_in_threadpool ішінде іске қосылады)
# ─────────────────────────────────────────────────────────────────────────────

def _process_excel(tmp_path: str) -> dict:
    """
    Excel файлын оқып, барлық пациенттерді классификациялайды.
    Синхронды — Event Loop блокталмайды (run_in_threadpool).
    """
    _ensure_base_dir()

    try:
        from data_parser import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            f"data_parser модулі табылмады: {exc}. "
            "data_parser.py файлы cardio-ml/ папкасында болуы керек."
        )

    from services.risk_service import classify_risk
    from services.alert_service import check_alerts

    patients_raw = load_dataset(tmp_path)
    total_raw    = len(patients_raw)

    results:      List[ImportPatientResult] = []
    distribution: Dict[str, int]            = {}
    errors:       List[str]                 = []

    for i, p in enumerate(patients_raw):
        row_num = i + 2  # Excel: 1-жол = header

        try:
            patient_data = _to_patient(p)
            risk_result  = classify_risk(patient_data)

            alert_list = check_alerts(
                AlertCheckRequest(current=patient_data, visit_history=[])
            )

            critical_codes: List[str] = [
                a.alert_code for a in alert_list
                if a.priority in ("CRITICAL", "HIGH")
            ]

            grp = risk_result.risk_group
            distribution[grp] = distribution.get(grp, 0) + 1

            # FIX v2.2: nyha_class, confidence, ef_lv қосылды
            results.append(
                ImportPatientResult(
                    row_number=row_num,
                    patient_name=p.get("name") or None,
                    risk_group=risk_result.risk_group,
                    risk_score=risk_result.risk_score,
                    risk_color=risk_result.risk_color,
                    alert_count=len(alert_list),
                    critical_alerts=critical_codes,
                    nyha_class=p.get("symptom_class"),
                    confidence=risk_result.confidence,
                    ef_lv=p.get("ef_lv"),
                )
            )

        except Exception as exc:
            name = p.get("name") or "белгісіз"
            errors.append(f"Жол {row_num} ({name}): {str(exc)}")

    return {
        "results":              results,
        "distribution":         distribution,
        "patients_with_alerts": sum(1 for r in results if r.alert_count > 0),
        "patients_critical":    sum(1 for r in results if r.critical_alerts),
        "errors":               errors,
        "total_raw":            total_raw,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  ENDPOINT: POST /ml/import/classify
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/import/classify",
    response_model=ImportResponse,
    summary="Импорт Excel-датасета пациентов v2.2",
    description=(
        "Принимает `.xlsx` / `.xls` файл с данными пациентов ХСН.\n\n"
        "**Процесс:**\n"
        "1. Парсинг каждой строки: ФВ/ЛЖ, Тест 6 мин, Биохимия, ОАК, ЭКГ, "
        "Рост/Вес, Коморбидность, NYHA\n"
        "2. Классификация риска ACC/AHA (A / B / C / C→D / D)\n"
        "3. Проверка алертов (ALERT, COMBO)\n"
        "4. Сводная статистика по группам риска\n\n"
        "**Исправления v2.2:** коморбидность и NYHA из датасета "
        "теперь учитываются в классификации.\n\n"
        "**Ошибочные строки** не прерывают импорт — "
        "возвращаются в поле `errors`.\n\n"
        "**Ожидаемые колонки:** `Ф.И.О`, `ФВ/ЛЖ`, `Тест 6 minute`, "
        "`Лаб.Биохимия анализ.`, `ОАК`, `ЭКГ`, `Рост`, `Вес `, "
        "`основной диагноз `, `Сопутствующий диагноз `"
    ),
    tags=["Импорт данных"],
)
async def import_classify(
    file: UploadFile = File(
        ...,
        description="Excel файл (.xlsx / .xls) с данными пациентов ХСН",
    ),
) -> ImportResponse:

    # ── 1. Формат тексеру ─────────────────────────────────────────────────────
    filename = file.filename or ""
    if not filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Тек .xlsx немесе .xls форматы қолданылады.",
        )

    # ── 2. Файл мазмұнын оқу ─────────────────────────────────────────────────
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Файл бос.")

    # ── 3. Temp файл жасау ───────────────────────────────────────────────────
    suffix   = ".xlsx" if filename.lower().endswith(".xlsx") else ".xls"
    tmp_path: Optional[str] = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # ── 4. Ауыр өңдеуді бөлек ағынға шығару ────────────────────────────
        data = await run_in_threadpool(_process_excel, tmp_path)

    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Excel өңдеу қатесі: {str(exc)}",
        )
    finally:
        # Temp файл МІНДЕТТІ жойылады (қате болса да)
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # ── 5. Нәтиже валидациясы ───────────────────────────────────────────────
    if not data["results"] and not data["errors"]:
        raise HTTPException(
            status_code=422,
            detail=(
                "Датасетте өңдеуге жарамды пациент жоқ. "
                "ФВ/ЛЖ немесе Ф.И.О бағандарын тексеріңіз."
            ),
        )

    if not data["results"] and data["errors"]:
        raise HTTPException(
            status_code=422,
            detail=f"Барлық жолдарда қате кетті: {data['errors'][:3]}",
        )

    # ── 6. total_patients = сәтті + қателі ──────────────────────────────────
    return ImportResponse(
        total_patients=data["total_raw"],
        distribution=data["distribution"],
        patients_with_alerts=data["patients_with_alerts"],
        patients_critical=data["patients_critical"],
        results=data["results"],
        errors=data["errors"],
    )