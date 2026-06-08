"""
schemas.py — CardioTracker ML, Pydantic v2 models
CardioTracker ML v2.2

Синхронизирован с:
  risk_service.py v2.2, alert_service.py v2.2,
  summary_service.py v2.2, dynamics_service.py v2.2,
  data_parser.py v2.2

Түзетулер v2.2:
  ✅ PatientData — data_parser.py parse_row() нәтижесімен толық сәйкес:
       has_hypertension, has_prior_mi, has_copd, has_afib_history, has_ckd
       symptom_class, has_symptoms — бұрын да бар, дұрыс
  ✅ VisitRecord — weight_kg ge=1 le=300 шектері қосылды (PatientData-мен сәйкес)
  ✅ ImportPatientResult — nyha_class, confidence өрістері қосылды
       (import_data роутеры batch нәтижеде кеңірек мәліметтер береді)
  ✅ AlertResponse — medium_count өрісі қосылды (MEDIUM алерттер саны)
  ✅ DynamicsResponse — error_message өрісі қосылды (analyze_dynamics exception үшін)
  ✅ Барлық модельдерде json_schema_extra examples жаңартылды
  ✅ Версия коментарийлері v2.2-ге жаңартылды
"""

from __future__ import annotations
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
#  BASE
# ─────────────────────────────────────────────────────────────────────────────

class CardioBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",            # белгісіз өрістер → тыныш өткізіп жібер
        populate_by_name=True,     # alias да, нақты ат да жұмыс істейді
        str_strip_whitespace=True, # string өрістерінен бос орындар тазаланады
    )


# =============================================================================
#  1. PATIENT DATA — негізгі кіріс объекті
# =============================================================================

class PatientData(CardioBaseModel):
    """
    Клинические данные пациента — основной входной объект.
    Синхронизирован с data_parser.parse_row() v2.2.
    """

    patient_id: Optional[str] = Field(
        default=None,
        description="ID пациента (Spring Boot-тан integer string ретінде келеді)",
        json_schema_extra={"example": "42"},
    )

    # ── Кардиология ──────────────────────────────────────────────────────────
    ef: Optional[float] = Field(
        default=None, ge=0, le=100,
        description="ФВ ЛЖ, % (норма ≥55%; датасет: 40–68%)",
        json_schema_extra={"example": 35.0},
    )
    six_min_walk: Optional[float] = Field(
        default=None, ge=0,
        description="Тест 6 мин, м (норма >550 ФК I; ТЗ: >550/426-550/150-425/<150)",
        json_schema_extra={"example": 280.0},
    )
    nt_probnp: Optional[float] = Field(
        default=None, ge=0,
        description="NT-proBNP, пг/мл (норма <125; датасетте жоқ — визитте енгізіледі)",
        json_schema_extra={"example": 2100.0},
    )

    # ── Биохимия ─────────────────────────────────────────────────────────────
    creatinine: Optional[float] = Field(
        default=None, ge=0,
        description="Креатинин, мкмоль/л (норма 60–110; датасет: 7–115)",
        json_schema_extra={"example": 115.0},
    )
    hemoglobin: Optional[float] = Field(
        default=None, ge=0,
        description="Гемоглобин, г/л (норма ≥120; <110 → ALERT-03; датасет: 79–177)",
        json_schema_extra={"example": 108.0},
    )
    urea: Optional[float] = Field(
        default=None, ge=0,
        description="Мочевина, ммоль/л (норма 2.5–8.3)",
        json_schema_extra={"example": 9.5},
    )
    bilirubin: Optional[float] = Field(
        default=None, ge=0,
        description="Билирубин, мкмоль/л (норма 3–20; >25 → застойная печень)",
        json_schema_extra={"example": 22.0},
    )
    ast: Optional[float] = Field(
        default=None, ge=0,
        description="АСТ, Ед/л (АСТ/АЛТ >2.0 → застойная гепатопатия)",
        json_schema_extra={"example": 38.0},
    )
    alt: Optional[float] = Field(
        default=None, ge=0,
        description="АЛТ, Ед/л",
        json_schema_extra={"example": 18.0},
    )

    # ── Антропометрия ────────────────────────────────────────────────────────
    height_cm: Optional[float] = Field(
        default=None, ge=50, le=250,
        description="Рост, см (датасет: 100–220)",
        json_schema_extra={"example": 168.0},
    )
    weight_kg: Optional[float] = Field(
        default=None, ge=1, le=300,
        description="Вес, кг (датасет: 30–250)",
        json_schema_extra={"example": 88.0},
    )

    # ── ЭКГ флаги ────────────────────────────────────────────────────────────
    ecg_af: bool          = Field(default=False, description="Фибрилляция предсердий (датасет: 13/134)")
    ecg_tachycardia: bool = Field(default=False, description="Тахикардия")
    ecg_blockade: bool    = Field(default=False, description="Блокада ножки пучка Гиса")
    ecg_st_changes: bool  = Field(default=False, description="Изменения ST (датасет: 18/134)")

    # ── Симптомы ─────────────────────────────────────────────────────────────
    has_symptoms: bool           = Field(default=False, description="Клинические симптомы ХСН")
    symptom_class: Optional[int] = Field(
        default=None, ge=1, le=4,
        description="NYHA ФК (1–4; датасет: ФК II=47, ФК III=70)",
        json_schema_extra={"example": 3},
    )

    # ── Демография ───────────────────────────────────────────────────────────
    age: Optional[int] = Field(
        default=None, ge=0, le=120,
        description="Возраст, лет",
        json_schema_extra={"example": 68},
    )
    sex: Optional[Literal["M", "F"]] = Field(
        default=None,
        description="Пол: M / F",
        json_schema_extra={"example": "M"},
    )

    # ── Коморбидность — data_parser.parse_comorbidities()-мен синхрон ────────
    has_diabetes: bool     = Field(default=False, description="Сахарный диабет (датасет: 14/134)")
    has_hypertension: bool = Field(default=False, description="Артериальная гипертония (датасет: 98/134)")
    has_copd: bool         = Field(default=False, description="ХОБЛ (датасет: 3/134)")
    has_prior_mi: bool     = Field(default=False, description="Инфаркт миокарда в анамнезе (датасет: 43/134)")
    has_afib_history: bool = Field(default=False, description="ФП в анамнезе (датасет: 18/134)")
    has_ckd: bool          = Field(default=False, description="Хроническая болезнь почек")


# =============================================================================
#  2. RISK CLASSIFICATION
# =============================================================================

class RiskResponse(CardioBaseModel):
    """
    Результат оценки группы риска (ACC/AHA).
    Единая модель — risk_service.py classify_risk() қайтарады.
    """

    patient_id: Optional[str] = Field(default=None)

    risk_group: str = Field(
        ...,
        description="ТЗ 5.2: норма / C / C→D / D",
        json_schema_extra={"example": "C→D"},
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Балл риска 0.0–1.0",
        json_schema_extra={"example": 0.72},
    )
    risk_color: str = Field(
        ...,
        description="green / blue / yellow / orange / red",
        json_schema_extra={"example": "orange"},
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Факторы, повлиявшие на балл",
    )
    recommendation: str = Field(
        ...,
        description="Рекомендация врачу",
        json_schema_extra={"example": "2 аптада бір бақылау. Госпитализацияны қарастыру."},
    )
    score_breakdown: List[str] = Field(
        default_factory=list,
        description="Детальный breakdown по каждому параметру (+X.XX)",
    )
    confidence: Optional[str] = Field(
        default=None,
        description="высокая / средняя / низкая / недостаточно данных",
    )
    confidence_pct: Optional[int] = Field(
        default=None, ge=0, le=100,
        description="Заполненность данных, %",
        json_schema_extra={"example": 72},
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Клинические предупреждения (парадоксы, критические значения)",
    )
    bmi: Optional[float] = Field(
        default=None, ge=0,
        description="BMI (кг/м²), автоматически рассчитывается из height_cm + weight_kg",
        json_schema_extra={"example": 24.5},
    )


# =============================================================================
#  3. VISIT RECORD
# =============================================================================

class VisitRecord(CardioBaseModel):
    """
    Запись одного визита пациента.
    Используется в: AlertCheckRequest, DynamicsRequest, SummaryRequest.
    """

    visit_date: Optional[str] = Field(
        default=None,
        description="Дата визита YYYY-MM-DD (build_day_axis() үшін қажет)",
        json_schema_extra={"example": "2024-03-15"},
    )
    ef: Optional[float] = Field(
        default=None,
        description="ФВ ЛЖ, %",
        json_schema_extra={"example": 36.0},
    )
    nt_probnp: Optional[float] = Field(
        default=None,
        description="NT-proBNP, пг/мл",
        json_schema_extra={"example": 1650.0},
    )
    six_min_walk: Optional[float] = Field(
        default=None,
        description="Тест 6 мин, м",
        json_schema_extra={"example": 310.0},
    )
    creatinine: Optional[float] = Field(
        default=None,
        description="Креатинин, мкмоль/л",
        json_schema_extra={"example": 142.0},
    )
    hemoglobin: Optional[float] = Field(
        default=None,
        description="Гемоглобин, г/л",
        json_schema_extra={"example": 112.0},
    )
    urea: Optional[float] = Field(
        default=None,
        description="Мочевина, ммоль/л",
        json_schema_extra={"example": 9.2},
    )
    bilirubin: Optional[float] = Field(
        default=None,
        description="Билирубин, мкмоль/л",
        json_schema_extra={"example": 18.0},
    )
    ast: Optional[float] = Field(
        default=None,
        description="АСТ, Ед/л",
        json_schema_extra={"example": 36.0},
    )
    alt: Optional[float] = Field(
        default=None,
        description="АЛТ, Ед/л",
        json_schema_extra={"example": 40.0},
    )
    weight_kg: Optional[float] = Field(
        default=None, ge=1, le=300,
        description="Вес, кг (TREND-04 салмақ өзгерісі үшін)",
        json_schema_extra={"example": 78.5},
    )
    symptom_class: Optional[int] = Field(
        default=None, ge=1, le=4,
        description="NYHA ФК",
        json_schema_extra={"example": 3},
    )


# =============================================================================
#  4. ALERTS
# =============================================================================

class AlertCheckRequest(CardioBaseModel):
    """Запрос на проверку алертов."""

    current: PatientData = Field(..., description="Данные текущего визита")
    visit_history: Optional[List[VisitRecord]] = Field(
        default_factory=list,
        description="История визитов (от старого к новому). None → [] ретінде өңделеді.",
    )


class SingleAlert(CardioBaseModel):
    """Один сработавший алерт."""

    alert_code: str = Field(
        ...,
        description="Код: ALERT-01..06, COMBO-01/05, TREND-01..04",
        json_schema_extra={"example": "ALERT-01"},
    )
    priority: str = Field(
        ...,
        description="CRITICAL / HIGH / MEDIUM / LOW",
        json_schema_extra={"example": "CRITICAL"},
    )
    message: str              = Field(..., description="Описание алерта на казахском")
    parameter: Optional[str]  = Field(default=None, description="Затронутый параметр")
    value: Optional[float]    = Field(default=None, description="Текущее значение")
    threshold: Optional[float]= Field(default=None, description="Порог срабатывания")


class AlertResponse(CardioBaseModel):
    """
    Ответ эндпоинта /ml/check-alerts.
    FIX v2.2: medium_count өрісі қосылды.
    """

    patient_id: Optional[str]      = Field(default=None)
    total_alerts: int               = Field(..., description="Всего алертов")
    critical_count: int             = Field(..., description="CRITICAL алертов")
    high_count: int                 = Field(..., description="HIGH алертов")
    medium_count: int               = Field(default=0, description="MEDIUM алертов")
    alerts: List[SingleAlert]       = Field(default_factory=list)
    requires_immediate_action: bool = Field(..., description="True если есть CRITICAL")


class NtProBnpParadoxRequest(CardioBaseModel):
    """Запрос на проверку парадокса NT-proBNP."""

    patient_id: Optional[str]  = Field(default=None)
    nt_probnp: Optional[float] = Field(default=None, ge=0, description="NT-proBNP, пг/мл")
    ef: Optional[float]        = Field(default=None, ge=0, le=100, description="ФВ ЛЖ, %")
    has_symptoms: bool         = Field(default=False, description="Клинические симптомы ХСН")


class NtProBnpParadoxResponse(CardioBaseModel):
    """Ответ проверки парадокса NT-proBNP."""

    patient_id: Optional[str]       = Field(default=None)
    paradox_detected: bool           = Field(..., description="True если парадокс выявлен")
    message: str                     = Field(...)
    recommendation: str              = Field(...)
    nt_probnp_value: Optional[float] = Field(default=None)
    ef_value: Optional[float]        = Field(default=None)


# =============================================================================
#  5. AI SUMMARY
# =============================================================================

class SummaryRequest(CardioBaseModel):
    """Запрос на генерацию клинического заключения."""

    patient_data: PatientData                  = Field(...)
    visit_history: Optional[List[VisitRecord]] = Field(
        default_factory=list,
        description="История визитов. [] → бірінші визит режимі.",
    )
    include_recommendations: bool = Field(
        default=True,
        description="Ұсыныстар бөлімін қосу керек пе",
    )


class SummaryResponse(CardioBaseModel):
    """Ответ — текстовое клиническое заключение."""

    patient_id: Optional[str] = Field(default=None)
    summary_text: str          = Field(..., description="Полный текст заключения на казахском")
    risk_group: str            = Field(..., description="Итоговая группа риска")
    alert_count: int           = Field(..., description="Количество активных алертов")
    generated_at: str          = Field(..., description="Дата и время генерации (ISO 8601)")


# =============================================================================
#  6. DYNAMICS
# =============================================================================

class DynamicsRequest(CardioBaseModel):
    """Запрос анализа динамики по визитам."""

    patient_id: Optional[str] = Field(default=None)
    visits: List[VisitRecord] = Field(
        ...,
        description="Минимум 2 визита, от старого к новому",
    )

    @field_validator("visits")
    @classmethod
    def visits_must_have_at_least_two(cls, v: List[VisitRecord]) -> List[VisitRecord]:
        """
        Pydantic v2-де List үшін min_length жұмыс істемейді.
        Осы validator арқылы тексеріледі.
        """
        if len(v) < 2:
            raise ValueError(
                f"Динамика анализі үшін минимум 2 визит керек. "
                f"Берілді: {len(v)} визит."
            )
        return v


class TrendItem(CardioBaseModel):
    """Тренд одного параметра за все визиты."""

    parameter: str        = Field(
        ...,
        description="Название параметра (PARAM_CONFIGS label-мен сәйкес)",
        json_schema_extra={"example": "ФВ ЛЖ"},
    )
    direction: str        = Field(
        ...,
        description="улучшение / стабильно / ухудшение / значительное ухудшение",
    )
    change_percent: float = Field(..., description="Изменение, % (первый → последний визит)")
    first_value: float    = Field(..., description="Значение на первом визите")
    last_value: float     = Field(..., description="Значение на последнем визите")
    is_critical: bool     = Field(default=False, description="True если изменение критическое")
    severity: Optional[str]               = Field(
        default=None,
        description="normal / mild / moderate / severe",
    )
    slope: Optional[float]                = Field(
        default=None,
        description="Наклон регрессии (linear_regression slope)",
    )
    velocity_per_month: Optional[float]   = Field(
        default=None,
        description="Изменение в месяц (только если visit_date заполнен)",
    )
    forecast_next: Optional[float]        = Field(
        default=None,
        description="Прогноз на следующий визит",
    )
    values_history: Optional[List[float]] = Field(
        default=None,
        description="Все значения по визитам (для графика на фронте)",
    )


class DynamicsResponse(CardioBaseModel):
    """Результат анализа динамики."""

    patient_id: Optional[str]                 = Field(default=None)
    visits_analyzed: int                       = Field(..., description="Количество визитов в анализе")
    overall_trend: str                         = Field(..., description="Общий тренд: ухудшение / стабильно / улучшение")
    trends: List[TrendItem]                    = Field(default_factory=list)
    risk_progression: str                      = Field(..., description="Изменение группы риска")
    next_visit_recommendation: str             = Field(..., description="Рекомендация по следующему визиту")
    avg_interval_days: Optional[int]           = Field(
        default=None,
        description="Средний интервал между визитами, дней (только если visit_date заполнен)",
    )
    visit_dates: Optional[List[Optional[str]]] = Field(
        default=None,
        description="Даты визитов (для графика на фронте)",
    )
    # FIX v2.2: error_message — analyze_dynamics exception болса роутер осыны толтырады
    error_message: Optional[str]               = Field(
        default=None,
        description="Қате болса — себебі (напр. 'Минимум 2 визит керек')",
    )


# =============================================================================
#  7. IMPORT DATA
# =============================================================================

class ImportPatientResult(CardioBaseModel):
    """
    Результат классификации одного пациента из Excel.
    FIX v2.2: nyha_class, confidence өрістері қосылды.
    """

    row_number: int             = Field(..., description="Номер строки в файле")
    patient_name: Optional[str] = Field(default=None, description="ФИО из датасета")
    risk_group: str             = Field(...)
    risk_score: float           = Field(..., ge=0.0, le=1.0)
    risk_color: str             = Field(...)
    alert_count: int            = Field(...)
    critical_alerts: List[str]  = Field(
        default_factory=list,
        description="Коды CRITICAL/HIGH алертов",
    )
    # FIX v2.2: қосымша өрістер
    nyha_class: Optional[int]   = Field(
        default=None,
        description="NYHA ФК датасеттен (1–4)",
    )
    confidence: Optional[str]   = Field(
        default=None,
        description="Классификация сенімділігі: высокая / средняя / низкая",
    )
    ef_lv: Optional[float]      = Field(
        default=None,
        description="ФВ/ЛЖ датасеттен, %",
    )


class ImportResponse(CardioBaseModel):
    """Итоговый ответ импорта Excel-датасета."""

    total_patients: int              = Field(..., description="Всего обработано пациентов")
    distribution: Dict[str, int]     = Field(
        default_factory=dict,
        description="Распределение по группам риска",
    )
    patients_with_alerts: int        = Field(..., description="Пациентов с хотя бы одним алертом")
    patients_critical: int           = Field(..., description="Пациентов с CRITICAL алертом")
    errors: List[str]                = Field(default_factory=list)
    results: List[ImportPatientResult] = Field(default_factory=list)