from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Mapping, Tuple

import yaml


@dataclass(frozen=True)
class PersonaProfile:
    """Base behavioral priors for a customer persona."""

    visit_prob: float
    browse_prob: float
    search_prob: float
    add_to_cart_prob: float
    remove_from_cart_prob: float
    purchase_given_cart_prob: float
    purchase_given_visit_prob: float
    coupon_open_prob: float
    coupon_redeem_prob: float
    avg_order_mean: float
    avg_order_std: float
    churn_sensitivity: float
    price_sensitivity: float
    recovery_prob: float
    acquisition_weight: float


@dataclass(frozen=True)
class UpliftSegmentProfile:
    """Latent response type for treatment-effect simulation."""

    treatment_lift: float
    coupon_open_delta: float = 0.0
    coupon_redeem_delta: float = 0.0


_PERSONAS_YAML_PATH = Path(__file__).with_name("personas.yaml")


def _expected_field_names(model_cls) -> set[str]:
    return {field.name for field in fields(model_cls)}


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Persona configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}

    if not isinstance(data, dict):
        raise ValueError("Persona configuration YAML must contain a top-level mapping.")

    return data


def _require_mapping(value, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"'{name}' must be a mapping in personas.yaml.")
    return value


def _build_profiles(section: Mapping[str, object], model_cls, section_name: str):
    expected = _expected_field_names(model_cls)
    profiles = {}

    for profile_name, raw_values in section.items():
        raw_values = _require_mapping(raw_values, f"{section_name}.{profile_name}")

        raw_keys = set(raw_values.keys())
        missing = expected - raw_keys
        extra = raw_keys - expected
        if missing or extra:
            problems = []
            if missing:
                problems.append(f"missing={sorted(missing)}")
            if extra:
                problems.append(f"extra={sorted(extra)}")
            joined = ", ".join(problems)
            raise ValueError(f"Invalid fields for {section_name}.{profile_name}: {joined}")

        profiles[str(profile_name)] = model_cls(**{key: float(raw_values[key]) for key in expected})

    if not profiles:
        raise ValueError(f"'{section_name}' must define at least one profile.")

    return profiles


def _build_weight_matrix(
    weight_section: Mapping[str, object],
    personas: Mapping[str, PersonaProfile],
    uplift_segments: Mapping[str, UpliftSegmentProfile],
) -> Dict[str, Dict[str, float]]:
    persona_names = set(personas.keys())
    uplift_names = set(uplift_segments.keys())
    weight_persona_names = set(weight_section.keys())

    missing_personas = persona_names - weight_persona_names
    extra_personas = weight_persona_names - persona_names
    if missing_personas or extra_personas:
        problems = []
        if missing_personas:
            problems.append(f"missing personas={sorted(missing_personas)}")
        if extra_personas:
            problems.append(f"unknown personas={sorted(extra_personas)}")
        joined = ", ".join(problems)
        raise ValueError(f"Invalid persona_to_uplift_weights keys: {joined}")

    matrix: Dict[str, Dict[str, float]] = {}
    for persona_name, raw_weights in weight_section.items():
        raw_weights = _require_mapping(raw_weights, f"persona_to_uplift_weights.{persona_name}")
        segment_names = set(raw_weights.keys())
        missing_segments = uplift_names - segment_names
        extra_segments = segment_names - uplift_names
        if missing_segments or extra_segments:
            problems = []
            if missing_segments:
                problems.append(f"missing segments={sorted(missing_segments)}")
            if extra_segments:
                problems.append(f"unknown segments={sorted(extra_segments)}")
            joined = ", ".join(problems)
            raise ValueError(
                f"Invalid uplift weights for persona '{persona_name}': {joined}"
            )

        normalized_weights = {segment: float(raw_weights[segment]) for segment in uplift_segments.keys()}
        total = sum(normalized_weights.values())
        if total <= 0:
            raise ValueError(
                f"persona_to_uplift_weights.{persona_name} must sum to a positive value."
            )
        matrix[str(persona_name)] = normalized_weights

    return matrix


def load_persona_bundle(
    path: str | Path | None = None,
) -> Tuple[Dict[str, PersonaProfile], Dict[str, UpliftSegmentProfile], Dict[str, Dict[str, float]]]:
    """Load simulator persona definitions from YAML and validate cross-references."""

    config_path = Path(path) if path is not None else _PERSONAS_YAML_PATH
    raw = _read_yaml(config_path)

    personas = _build_profiles(
        _require_mapping(raw.get("personas"), "personas"),
        PersonaProfile,
        "personas",
    )
    uplift_segments = _build_profiles(
        _require_mapping(raw.get("uplift_segments"), "uplift_segments"),
        UpliftSegmentProfile,
        "uplift_segments",
    )
    weight_matrix = _build_weight_matrix(
        _require_mapping(raw.get("persona_to_uplift_weights"), "persona_to_uplift_weights"),
        personas,
        uplift_segments,
    )

    return personas, uplift_segments, weight_matrix


DEFAULT_PERSONAS, DEFAULT_UPLIFT_SEGMENTS, PERSONA_TO_UPLIFT_WEIGHTS = load_persona_bundle()


def get_persona_names():
    return list(DEFAULT_PERSONAS.keys())


def get_uplift_segment_names():
    return list(DEFAULT_UPLIFT_SEGMENTS.keys())
