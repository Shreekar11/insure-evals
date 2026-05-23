"""
Deterministic dosage/unit converter — the real callable tool for agentic tool-use.
No external calls; purely arithmetic so results are always reproducible.
"""
from dataclasses import dataclass


@dataclass
class DosageResult:
    query: str
    result: str
    numeric_value: float | None = None
    unit: str = ""


def convert_dosage(
    value: float,
    from_unit: str,
    to_unit: str | None = None,
    weight_kg: float | None = None,
    drug: str = "",
) -> DosageResult:
    """
    Convert between mass units (mg ↔ g ↔ mcg) and compute per-kg dosing.

    Examples
    --------
    convert_dosage(500, "mg", "g")                     → 0.5 g
    convert_dosage(15, "mg/kg", weight_kg=70)          → 1050 mg total
    convert_dosage(1, "g", "mg")                       → 1000 mg
    """
    from_unit = from_unit.strip().lower()
    to_unit = (to_unit or "").strip().lower()
    query = f"{value} {from_unit}" + (f" ({drug})" if drug else "")

    # ── mg/kg dosing ────────────────────────────────────────────────────────
    if from_unit == "mg/kg":
        if weight_kg is None:
            return DosageResult(query=query, result="Error: weight_kg required for mg/kg conversion")
        total_mg = value * weight_kg
        result_str = f"{value} mg/kg × {weight_kg} kg = {total_mg:.1f} mg"
        if drug:
            result_str = f"{drug}: {result_str}"
        return DosageResult(query=query, result=result_str, numeric_value=total_mg, unit="mg")

    # ── unit-to-unit conversions ─────────────────────────────────────────────
    CONVERSIONS: dict[tuple[str, str], float] = {
        ("mg", "g"): 1e-3,
        ("g", "mg"): 1e3,
        ("mg", "mcg"): 1e3,
        ("mcg", "mg"): 1e-3,
        ("g", "mcg"): 1e6,
        ("mcg", "g"): 1e-6,
        ("mg", "µg"): 1e3,
        ("µg", "mg"): 1e-3,
    }

    if not to_unit:
        # No conversion — just return the value as stated
        return DosageResult(query=query, result=f"{value} {from_unit}", numeric_value=value, unit=from_unit)

    key = (from_unit, to_unit)
    if key not in CONVERSIONS:
        return DosageResult(
            query=query,
            result=f"Error: unsupported conversion {from_unit} → {to_unit}. Supported: mg↔g↔mcg",
        )

    converted = value * CONVERSIONS[key]
    # Format: avoid scientific notation for typical clinical values
    if converted == int(converted):
        val_str = str(int(converted))
    else:
        val_str = f"{converted:.4g}"

    result_str = f"{value} {from_unit} = {val_str} {to_unit}"
    if drug:
        result_str = f"{drug}: {result_str}"
    return DosageResult(query=query, result=result_str, numeric_value=converted, unit=to_unit)


# ── Tool descriptor (used by agents that support OpenAI-style function calling) ──
DOSAGE_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "convert_dosage",
        "description": (
            "Convert drug dosage between units (mg, g, mcg) or compute total dose "
            "from a per-kg dose and patient weight. Use this for any dosage calculation question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "Numeric dose value"},
                "from_unit": {
                    "type": "string",
                    "description": "Source unit: mg, g, mcg, µg, or mg/kg",
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit (omit for mg/kg conversions)",
                },
                "weight_kg": {
                    "type": "number",
                    "description": "Patient weight in kg (required for mg/kg dosing)",
                },
                "drug": {"type": "string", "description": "Drug name (optional, for labelling)"},
            },
            "required": ["value", "from_unit"],
        },
    },
}
