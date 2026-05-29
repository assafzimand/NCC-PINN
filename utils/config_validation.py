"""Config validation for per-problem required fields.

All features that were previously global defaults are now required to be
explicitly specified in each problem section. This ensures full transparency
of what actually ran from the config file alone.
"""

from typing import Dict, Any


# Required per-problem feature keys (no global defaults allowed)
REQUIRED_PROBLEM_FEATURES = [
    'rwf',
    'fourier_features',
    'init',
    'lra',
    'adaptive_sampling',
    'grad_clip_norm',
    'expert_grad_clip_norm',
]

# Required nested keys within each feature
REQUIRED_NESTED_KEYS = {
    'fourier_features': ['enabled', 'dim', 'scale', 'periodic'],
    'init': ['hidden', 'output', 'ls_use_bias', 'spectral_norm'],
    'lra': ['enabled', 'update_every', 'alpha'],
    'adaptive_sampling': ['enabled', 'adaptive_ratio'],
}


def validate_problem_config(cfg: Dict[str, Any]) -> None:
    """Validate that the problem section has all required feature keys.

    Raises:
        ValueError: If any required keys are missing from the problem section.
    """
    problem = cfg.get('problem')
    if not problem:
        raise ValueError("Config missing 'problem' key")

    problem_cfg = cfg.get(problem)
    if not problem_cfg:
        raise ValueError(f"Config missing problem section for '{problem}'")

    missing = []
    nested_missing = []

    for key in REQUIRED_PROBLEM_FEATURES:
        if key not in problem_cfg:
            missing.append(key)
        elif key in REQUIRED_NESTED_KEYS:
            nested = problem_cfg[key]
            if not isinstance(nested, dict):
                nested_missing.append(
                    f"{key} (expected dict, got {type(nested).__name__})")
            else:
                for nested_key in REQUIRED_NESTED_KEYS[key]:
                    if nested_key not in nested:
                        nested_missing.append(f"{key}.{nested_key}")

    if missing or nested_missing:
        error_parts = []
        if missing:
            error_parts.append(f"Missing top-level keys: {missing}")
        if nested_missing:
            error_parts.append(f"Missing nested keys: {nested_missing}")

        raise ValueError(
            f"Problem '{problem}' config is missing required feature keys.\n"
            f"{chr(10).join(error_parts)}\n\n"
            f"All per-problem features must be explicitly specified.\n"
            f"Required keys: {REQUIRED_PROBLEM_FEATURES}\n"
            f"Add missing keys to '{problem}' section in experiments_plan.yaml."
        )


def get_problem_feature(
    cfg: Dict[str, Any],
    feature: str,
    nested_key: str = None
) -> Any:
    """Get a feature value from the problem config section.

    This replaces cfg.get() patterns for moved features. Raises KeyError if
    the feature is not found (no silent defaults).

    Args:
        cfg: Full config dictionary
        feature: Top-level feature name (e.g., 'rwf', 'fourier_features')
        nested_key: Optional nested key within the feature dict

    Returns:
        The feature value

    Raises:
        KeyError: If the feature or nested key is not found
    """
    problem = cfg['problem']
    problem_cfg = cfg[problem]

    if feature not in problem_cfg:
        raise KeyError(
            f"Feature '{feature}' not found in problem '{problem}' config. "
            f"This is a required per-problem field."
        )

    value = problem_cfg[feature]

    if nested_key is not None:
        if not isinstance(value, dict):
            raise KeyError(
                f"Feature '{feature}' in problem '{problem}' is not a dict, "
                f"cannot access nested key '{nested_key}'"
            )
        if nested_key not in value:
            raise KeyError(
                f"Nested key '{nested_key}' not found in "
                f"'{feature}' for problem '{problem}'"
            )
        return value[nested_key]

    return value


def merge_problem_features_to_toplevel(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Copy per-problem features to top-level for backward compatibility.

    This is a transitional helper that copies the per-problem features to
    the top-level config so existing code that reads cfg['rwf'] etc. still
    works. Should be called after validation but before trainer runs.

    Returns:
        Modified config dict with features copied to top-level
    """
    problem = cfg['problem']
    problem_cfg = cfg[problem]

    for feature in REQUIRED_PROBLEM_FEATURES:
        if feature in problem_cfg:
            cfg[feature] = problem_cfg[feature]

    return cfg
