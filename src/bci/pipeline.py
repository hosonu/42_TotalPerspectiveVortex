"""Build the default sklearn ``Pipeline`` for motor-imagery decoding."""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from bci.csp import CustomCSP
from bci.features import LogVarianceTransformer
from bci.classifier import CustomLogisticRegression


def make_motor_imagery_pipeline(
    *,
    n_csp_components: int = 6,
    logvar_filters_per_end: int = 3,
    csp: CustomCSP | None = None,
    use_bonus: bool = False,
) -> Pipeline:
    """Build ``CSP → log-variance → LDA``.

    With defaults, log-variance uses the first three and last three CSP filters
    (six log-var features), which matches ``n_csp_components == 6``.
    """
    if csp is None:
        csp = CustomCSP(n_components=n_csp_components)

    clf = CustomLogisticRegression() if use_bonus else LinearDiscriminantAnalysis()
    return Pipeline(
        [
            ("csp", csp),
            (
                "logvar",
                LogVarianceTransformer(filters_per_end=logvar_filters_per_end),
            ),
            ("clf", clf),
        ]
    )
