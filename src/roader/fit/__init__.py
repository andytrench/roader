"""Stages 6-7: alignment, profile and cross-section fitting. Lane `pipeline`.

FR-AF-1..5, FR-XS-1..3. Rules enter as soft penalties (FR-AF-4).

Public API:
    fit_alignment / to_alignment     L1 line/arc/spiral chain, curvature continuous (FR-AF-1)
    fit_profile / to_profile         L2 grades and parabolic curves, robust (FR-AF-2, FR-OC-1)
    fit_edges, fit_cross_slopes      L3 edge lines and banking (FR-XS-1, FR-XS-2)
    infer_lanes                      FR-XS-3
    fit_road(seed, evidence, rules)  one road end to end with per-element provenance (FR-AF-5)
"""
