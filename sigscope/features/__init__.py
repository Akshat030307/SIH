"""Handcrafted classification features (CLAUDE.md §5.2).

Higher-order cumulants (C20, C21, C40, C41, C42, C63), instantaneous amplitude/phase/
frequency statistics, and spectral features. Computed on the isolated, power-normalised
burst. About 30 features, all cheap and all explainable to a judge. Not implemented yet
(Phase 6, §8).
"""

from __future__ import annotations
