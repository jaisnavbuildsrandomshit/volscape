"""End-to-end research pipeline: CSV bytes -> validated chain -> IV -> fits -> diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from . import diagnostics as dg
from .data import ValidationResult, ValidationSettings, load_and_validate
from .iv import IVSettings, solve_chain_iv
from .surface import FitSettings, SurfaceModel, build_baseline, fit_svi_surface, select_observations


@dataclass
class PipelineSettings:
    validation: ValidationSettings = field(default_factory=ValidationSettings)
    iv: IVSettings = field(default_factory=IVSettings)
    fit: FitSettings = field(default_factory=FitSettings)
    min_vega_volpt: float = 0.005
    extrapolate_k: bool = False

    def to_dict(self):
        return dict(validation=self.validation.to_dict(), iv=self.iv.to_dict(), fit=self.fit.to_dict(),
                    min_vega_volpt=self.min_vega_volpt, extrapolate_k=self.extrapolate_k)


@dataclass
class PipelineResult:
    validation: ValidationResult
    settings: PipelineSettings
    chain: pd.DataFrame | None = None
    obs: pd.DataFrame | None = None
    fits: list = field(default_factory=list)
    svi: SurfaceModel | None = None
    baseline: SurfaceModel | None = None
    messages: list = field(default_factory=list)

    @property
    def ok(self):
        return self.validation.ok and self.chain is not None

    def model(self, method: str) -> SurfaceModel:
        return self.svi if method == "svi" else self.baseline

    def diagnostics(self, method: str) -> dict:
        m = self.model(method)
        used = set(zip(self.obs.expiry, self.obs.strike, self.obs.option_type))
        recon = dg.reconstruction(self.chain, m, used)
        per_exp, agg = dg.error_summary(recon, self.chain, "used")
        _, agg_all = dg.error_summary(recon, self.chain, "all")
        cps, dens = dg.call_price_checks(m, self.chain)
        return dict(recon=recon, per_expiry=per_exp, aggregate=agg, aggregate_all=agg_all,
                    calendar=dg.calendar_check(m), call_checks=cps, density=dens,
                    butterfly=dg.butterfly_svi(self.fits) if method == "svi" else pd.DataFrame(),
                    headline=dg.headline_metrics(m, agg, len(self.chain), self.validation.n_input))


def run_pipeline(content: bytes, source_label: str, is_synthetic: bool,
                 settings: PipelineSettings | None = None) -> PipelineResult:
    settings = settings or PipelineSettings()
    val = load_and_validate(content, source_label, is_synthetic, settings.validation)
    res = PipelineResult(val, settings)
    if not val.ok:
        return res
    chain = solve_chain_iv(val.accepted, settings.iv, settings.min_vega_volpt)
    obs = select_observations(chain, settings.fit)
    fits = fit_svi_surface(obs, settings.fit)
    res.chain, res.obs, res.fits = chain, obs, fits
    res.svi = SurfaceModel("svi", fits, settings.extrapolate_k)
    res.baseline = SurfaceModel("baseline", build_baseline(obs), settings.extrapolate_k)
    n_ok = sum(f.usable for f in fits)
    if n_ok < 2:
        res.messages.append(f"SVI surface unavailable: {n_ok} usable slice(s); at least 2 distinct "
                            "maturities with accepted fits are required.")
    if not res.baseline.available:
        res.messages.append("Baseline surface unavailable: fewer than 2 maturities with >= 3 fit points.")
    return res
