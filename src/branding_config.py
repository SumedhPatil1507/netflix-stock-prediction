"""
White-Label Branding Configuration — Alpha Engine Pro.
Loaded from environment variables (override per client) or branding.yaml.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List
import yaml

BRANDING_YAML = os.path.join(os.path.dirname(__file__), "..", "branding.yaml")

@dataclass
class BrandingConfig:
    app_name: str = "Alpha Engine Pro"
    tagline: str = "Institutional Quant Research & Execution Platform"
    primary_color: str = "#e50914"
    secondary_color: str = "#ffd700"
    accent_color: str = "#00bcd4"
    logo_url: str = ""
    tenant_id: str = "default"
    allowed_tickers: List[str] = field(default_factory=list)
    hitl_threshold_usd: float = 10_000.0
    compliance_mode: str = "sebi"
    show_research_copilot: bool = True
    show_strategy_lab: bool = True
    show_track_record: bool = True
    show_compliance: bool = True
    footer_text: str = "Alpha Engine Pro · For institutional use only"

_BRAND_CACHE: BrandingConfig | None = None

def get_branding() -> BrandingConfig:
    global _BRAND_CACHE
    if _BRAND_CACHE is not None:
        return _BRAND_CACHE
    cfg = BrandingConfig()
    if os.path.exists(BRANDING_YAML):
        try:
            with open(BRANDING_YAML) as f:
                data = yaml.safe_load(f) or {}
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except Exception:
            pass
    env_map = {
        "APP_NAME": "app_name", "TAGLINE": "tagline",
        "PRIMARY_COLOR": "primary_color", "SECONDARY_COLOR": "secondary_color",
        "LOGO_URL": "logo_url", "TENANT_ID": "tenant_id",
        "COMPLIANCE_MODE": "compliance_mode",
        "HITL_THRESHOLD_USD": "hitl_threshold_usd",
        "ALLOWED_TICKERS": "allowed_tickers",
    }
    for env_key, attr in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            if attr == "allowed_tickers":
                setattr(cfg, attr, [t.strip() for t in val.split(",") if t.strip()])
            elif attr == "hitl_threshold_usd":
                try:
                    setattr(cfg, attr, float(val))
                except ValueError:
                    pass
            else:
                setattr(cfg, attr, val)
    _BRAND_CACHE = cfg
    return cfg

class TenantIsolation:
    @staticmethod
    def namespace(tenant_id: str, resource_name: str) -> str:
        return f"{tenant_id}/{resource_name}"

    @staticmethod
    def filter_strategies(strategy_names: List[str], tenant_id: str, registry) -> List[str]:
        result = []
        for name in strategy_names:
            try:
                cfg = registry.get(name)
                if cfg.tenant_id == tenant_id or tenant_id == "default":
                    result.append(name)
            except Exception:
                pass
        return result

    @staticmethod
    def filter_tickers(tickers: List[str], allowed: List[str]) -> List[str]:
        if not allowed:
            return tickers
        return [t for t in tickers if t in allowed]
