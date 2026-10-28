"""RAG + LLM explanation of a flagged subgraph.

Providers (GRAPHGUARD_LLM): template (offline, deterministic) | anthropic | openai.
`grounded=False` skips retrieval - used as the ungrounded ablation for the groundedness evaluation.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from .anomaly import Evidence
from .config import CFG, Config
from .rag import Chunk, Retriever

SYSTEM = (
    "You are a financial-crime analyst assistant. Explain why a cryptocurrency transaction was flagged by a graph "
    "neural network. Use ONLY (a) the structural evidence JSON and (b) the numbered reference passages. Cite passages "
    "as [ID] after each claim they support. Do not invent amounts, entities, dates or identities. State that the "
    "indicators are grounds for review, not proof, and end with one concrete recommended analyst action."
)

TYPE_LABEL = {"fan_out": "fan-out (dispersal)", "fund_cycling": "rapid fund cycling", "community_bridging": "community bridging"}


def build_prompt(ev: Evidence, passages: list[tuple[Chunk, float]]) -> str:
    refs = "\n".join(f"[{c.cid}] ({c.source}) {c.text}" for c, _ in passages) or "(none - no reference passages supplied)"
    facts = json.dumps({"tx_id": ev.tx_id, "risk_score": round(ev.risk, 3), "detected_patterns": ev.types, **ev.facts}, indent=1)
    return f"STRUCTURAL EVIDENCE:\n{facts}\n\nREFERENCE PASSAGES:\n{refs}\n\nWrite a 120-180 word explanation for the compliance analyst."


def _template(ev: Evidence, passages: list[tuple[Chunk, float]]) -> str:
    f = ev.facts
    cite = lambda t: next((f"[{c.cid}]" for c, _ in passages if t.lower() in c.cid.lower()), "")  # noqa: E731
    lines = [f"Transaction {ev.tx_id} received a GNN risk score of {ev.risk:.2f}."]
    if not ev.types:
        lines.append("No fan-out, cycling or community-bridging pattern was detected; the score rests on learned "
                     "feature and neighbourhood signals and needs manual review.")
    if "fan_out" in ev.types:
        lines.append(f"Fan-out: it sends to {f['out_degree']} counterparties against {f['in_degree']} inputs "
                     f"(ratio {f['fanout_ratio']}), consistent with dispersal/layering behaviour {cite('fanout')}{cite('peel')}.")
    if "fund_cycling" in ev.types:
        lines.append(f"Fund cycling: it lies on a directed cycle (component of {f['cycle_component_size']} transactions, "
                     f"{f['reciprocal_edges']} reciprocal edges), a round-tripping/wash-trading signature {cite('cycle')}{cite('wash')}.")
    if "community_bridging" in ev.types:
        lines.append(f"Community bridging: {f['cross_community_neighbours']} of its neighbours ({f['bridge_ratio']:.0%}) belong to "
                     f"different communities, an unusual intermediary position {cite('bridge')}.")
    if f["neighbour_illicit_labels"]:
        lines.append(f"{f['neighbour_illicit_labels']} labelled-illicit transaction(s) sit in its {f['subgraph_size']}-node neighbourhood.")
    lines.append("These are indicators warranting review, not proof of wrongdoing; on-chain evidence cannot establish "
                 "intent or identity [REG-LIMITS][REG-SAR]." if any(c.cid == "REG-LIMITS" for c, _ in passages) else
                 "These are indicators warranting review, not proof of wrongdoing.")
    lines.append("Recommended action: escalate for analyst review of the neighbourhood and trace onward flows before any SAR decision.")
    return " ".join(lines)


def _anthropic(prompt: str, cfg: Config) -> str:
    import anthropic  # reads ANTHROPIC_API_KEY
    r = anthropic.Anthropic().messages.create(model=cfg.llm_model, max_tokens=500, system=SYSTEM,
                                              messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in r.content if b.type == "text")


def _openai(prompt: str, cfg: Config) -> str:
    from openai import OpenAI
    r = OpenAI().chat.completions.create(model=os.environ.get("GRAPHGUARD_LLM_MODEL", "gpt-4o-mini"), max_tokens=500,
                                         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    return r.choices[0].message.content or ""


class Explainer:
    def __init__(self, cfg: Config = CFG, retriever: Retriever | None = None):
        self.cfg = cfg
        self.retriever = retriever or Retriever(cfg)

    def explain(self, ev: Evidence, grounded: bool = True, k: int = 4) -> dict:
        passages = self.retriever.for_types(ev.types, k) if grounded else []
        if grounded:  # always include the caveat passages
            have = {c.cid for c, _ in passages}
            passages += [(c, 0.0) for c in self.retriever.chunks if c.cid in ("REG-LIMITS", "REG-SAR") and c.cid not in have]
        prompt = build_prompt(ev, passages)
        provider = self.cfg.llm_provider
        try:
            text = {"anthropic": _anthropic, "openai": _openai}[provider](prompt, self.cfg) if provider in ("anthropic", "openai") \
                else _template(ev, passages)
        except Exception as e:  # noqa: BLE001 - never lose a flag because the LLM is down
            text = _template(ev, passages) + f" (LLM provider '{provider}' failed: {type(e).__name__}; template explanation used.)"
            provider = "template-fallback"
        return {"tx_id": ev.tx_id, "risk": ev.risk, "patterns": ev.types, "primary_pattern": ev.primary, "explanation": text,
                "provider": provider, "grounded": grounded,
                "citations": [{"id": c.cid, "source": c.source, "score": round(s, 3)} for c, s in passages],
                "evidence": asdict(ev)}
