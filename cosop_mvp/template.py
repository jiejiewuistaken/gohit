from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SectionSpec:
    key: str
    title: str
    queries: tuple[str, ...]
    guidance: str


def cosop_sections() -> list[SectionSpec]:
    # Minimal query sets for retrieval-based population (MVP).
    return [
        SectionSpec(
            key="exec_summary",
            title="Executive summary",
            queries=("overview", "strategy", "theory of change", "objectives", "financing", "risks", "lessons"),
            guidance=(
                "Summarize the key elements of the COSOP: context, lessons learned, ToC, "
                "strategic objectives, financial framework and expected key outcomes."
            ),
        ),
        SectionSpec(
            key="I.A",
            title="I. Country context — A. Socioeconomic setting",
            queries=("GDP", "growth", "inflation", "debt", "population", "unemployment", "poverty", "fragility"),
            guidance="Provide a concise overview of the country’s socioeconomic context and complete Table 1.",
        ),
        SectionSpec(
            key="I.B",
            title="I. Country context — B. Transition scenario",
            queries=("fiscal", "debt", "fiscal space", "ODA", "transition", "IMF", "World Bank", "financing"),
            guidance="Summarize the analysis for transition scenario (appendix III) and financing mix.",
        ),
        SectionSpec(
            key="I.C",
            title="I. Country context — C. Food system, agricultural and rural sector agenda",
            queries=("agriculture", "rural", "food system", "smallholder", "value chain", "infrastructure", "climate", "gender", "youth", "nutrition"),
            guidance="Describe sector performance, constraints/opportunities, and policy/institutional framework.",
        ),
        SectionSpec(
            key="II.A",
            title="II. IFAD engagement: lessons learned — A. Results achieved during the previous COSOP",
            queries=("results", "achieved", "outcomes", "impact", "completion review", "CCR", "evaluation"),
            guidance="Summarize key results achieved during previous COSOP using CCR and evaluations.",
        ),
        SectionSpec(
            key="II.B",
            title="II. IFAD engagement: lessons learned — B. Lessons from the previous COSOP and other sources",
            queries=("lessons", "constraints", "what worked", "recommendations", "CSPE", "PCR", "partners"),
            guidance="Summarize lessons learned and explain how the proposed COSOP takes them into account.",
        ),
        SectionSpec(
            key="III.A",
            title="III. Strategy — A. COSOP theory of change",
            queries=("theory of change", "ToC", "causal chain", "assumptions", "outputs", "outcomes", "impacts"),
            guidance="Lay out causal chain and assumptions; link to RMF indicators.",
        ),
        SectionSpec(
            key="III.B",
            title="III. Strategy — B. Overall goal and strategic objectives",
            queries=("SDG", "poverty", "hunger", "strategic objective", "sustainability", "scaling up", "mainstreaming"),
            guidance="Define goal/objectives; include sustainability, scaling up, and mainstreaming themes.",
        ),
        SectionSpec(
            key="III.C",
            title="III. Strategy — C. Target group and targeting strategy",
            queries=("target group", "targeting", "women", "youth", "Indigenous", "disability", "elite capture", "geographic"),
            guidance="Define target groups and targeting strategy; address inclusion and elite capture risk.",
        ),
        SectionSpec(
            key="IV",
            title="IV. IFAD interventions (A–G)",
            queries=("policy engagement", "CLPE", "institution building", "innovation", "knowledge management", "ICT", "partnership", "SSTC", "private sector"),
            guidance="Describe IFAD interventions: financing instruments, CLPE, institutions, innovation, KM, ICT4D, partnerships/SSTC.",
        ),
        SectionSpec(
            key="V",
            title="V. COSOP implementation (A–E)",
            queries=("PBAS", "BRAM", "financing", "cofinancing", "implementation", "M&E", "country office", "transparency"),
            guidance="Present investment volume/sources, resources for additional activities, transparency, management, M&E.",
        ),
        SectionSpec(
            key="VI",
            title="VI. Target group engagement",
            queries=("consultation", "stakeholder", "feedback", "grievance", "participatory", "third-party monitoring"),
            guidance="Describe stakeholder consultation and target group feedback mechanisms.",
        ),
        SectionSpec(
            key="VII",
            title="VII. Risk management",
            queries=("risk", "macroeconomic", "governance", "fragility", "procurement", "financial management", "mitigation"),
            guidance="Highlight key risks linked to objectives and summarize the ICRM (appendix XI).",
        ),
    ]

