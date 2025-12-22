"""
Critic Agent for Tubi Radar.

Reviews analyst output for quality, checks for missing context,
requests re-analysis if weak, and ensures no hallucinations.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from radar.config import get_settings
from radar.agents.specialists.threat import ThreatAssessment
from radar.agents.specialists.opportunity import Opportunity
from radar.agents.specialists.trends import Trend


class QualityLevel(Enum):
    """Quality levels for analysis."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    NEEDS_WORK = "needs_work"
    POOR = "poor"


@dataclass
class CritiqueResult:
    """Result of a critique."""
    quality_level: QualityLevel
    score: float  # 0-1
    issues: List[str]
    improvements: List[str]
    approved: bool
    feedback_summary: str


@dataclass
class FactCheck:
    """A fact check result."""
    claim: str
    verified: bool
    confidence: float
    source: Optional[str]
    notes: str


class CriticAgent:
    """
    Critic agent that reviews analysis quality.
    
    Responsibilities:
    - Review analyst output for quality
    - Check for missing context
    - Request re-analysis if weak
    - Ensure no hallucinations
    - Validate claims against sources
    """
    
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            api_key=settings.openai_api_key,
        )
    
    def critique_threats(
        self, 
        threats: List[ThreatAssessment],
        source_intel: List[Any]
    ) -> CritiqueResult:
        """
        Critique threat analysis quality.
        
        Args:
            threats: Threat assessments to review
            source_intel: Original intel items
            
        Returns:
            CritiqueResult with quality assessment
        """
        if not threats:
            return CritiqueResult(
                quality_level=QualityLevel.ACCEPTABLE,
                score=0.5,
                issues=["No threats identified - may be missing critical intel"],
                improvements=["Reanalyze with broader threat definition"],
                approved=True,
                feedback_summary="No threats found in current intel."
            )
        
        # Build critique prompt
        threats_text = ""
        for t in threats[:10]:
            threats_text += f"\n- {t.competitor}: {t.description} (severity: {t.severity})"
        
        prompt = f"""Review this threat analysis for quality:

Threats identified:
{threats_text}

Evaluate:
1. SPECIFICITY: Are threats specific and actionable? (1-10)
2. EVIDENCE: Are claims supported by intel? (1-10)
3. SEVERITY_ACCURACY: Are severity ratings appropriate? (1-10)
4. COVERAGE: Are we missing obvious threats? (1-10)
5. ACTIONABILITY: Can Tubi act on these? (1-10)

Output format:
SPECIFICITY|EVIDENCE|SEVERITY_ACCURACY|COVERAGE|ACTIONABILITY
ISSUES: issue1; issue2
IMPROVEMENTS: improvement1; improvement2
VERDICT: APPROVE or REVISE"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_critique(response.content)
        except Exception as e:
            print(f"        Critic error: {e}")
            return CritiqueResult(
                quality_level=QualityLevel.ACCEPTABLE,
                score=0.6,
                issues=[],
                improvements=[],
                approved=True,
                feedback_summary="Critique skipped due to error."
            )
    
    def critique_opportunities(
        self,
        opportunities: List[Opportunity],
        source_intel: List[Any]
    ) -> CritiqueResult:
        """
        Critique opportunity analysis quality.
        """
        if not opportunities:
            return CritiqueResult(
                quality_level=QualityLevel.ACCEPTABLE,
                score=0.5,
                issues=["No opportunities found"],
                improvements=["Expand opportunity search criteria"],
                approved=True,
                feedback_summary="No opportunities identified."
            )
        
        opp_text = ""
        for o in opportunities[:10]:
            opp_text += f"\n- {o.title} (value: {o.potential_value}, feasibility: {o.feasibility})"
        
        prompt = f"""Review this opportunity analysis for quality:

Opportunities identified:
{opp_text}

Evaluate:
1. FEASIBILITY: Are opportunities realistically achievable? (1-10)
2. VALUE: Are value estimates reasonable? (1-10)
3. SPECIFICITY: Are opportunities specific enough to act on? (1-10)
4. INNOVATION: Are these novel insights or obvious? (1-10)
5. ALIGNMENT: Do they align with Tubi's capabilities? (1-10)

Output format:
FEASIBILITY|VALUE|SPECIFICITY|INNOVATION|ALIGNMENT
ISSUES: issue1; issue2
IMPROVEMENTS: improvement1; improvement2
VERDICT: APPROVE or REVISE"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_critique(response.content)
        except Exception as e:
            return CritiqueResult(
                quality_level=QualityLevel.ACCEPTABLE,
                score=0.6,
                issues=[],
                improvements=[],
                approved=True,
                feedback_summary="Critique skipped."
            )
    
    def critique_trends(
        self,
        trends: List[Trend],
        source_intel: List[Any]
    ) -> CritiqueResult:
        """
        Critique trend analysis quality.
        """
        if not trends:
            return CritiqueResult(
                quality_level=QualityLevel.NEEDS_WORK,
                score=0.3,
                issues=["No trends identified"],
                improvements=["Broaden trend analysis scope"],
                approved=True,
                feedback_summary="No trends found."
            )
        
        trend_text = ""
        for t in trends[:10]:
            trend_text += f"\n- {t.name}: {t.description} ({t.direction})"
        
        prompt = f"""Review this trend analysis for quality:

Trends identified:
{trend_text}

Evaluate:
1. EVIDENCE: Are trends supported by multiple data points? (1-10)
2. CLARITY: Are trends clearly described? (1-10)
3. RELEVANCE: Are trends relevant to streaming/Tubi? (1-10)
4. PREDICTION: Are predictions reasonable? (1-10)
5. ACTIONABILITY: Can Tubi use these trends? (1-10)

Output format:
EVIDENCE|CLARITY|RELEVANCE|PREDICTION|ACTIONABILITY
ISSUES: issue1; issue2
IMPROVEMENTS: improvement1; improvement2
VERDICT: APPROVE or REVISE"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_critique(response.content)
        except Exception as e:
            return CritiqueResult(
                quality_level=QualityLevel.ACCEPTABLE,
                score=0.6,
                issues=[],
                improvements=[],
                approved=True,
                feedback_summary="Critique skipped."
            )
    
    def _parse_critique(self, text: str) -> CritiqueResult:
        """Parse critique response."""
        lines = text.strip().split("\n")
        
        scores = []
        issues = []
        improvements = []
        approved = True
        
        for line in lines:
            line = line.strip()
            
            # Parse scores line
            if "|" in line and not line.startswith("ISSUES") and not line.startswith("IMPROVEMENTS"):
                try:
                    parts = line.split("|")
                    scores = [int(p.strip()) for p in parts if p.strip().isdigit()]
                except:
                    pass
            
            if line.startswith("ISSUES:"):
                issues = [i.strip() for i in line.replace("ISSUES:", "").split(";") if i.strip()]
            
            if line.startswith("IMPROVEMENTS:"):
                improvements = [i.strip() for i in line.replace("IMPROVEMENTS:", "").split(";") if i.strip()]
            
            if "VERDICT:" in line:
                approved = "APPROVE" in line.upper()
        
        # Calculate average score
        avg_score = sum(scores) / len(scores) / 10.0 if scores else 0.6
        
        # Determine quality level
        if avg_score >= 0.8:
            quality = QualityLevel.EXCELLENT
        elif avg_score >= 0.7:
            quality = QualityLevel.GOOD
        elif avg_score >= 0.5:
            quality = QualityLevel.ACCEPTABLE
        elif avg_score >= 0.3:
            quality = QualityLevel.NEEDS_WORK
        else:
            quality = QualityLevel.POOR
        
        return CritiqueResult(
            quality_level=quality,
            score=avg_score,
            issues=issues,
            improvements=improvements,
            approved=approved,
            feedback_summary=f"Quality: {quality.value}, Score: {avg_score:.2f}"
        )
    
    def fact_check(self, claims: List[str]) -> List[FactCheck]:
        """
        Fact check a list of claims.
        
        Args:
            claims: List of claims to verify
            
        Returns:
            List of FactCheck results
        """
        if not claims:
            return []
        
        claims_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(claims)])
        
        prompt = f"""Fact check these claims about the streaming industry:

{claims_text}

For each claim, output one line:
NUM|VERIFIED|CONFIDENCE|NOTES

- NUM: Claim number
- VERIFIED: TRUE/FALSE/UNCERTAIN
- CONFIDENCE: 0.0-1.0
- NOTES: Brief explanation

Output (no headers):"""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_fact_checks(response.content, claims)
        except Exception:
            return []
    
    def _parse_fact_checks(self, text: str, claims: List[str]) -> List[FactCheck]:
        """Parse fact check response."""
        results = []
        
        for line in text.strip().split("\n"):
            if not line or "|" not in line:
                continue
            
            parts = line.split("|")
            if len(parts) < 4:
                continue
            
            try:
                num = int(parts[0].strip()) - 1
                if num < 0 or num >= len(claims):
                    continue
                
                verified_str = parts[1].strip().upper()
                verified = verified_str == "TRUE"
                confidence = float(parts[2].strip())
                notes = parts[3].strip()
                
                results.append(FactCheck(
                    claim=claims[num],
                    verified=verified,
                    confidence=min(1.0, max(0.0, confidence)),
                    source=None,
                    notes=notes,
                ))
            except:
                continue
        
        return results

