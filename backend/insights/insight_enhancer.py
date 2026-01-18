"""
LLM Insight Enhancer

Uses LLM to make insights more contextual, friendly, and actionable.
"""

import httpx
from typing import Optional
from core.logging_config import llm_logger as logger
from config import get_settings


class InsightEnhancer:
    """
    Enhances raw insights using LLM to make them:
    - More contextual (uses business domain)
    - More friendly (natural language)
    - More actionable (what to do about it)
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama.base_url
        self.model = self.settings.ollama.model
    
    async def enhance_insight(
        self,
        raw_statement: str,
        business_domain: str,
        insight_type: str,
        measure: str,
        dimension: Optional[str] = None,
    ) -> str:
        """
        Enhance a single insight statement using LLM.
        
        Returns enhanced statement or original if LLM fails.
        """
        logger.debug(f"Enhancing insight: {raw_statement[:50]}...")
        
        prompt = f"""You are a business analyst. Make this insight statement more friendly, specific, and actionable.

Context:
- Business Domain: {business_domain}
- Insight Type: {insight_type}
- Measure: {measure}
- Dimension: {dimension or 'N/A'}

Original Statement:
{raw_statement}

Instructions:
1. Keep it to 1-2 sentences maximum
2. Use natural business language
3. If there's an obvious action, hint at it
4. Don't add generic advice like "investigate further"
5. Keep specific numbers and percentages

Return ONLY the enhanced statement, nothing else."""

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 150,
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    enhanced = result.get("response", "").strip()
                    
                    # Validate the enhanced statement
                    if enhanced and len(enhanced) > 20 and len(enhanced) < 300:
                        logger.success(f"Enhanced: {enhanced[:50]}...")
                        return enhanced
                    
        except Exception as e:
            logger.warning(f"Failed to enhance insight: {e}")
        
        # Return original if enhancement fails
        return raw_statement
    
    async def batch_enhance(
        self,
        insights: list[dict],
        business_domain: str,
    ) -> list[dict]:
        """
        Enhance a batch of insights in parallel.
        
        Only enhances top insights to save time.
        """
        import asyncio
        
        logger.info(f"Batch enhancing top {min(5, len(insights))} insights...")
        
        # Only enhance top 5 to save time
        top_insights = insights[:5]
        
        tasks = [
            self.enhance_insight(
                raw_statement=ins.get("statement", ""),
                business_domain=business_domain,
                insight_type=ins.get("insight_type", ""),
                measure=ins.get("measure", ""),
                dimension=ins.get("dimension"),
            )
            for ins in top_insights
        ]
        
        enhanced_statements = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update insights with enhanced statements
        for i, (ins, enhanced) in enumerate(zip(top_insights, enhanced_statements)):
            if isinstance(enhanced, str) and enhanced != ins.get("statement"):
                insights[i] = {**ins, "statement": enhanced}
        
        logger.success(f"Enhanced {len([e for e in enhanced_statements if isinstance(e, str)])} insights")
        return insights


# Global instance
insight_enhancer = InsightEnhancer()
