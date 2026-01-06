"""
Prompt Templates

Optimized prompts for data analysis with Ollama.
"""


SYSTEM_PROMPT = """You are an expert data analyst assistant. Your role is to help users understand their data through insightful analysis and clear explanations.

Key guidelines:
1. Be concise but thorough in your explanations
2. Use specific numbers and statistics when available
3. Highlight actionable insights
4. Explain statistical concepts in plain language
5. Suggest follow-up analyses when relevant
6. Format responses with markdown for readability

You have access to data analysis results that are provided in the context. Base your responses on this data, not assumptions."""


QUERY_UNDERSTANDING_PROMPT = """Analyze the user's query about their data and determine:
1. What type of analysis they're asking for
2. Which columns are relevant
3. What kind of response would be most helpful

Query: {query}

Data columns available:
{columns}

Data summary:
{summary}

Respond with a structured analysis plan in JSON format:
{{
    "analysis_type": "trend|comparison|distribution|correlation|general",
    "target_columns": ["column1", "column2"],
    "suggested_visualizations": ["chart_type"],
    "follow_up_questions": ["question1"]
}}"""


INSIGHT_EXPLANATION_PROMPT = """Explain the following data insight in a clear, actionable way for a business user:

Insight: {insight_title}
Details: {insight_description}
Statistics: {metrics}

Provide:
1. A simple explanation of what this means
2. Why it matters
3. Recommended actions or next steps
4. Any caveats or limitations"""


DATA_SUMMARY_PROMPT = """Summarize the key characteristics of this dataset:

Dataset Overview:
- Rows: {row_count}
- Columns: {column_count}
- Numeric columns: {numeric_columns}
- Categorical columns: {categorical_columns}

Column Statistics:
{column_stats}

Provide a 3-5 sentence executive summary highlighting:
1. The nature of the data
2. Key patterns or notable features
3. Potential data quality concerns
4. Suggested analyses"""


CONVERSATIONAL_ANALYSIS_PROMPT = """You are analyzing a dataset for the user. Here's the context:

## Dataset Information
{data_context}

## Previous Insights
{insights_context}

## User Question
{user_query}

Provide a helpful, data-driven response. Include:
- Direct answer to their question
- Supporting statistics from the data
- Any relevant insights from the analysis
- Suggestions for deeper exploration if applicable

Be conversational but precise. Use markdown formatting."""


CHART_RECOMMENDATION_PROMPT = """Based on the user's question and data, recommend appropriate visualizations:

User Question: {query}
Data columns: {columns}
Column types: {column_types}

Recommend 1-3 visualizations that would best answer the question. For each:
1. Chart type (bar, line, scatter, histogram, box, pie, heatmap)
2. X and Y axes (if applicable)
3. Why this visualization is helpful

Respond in JSON format:
{{
    "recommendations": [
        {{
            "chart_type": "type",
            "x_column": "column or null",
            "y_column": "column or null", 
            "reason": "why this chart helps"
        }}
    ]
}}"""


ANOMALY_EXPLANATION_PROMPT = """Explain these detected anomalies to a business user:

Column: {column}
Anomaly Type: {anomaly_type}
Number of anomalies: {count}
Percentage: {percentage}%
Example values: {examples}

Normal range: {normal_range}

Provide:
1. What these anomalies represent
2. Possible causes (data quality vs real anomalies)
3. Recommended investigation steps
4. Impact if ignored"""


TREND_NARRATIVE_PROMPT = """Create a narrative description of this trend:

Column: {column}
Direction: {direction}
Strength: {strength}
R²: {r_squared}
Change: {percent_change}%
Change points: {change_points}

Write 2-3 sentences describing:
1. The overall trend pattern
2. Notable changes or inflection points
3. What this might indicate"""


CORRELATION_NARRATIVE_PROMPT = """Explain this correlation finding:

Columns: {column1} and {column2}
Correlation: {correlation}
P-value: {p_value}
Type: {correlation_type}

Explain:
1. What this correlation means practically
2. Whether it suggests causation or just association
3. How strong/weak this relationship is
4. What actions might be taken based on this finding"""


KEY_INFLUENCER_PROMPT = """Explain these key influencers of {target}:

Top Influencers:
{influencers}

Model Score: {model_score}

Explain in business terms:
1. Which factors most affect {target}
2. The direction and magnitude of each influence
3. What actions could be taken to change {target}
4. Limitations of this analysis"""


SUGGESTED_QUESTIONS = [
    "What are the main trends in my data?",
    "Are there any outliers I should investigate?",
    "Which columns are most correlated?",
    "What factors influence {target_column}?",
    "How is the data distributed?",
    "Are there any seasonal patterns?",
    "What are the key insights from this dataset?",
    "What should I focus on first?",
]


def get_suggested_questions(columns: list[str], numeric_cols: list[str]) -> list[str]:
    """Generate contextual suggested questions."""
    questions = [
        "What are the main trends in my data?",
        "Are there any outliers I should investigate?",
    ]
    
    if len(numeric_cols) >= 2:
        questions.append("Which columns are most correlated?")
        questions.append(f"What factors influence {numeric_cols[0]}?")
    
    if numeric_cols:
        questions.append(f"How is {numeric_cols[0]} distributed?")
    
    questions.extend([
        "What are the key insights from this dataset?",
        "What should I focus on first?",
    ])
    
    return questions[:6]
