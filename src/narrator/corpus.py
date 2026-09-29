"""
Market corpus containing financial news and earnings call transcripts for RAG vector store.
Provides high-fidelity transcripts and news for NFLX and other popular tickers.
"""
from __future__ import annotations
from typing import List, Dict

SAMPLE_CORPUS: List[Dict[str, str]] = [
    # ── NFLX Earnings Call Transcripts ──────────────────────────────────────────
    {
        "id": "nflx_ec_q4_2024_01",
        "ticker": "NFLX",
        "source_type": "earnings_transcript",
        "title": "Netflix Q4 2024 Earnings Call - Management Remarks on Monetization & Margin",
        "date": "2025-01-23",
        "text": (
            "Netflix Management (Co-CEO Greg Peters): 'In Q4 2024, our operating margin expanded to 27.2%, "
            "up 450 basis points year-over-year, driven by disciplined content spending and strong ad-tier growth. "
            "Our advertising membership increased by 70% quarter-over-quarter, accounting for over 50% of new sign-ups "
            "in ad-tier countries. We expect full-year 2025 operating margins to reach 28% to 29%. Average Revenue per "
            "Member (ARM) grew 5% on a FX-neutral basis due to strategic price adjustments in EMEA and Latin America. "
            "Free cash flow reached $7.1 billion for the full year 2024, enabling $6.5 billion in share repurchases.'"
        ),
    },
    {
        "id": "nflx_ec_q4_2024_02",
        "ticker": "NFLX",
        "source_type": "earnings_transcript",
        "title": "Netflix Q4 2024 Earnings Call - Live Events & Content Slate Engagement",
        "date": "2025-01-23",
        "text": (
            "Netflix Management (Co-CEO Ted Sarandos): 'Engagement remains exceptionally strong with over 2 hours daily "
            "per member on average. The live streaming of the NFL Christmas Day games generated over 65 million viewers globally, "
            "demonstrating our technical capability to deliver concurrent mega-events at scale. Our 2025 content slate features "
            "the return of Stranger Things Season 5, Wednesday Season 2, and Squid Game Season 3, creating an unprecedented "
            "catalyst for retention and organic top-of-funnel customer acquisition.'"
        ),
    },
    {
        "id": "nflx_ec_q3_2024_01",
        "ticker": "NFLX",
        "source_type": "earnings_transcript",
        "title": "Netflix Q3 2024 Earnings Call - Paid Sharing & Subscriber Momentum",
        "date": "2024-10-17",
        "text": (
            "Netflix CFO Spencer Neumann: 'Paid sharing continues to deliver healthy conversion of borrower households into paying members. "
            "We added 5.1 million net paid additions in Q3, exceeding consensus expectations of 4.5 million. Total global paid memberships "
            "reached 282.7 million. We are forecasting Q4 revenue of $10.13 billion, representing 14.7% YoY growth. Content amortization "
            "is expected to scale efficiently at approximately $17 billion annually, generating sustained operating leverage.'"
        ),
    },
    {
        "id": "nflx_ec_q2_2024_01",
        "ticker": "NFLX",
        "source_type": "earnings_transcript",
        "title": "Netflix Q2 2024 Earnings Call - Ad-Tech Platform & Strategic Partnerships",
        "date": "2024-07-18",
        "text": (
            "Netflix Leadership: 'We are launching our in-house ad-tech platform across international markets in 2025, moving away "
            "from legacy third-party dependencies. This provides programmatic advertisers with deep targeting capabilities, proprietary "
            "measurement metrics, and direct bidding. Early programmatic pilots with The Trade Desk and Google Display & Video 360 "
            "are yielding significant CPM premiums over traditional broadcast television benchmarks.'"
        ),
    },
    # ── NFLX Financial News & Analyst Commentary ───────────────────────────────
    {
        "id": "nflx_news_2025_02_01",
        "ticker": "NFLX",
        "source_type": "financial_news",
        "title": "Wall Street Research Note: Morgan Stanley Raises NFLX Target on Ad Tier & Live Sports Scale",
        "date": "2025-02-14",
        "text": (
            "Morgan Stanley Equity Research reiterated an Overweight rating on Netflix (NASDAQ: NFLX) and raised its price target, "
            "citing rapid monetization of ad-supported tiers and expanding live broadcast inventory (WWE Raw, NFL Christmas Games). "
            "Analysts highlight that operating leverage and recurring free cash flow yield support multiple re-rating, projecting "
            "2025 EPS growth in excess of 24% year-over-year."
        ),
    },
    {
        "id": "nflx_news_2025_01_15",
        "ticker": "NFLX",
        "source_type": "financial_news",
        "title": "Bloomberg Markets: Streaming Industry Consolidation & Pricing Power Resilience",
        "date": "2025-01-15",
        "text": (
            "Streaming media sector analysis indicates Netflix maintains the lowest churn rate in the industry at 1.8%, compared to "
            "an industry average of 4.2%. Consumer pricing sensitivity studies demonstrate high willingness-to-pay elasticity, allowing "
            "Netflix to implement price revisions without triggering subscriber cancellations, preserving robust unit economics amidst "
            "elevated cost of capital."
        ),
    },
    {
        "id": "nflx_news_2025_02_20",
        "ticker": "NFLX",
        "source_type": "financial_news",
        "title": "Reuters: Macro Ad-Market Outlook & Foreign Exchange Headwinds",
        "date": "2025-02-20",
        "text": (
            "Macroeconomic surveys highlight steady enterprise digital ad spend growth (+12% YoY), benefiting scaled connected-TV "
            "platforms. However, foreign exchange volatility—specifically US dollar strength against the Euro and Japanese Yen—represents "
            "a potential top-line headwind of approximately 150-200 basis points for multinational subscription businesses."
        ),
    },
    # ── AAPL Data ─────────────────────────────────────────────────────────────
    {
        "id": "aapl_ec_q1_2025_01",
        "ticker": "AAPL",
        "source_type": "earnings_transcript",
        "title": "Apple Q1 2025 Earnings Call - Services Revenue & Apple Intelligence Rollout",
        "date": "2025-01-30",
        "text": (
            "Apple CEO Tim Cook: 'Services revenue reached an all-time record of $26.3 billion, up 14% year-over-year with over "
            "1 billion paid subscriptions across platforms. Apple Intelligence adoption is accelerating upgrade cycles across iPhone 16 "
            "Pro and enterprise workflows, driving gross margins to 46.2%.'"
        ),
    },
    {
        "id": "aapl_news_2025_02_10",
        "ticker": "AAPL",
        "source_type": "financial_news",
        "title": "Goldman Sachs: Apple Ecosystem Expansion & Recurring Cash Flow",
        "date": "2025-02-10",
        "text": (
            "Goldman Sachs maintained Buy rating on Apple (AAPL), noting robust Services gross margin expansion (>74%) and substantial "
            "capital return program via buybacks ($110B authorization) creating sustained downside protection against cyclical consumer hardware demand."
        ),
    },
    # ── TSLA Data ─────────────────────────────────────────────────────────────
    {
        "id": "tsla_ec_q4_2024_01",
        "ticker": "TSLA",
        "source_type": "earnings_transcript",
        "title": "Tesla Q4 2024 Earnings Call - Autonomy, Robotaxi & Energy Storage Growth",
        "date": "2025-01-29",
        "text": (
            "Tesla Leadership: 'Energy storage deployments reached a record 15.3 GWh in Q4, growing over 125% in 2024. Automotive gross "
            "margin excluding regulatory credits stabilized at 17.2%. Full Self-Driving (FSD) v13 cumulative miles exceeded 2.5 billion, "
            "positioning unsupervised Cybercab commercial rollouts for selected metropolitan regulatory approvals.'"
        ),
    },
    {
        "id": "tsla_news_2025_02_18",
        "ticker": "TSLA",
        "source_type": "financial_news",
        "title": "Barclays Market Note: Tesla Megapack Margin Contribution and EV Volume Projections",
        "date": "2025-02-18",
        "text": (
            "Barclays notes that Tesla Energy division is rapidly scaling to represent >20% of consolidated operating income, offsetting "
            "EV automotive price competition in European and Asian markets."
        ),
    },
    # ── GOOGL Data ────────────────────────────────────────────────────────────
    {
        "id": "googl_ec_q4_2024_01",
        "ticker": "GOOGL",
        "source_type": "earnings_transcript",
        "title": "Alphabet Q4 2024 Earnings Call - Cloud AI Momentum & Search Innovation",
        "date": "2025-02-04",
        "text": (
            "Alphabet CEO Sundar Pichai: 'Google Cloud revenue surged 35% YoY to $11.9 billion, with operating margin expanding to 17.5%. "
            "Over 70% of generative AI startups utilize Google Cloud infrastructure. AI Overviews in Google Search are driving measurable "
            "increases in user queries and commercial satisfaction.'"
        ),
    },
    # ── MSFT Data ────────────────────────────────────────────────────────────
    {
        "id": "msft_ec_q2_2025_01",
        "ticker": "MSFT",
        "source_type": "earnings_transcript",
        "title": "Microsoft Q2 FY25 Earnings Call - Azure AI Capacity & Enterprise Copilot",
        "date": "2025-01-28",
        "text": (
            "Microsoft CEO Satya Nadella: 'Azure revenue grew 31% with 12 points of growth directly attributable to AI services. "
            "Copilot commercial seat adoption doubled quarter-over-quarter, with enterprise retention exceeding 90% across Fortune 500 customers.'"
        ),
    },
    # ── AMZN Data ────────────────────────────────────────────────────────────
    {
        "id": "amzn_ec_q4_2024_01",
        "ticker": "AMZN",
        "source_type": "earnings_transcript",
        "title": "Amazon Q4 2024 Earnings Call - AWS Acceleration & Retail Fulfillment Efficiency",
        "date": "2025-02-06",
        "text": (
            "Amazon CEO Andy Jassy: 'AWS annualized revenue run rate surpassed $110 billion, growing 19% YoY. Regionalized fulfillment "
            "network reduced cost-to-serve per unit by $0.45 while delivering fastest prime delivery speeds in company history. Advertising "
            "revenue reached $17.3 billion, up 24% YoY.'"
        ),
    },
    # ── META Data ────────────────────────────────────────────────────────────
    {
        "id": "meta_ec_q4_2024_01",
        "ticker": "META",
        "source_type": "earnings_transcript",
        "title": "Meta Q4 2024 Earnings Call - Llama Ecosystem & Advantage+ Ad AI ROI",
        "date": "2025-01-29",
        "text": (
            "Meta CEO Mark Zuckerberg: 'Ad impressions increased 18% with average price per ad up 11% driven by Advantage+ AI creative "
            "and bidding optimizations. Family Daily Active People (DAP) reached 3.35 billion. Full-year 2024 operating income grew 42% "
            "to $69.4 billion.'"
        ),
    },
]


def get_corpus_documents(ticker: str | None = None) -> List[Dict[str, str]]:
    """Retrieve corpus documents, optionally filtered by ticker."""
    if not ticker:
        return list(SAMPLE_CORPUS)
    t = ticker.upper()
    filtered = [doc for doc in SAMPLE_CORPUS if doc.get("ticker", "").upper() == t]
    if not filtered:
        # Fallback to general market / all documents if specific ticker has no dedicated docs
        return list(SAMPLE_CORPUS)
    return filtered
