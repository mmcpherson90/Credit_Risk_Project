"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine_solution.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once on generated track) -> execute_query -> generate_response
    -> escalate to a human analyst if validation still fails.

Required files (place next to app.py, or change the paths in the sidebar):
    credit_risk_portfolio.db   - SQLite database (opened READ-ONLY)
    test_queries.csv           - ground-truth test cases (optional, used for examples + evaluation)

Required credentials (any ONE of these):
    - .streamlit/secrets.toml  ->  OPENAI_API_KEY = "..."  /  OPENAI_API_BASE = "..."
    - Environment variables    ->  OPENAI_API_KEY / OPENAI_API_BASE
    - config.json next to app.py (same format as the notebook)
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
from typing import TypedDict, List, Any, Dict, Optional  # Type hints and structured data types
import pandas as pd  # Data manipulation and analysis
import sqlite3  # Connect to and query the SQLite database
import sqlparse  # Parse and format SQL queries
import json  # Read and write JSON data
import re  # Perform regular expression-based text processing
import os  # Interact with the operating system and environment variables
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from langchain_openai import ChatOpenAI  # Use OpenAI chat models through LangChain
from langchain_core.prompts import ChatPromptTemplate  # Create reusable structured prompts
from langchain_core.messages import HumanMessage, SystemMessage  # Define system and user messages for LLM calls

import warnings  # Manage Python warning messages
warnings.filterwarnings('ignore')  # Suppress warning messages

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
AUDIT_LOG_FILE = BASE_DIR / "audit_log.jsonl"  # append-only audit trail (separate from the read-only DB)

# ----------------------------------------------------------------------------
# Credentials (OpenAI) - secrets.toml / environment variables / config.json
# ----------------------------------------------------------------------------
def _get_secret(name: str) -> Optional[str]:
    """Look for a credential in Streamlit secrets, then environment variables."""
    try:
        value = st.secrets[name]
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name)


def load_credentials() -> bool:
    """
    Load OpenAI credentials and store them as environment variables
    (same behaviour as the notebook, which read them from config.json).
    Returns True if an API key is available.
    """
    api_key = _get_secret("OPENAI_API_KEY")
    api_base = _get_secret("OPENAI_API_BASE")

    # Fallback: config.json next to app.py (same format used in the notebook)
    config_path = BASE_DIR / "config.json"
    if (not api_key or not api_base) and config_path.exists():
        try:
            with open(config_path, "r") as file:
                config = json.load(file)  # Load the JSON content as a dictionary
            api_key = api_key or config.get("OPENAI_API_KEY")
            api_base = api_base or config.get("OPENAI_API_BASE")
        except Exception:
            pass

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    return bool(api_key)


# ----------------------------------------------------------------------------
# Sidebar configuration
# ----------------------------------------------------------------------------
st.sidebar.title("🏦 Query Engine")
st.sidebar.caption("Northbridge Bank · Commercial Lending Credit Risk")

with st.sidebar.expander("⚙️ Configuration", expanded=False):
    DB_PATH = st.text_input(
        "SQLite database path",
        value=os.environ.get("DB_PATH", "credit_risk_portfolio.db"),
        help="Opened in read-only mode (?mode=ro).",
    )
    TEST_CSV_PATH = st.text_input(
        "Test queries CSV path",
        value=os.environ.get("TEST_QUERIES_PATH", "test_queries.csv"),
    )
    GEN_MODEL = st.text_input("Primary LLM (routing / SQL / narrative)", value="gpt-4o-mini")
    EVAL_MODEL = st.text_input("Evaluator LLM (relevance check)", value="gpt-4o")
    SHOW_TRACE = st.checkbox("Show pipeline trace with each answer", value=True)
    WRITE_AUDIT_FILE = st.checkbox(
        "Also append audit records to audit_log.jsonl",
        value=True,
        help="The database stays read-only; this writes only a local log file.",
    )


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (BASE_DIR / p)


# ----------------------------------------------------------------------------
# LLM setup  (llm = gpt-4o-mini, evaluator_llm = gpt-4o, temperature = 0)
# ----------------------------------------------------------------------------
if not load_credentials():
    st.error(
        "OpenAI credentials not found. Add `OPENAI_API_KEY` (and optionally `OPENAI_API_BASE`) to "
        "`.streamlit/secrets.toml`, set them as environment variables, or provide a `config.json`."
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str):
    return ChatOpenAI(model=model_name, temperature=0)


llm = get_llm(GEN_MODEL)
evaluator_llm = get_llm(EVAL_MODEL)


# ----------------------------------------------------------------------------
# Database loading (READ-ONLY connection)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_connection(db_path: str):
    # check_same_thread=False because Streamlit runs scripts in worker threads
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)


_db_file = _resolve(DB_PATH)
if not _db_file.exists():
    st.error(
        f"Database file not found: `{_db_file}`. Place `credit_risk_portfolio.db` next to `app.py` "
        "or update the path under Configuration in the sidebar."
    )
    st.stop()

conn = get_connection(str(_db_file))  # read-only connection


@st.cache_data(show_spinner=False)
def load_ground_truth(csv_path: str) -> Optional[pd.DataFrame]:
    p = Path(csv_path)
    if p.exists():
        return pd.read_csv(p)
    return None


ground_truth = load_ground_truth(str(_resolve(TEST_CSV_PATH)))

# ----------------------------------------------------------------------------
# Database schema provided to the LLM in the prompts
# ----------------------------------------------------------------------------
database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""

# ----------------------------------------------------------------------------
# Verified Query Template Library
# ----------------------------------------------------------------------------

# Verified Query 1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """SELECT s.sector_name,
     ROUND(SUM(l.total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
     ROUND(SUM(CASE WHEN l.asset_classification IN ('Substandard', 'Doubtful', 'Loss')
                    THEN l.total_outstanding ELSE 0 END) / 1000000.0, 2) AS npa_outstanding_mn
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_mn DESC"""

# Verified Query 2: Portfolio Outstanding by Loan Category
sql_2 = """SELECT loan_category,
     ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn,
     COUNT(*) AS loan_count
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_mn DESC"""

# Verified Query 3: IFRS 9 Stage-wise ECL Summary
sql_3 = """SELECT ifrs9_stage,
     COUNT(*) AS loan_count,
     ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_mn,
     ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage"""

# Verified Query 4: Provision Coverage Ratio by Sector
sql_4 = """SELECT s.sector_name,
     ROUND(AVG(p.provision_coverage_ratio), 2) AS avg_provision_coverage_ratio
FROM provisioning p
JOIN loan_master l ON p.loan_account_number = l.loan_account_number
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC"""

# Verified Query 5: Top 10 Loan Exposures
sql_5 = """SELECT * FROM (
    SELECT l.borrower_name,
         s.sector_name,
         ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
         l.asset_classification
    FROM loan_master l
    JOIN sector_master s ON l.sector_code = s.sector_code
    ORDER BY l.total_outstanding DESC
    LIMIT 10)"""

# Verified Query 6: Top 5 Business Group Exposures
sql_6 = """SELECT * FROM (
    SELECT group_name,
         COUNT(*) AS loan_count,
         ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_exposure_mn
    FROM loan_master
    WHERE group_name IS NOT NULL
    GROUP BY group_name
    ORDER BY total_exposure_mn DESC
    LIMIT 5)"""

# Verified Query 7: All Overdue Loan Accounts
sql_7 = """SELECT l.loan_account_number,
     l.borrower_name,
     s.sector_name,
     ROUND(l.total_outstanding / 1000000.0, 2) AS total_outstanding_mn,
     l.days_past_due,
     l.asset_classification
FROM loan_master l
JOIN sector_master s ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC"""

# Verified Query 8: DPD Bucket Distribution
sql_8 = """SELECT CASE
         WHEN days_past_due = 0 THEN '0 (Current)'
         WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
         WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
         WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
         ELSE '90+'
     END AS dpd_bucket,
     COUNT(*) AS loan_count,
     ROUND(SUM(total_outstanding) / 1000000.0, 2) AS total_outstanding_mn
FROM loan_master
GROUP BY dpd_bucket
ORDER BY MIN(days_past_due)"""

# Verified Query 9: Latest Rating Downgrades
sql_9 = """SELECT borrower_id,
     previous_rating,
     internal_rating,
     pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
  AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC, borrower_id ASC"""

# Verified Query 10: ECL Trend Across Reporting Quarters
sql_10 = """SELECT reporting_date,
     ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_mn
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC"""

# Verified Query Library: query ID -> description + SQL
verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}

# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

# ----------------------------------------------------------------------------
# Intent Classification Tool
# ----------------------------------------------------------------------------
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""

### ROLE
You are the query router for a bank's commercial-lending credit-risk query engine.
Decide whether the user's question can be answered by one of the pre-approved verified query templates,
or whether fresh SQL must be generated.

### INPUT
USER QUESTION:
{user_question}

### VERIFIED TEMPLATE LIBRARY
{library_descriptions}

### INSTRUCTIONS
1. Read the user question and work out what they want to know: which number, grouped by what, and for which time period.
2. Compare the question to each template description in the library below.
3. Match on meaning, not exact words. For example, 'non-performing', 'bad loans' and 'impaired' all mean
NPA (asset_classification is Substandard, Doubtful or Loss); 'overdue' and 'late' mean days_past_due > 0; 'book', 'portfolio'
and 'exposure' mean total_outstanding; 'biggest loans' means the top loans by total_outstanding; 'downgraded' means
rating_direction = 'Downgraded'.
4. If the question is about only one sector or category (for example just Real Estate), it can still use the template that shows
all sectors or categories, because the answer is one of its rows.
5. Check the time period. The stage summary, provision coverage and downgrade templates only show the latest quarter (2025-09-30).
If the question asks about a trend or change over several quarters, choose generated.
The ECL trend template only works for total ECL, not for one stage or one sector.
6. Check the shape of the answer. If the user asks to list individual loans, do not pick a template that only returns totals.
If the user asks for a summary, do not pick a template that returns every loan.
7. If the question needs a filter or column that none of the templates have (for example restructured loans, interest rate, state,
or secured vs unsecured), choose generated.
8. If a template fully answers the question, return its ID. If no template fits, or you are not sure, return null for the query_id
and set the route to generated. Do not guess and do not make up an ID.

### OUTPUT
Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# ----------------------------------------------------------------------------
# Query Generation Tool
# ----------------------------------------------------------------------------
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

### ROLE
You are a senior credit-risk data analyst writing SQL for a bank's read-only SQLite analytical database.
Write one query that answers the user's question.

### INPUT
User Question:
{user_question}

Database Schema (single source of truth):
{schema_context}

### INSTRUCTIONS
1. Output a single read-only SQLite-compatible query; using SELECT (or WITH ... SELECT).
2. Never use DROP, DELETE, UPDATE, INSERT, ALTER, or TRUNCATE.
3. Use only the tables and columns in the schema below. Never invent a column.
4. In SQLite, don't subtract dates directly (e.g. DATE(a) - DATE(b)); it quietly returns 0. Use julianday(a) - julianday(b)
to get the difference in days.

### OUTPUT
Return only the raw SQL. No explanation, no comments, no markdown fences.
"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# ----------------------------------------------------------------------------
# Query Validation Tool
# ----------------------------------------------------------------------------
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Check 1: Read-only shape check
    sql_upper = candidate_sql.upper().strip()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql.rstrip(';').rstrip():
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    # Prompt the LLM to evaluate business relevance
    relevance_prompt = f"""

### ROLE
You are an independent reviewer of SQL for a bank's credit-risk analytics team.
Judge whether the SQL below correctly answers the user's question.

### CONTEXT
{track_context}

### INPUT
User Question:
{user_question}

Canidate SQL:
{candidate_sql}

### INSTRUCTIONS
Assess whether the SQL genuinely answers what the user asked, considering:

1. Does it query the correct tables and columns? Joins should use the right keys and must not multiply rows
   (for example, provisioning joined to loan_master without a reporting_date filter counts every loan once per quarter).
2. Does it apply the right aggregations and groupings? A sum, count or average should be grouped by the dimension
   the user asked about (sector, loan category, IFRS 9 stage, business group, DPD bucket).
3. Does it handle the business definitions correctly? Exposure = total_outstanding; NPA (also called impaired or
   non-performing) = asset_classification IN ('Substandard', 'Doubtful', 'Loss'); overdue = days_past_due > 0.
4. Does it handle the time period correctly? "Latest" provisioning or rating data means '2025-09-30'. A trend or
   "over the last N quarters" question must NOT be filtered to a single date.
5. Does it resolve named entities correctly? A sector, loan category, stage or group named in the question must be matched to the
    real value in the data (for example 'Real Estate' in sector_name, 'SME' in loan_category,
    ifrs9_stage = 3), not to a made-up value.
6. Does it return the right shape of answer? A request to list loans needs row-level results; a request for a
   summary needs aggregated results; a "top N" request needs the matching N.
7. If this is a verified template, do not penalize it for returning a broader result set than the question's
   scope (for example all sectors when the question asks about one).
8. Only judge correctness against the question. Do not fail the query for style, column order or aliases.

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}
"""
    # send the question and response to the evaluator
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql']
        try:
            expected_cols = [d[0] for d in cur.execute(f"{expected_sql} LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"{candidate_sql} LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# ----------------------------------------------------------------------------
# Retry Generation Tool
# ----------------------------------------------------------------------------
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""

### ROLE
You are a senior credit-risk data analyst. The SQLite query below failed validation.
Fix it so it correctly answers the user's question while preserving the original intent.

### INPUT
User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

### INSTRUCTIONS
1. Read the validation error first. Change only the part it points to and leave the parts that already work as they are.
2. Keep the meaning of the user's question the same. Do not switch to a different metric, grouping or time period just to get
the query to pass.
3. Return one read-only statement: a SELECT, or a WITH ... SELECT. Anything that changes data
(INSERT, UPDATE, DELETE, DROP, ALTER, ATTACH, REPLACE) or a second statement is not allowed.
4. Take every table and column name from the given schema. If the error is about an unknown name, look up the correct one
in the schema instead of guessing.
5. Write SQL that runs in SQLite.
6. Keep the project definitions: NPA = asset_classification IN ('Substandard', 'Doubtful', 'Loss');
"latest" provisioning or rating data = '2025-09-30'; money amounts in USD millions (divide by 1000000.0, ROUND to 2 decimals,
column name ending in _mn).
7. If the error says the result does not answer the question, re-check the joins, the GROUP BY and the date filter before anything else.

### OUTPUT
Return the SQL only. No explanation, comments or markdown fences.
"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# ----------------------------------------------------------------------------
# Query Execution Tool
# ----------------------------------------------------------------------------
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# ----------------------------------------------------------------------------
# Response Generation Tool
# ----------------------------------------------------------------------------
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

### ROLE
You are a credit-risk analyst writing a short answer for a business user.
Answer the user's question using only the query result below.

### INPUT
User Question:
{user_question}

Query Result Data:
{dataframe.to_string()}

### INSTRUCTIONS
1. Start with a direct answer to the question that was asked, in the first sentence.
2. If the question is about one sector, loan category, IFRS 9 stage, business group or borrower, focus on those rows. Verified templates often return every row, so leave out the rows that are not relevant.
3. Bring in other rows only when they help the reader, for example to show how a sector ranks against the others or how a figure compares with the whole portfolio.
4. Use the exact figures in the result. Do not change the rounding, estimate, or add numbers that are not in the data. You may calculate a simple ratio from figures that are shown (for example NPA as a % of total outstanding), but say that it is calculated.
5. Point out anything that stands out, such as a sector with a much higher NPA share than its peers, loans that are 90+ days past due or classified Loss, a large share of exposure in one group, or a Stage 3 balance that keeps growing.
6. Always give the unit, based on the column name. Columns ending in _mn are USD millions, so write "$X million". Columns with ratio or rate in the name (for example provision_coverage_ratio, interest_rate) are percentages, so write "X%". loan_count is a number of loans, days_past_due is "X days", and pd_estimate is a probability between 0 and 1, so write 0.99 as "99%". Never write a bare number when the column tells you its unit.
7. If the result is empty, say that no records matched. Do not guess why.
8. Write in clear, professional language suitable for a credit committee memo. Do not mention SQL, tables or how the result was produced, and do not give credit decisions or recommendations that go beyond the data.
9. Keep it short: two to four sentences for a simple question, and a short paragraph (under 150 words) for a complex one.

### OUTPUT
Return only the natural language response text, with no markdown headers or bullet points unless truly needed.
"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# ============================================================================
# LLM BINDING AND PIPELINE ORCHESTRATION
# ============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, records the intermediate pipeline stages in log['trace']
                      (the notebook printed them; the app shows them in the UI).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None,
        'trace': []
    }

    def say(message):
        """Notebook print() statements are captured as a trace for the UI."""
        if verbose:
            log['trace'].append(message)

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    say(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    say(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    say(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    say(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        say(f"    Failed check: {gate.get('failed_check')}")
        say(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        say(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        say(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            say(f"    Retry failed check: {gate.get('failed_check')}")
            say(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        say(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    say(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        say(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    say(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# ============================================================================
# STREAMLIT UI HELPERS
# ============================================================================
def safe_markdown(text: str) -> str:
    """Escape '$' so Streamlit does not render '$X million ... $Y million' as LaTeX."""
    return (text or "").replace("$", "\\$")


def pretty_sql(sql: Optional[str]) -> str:
    if not sql:
        return ""
    return sqlparse.format(sql, reindent=True, keyword_case="upper")


def build_audit_record(res: Dict[str, Any]) -> Dict[str, Any]:
    """Create a JSON-serialisable audit record for one analytical output."""
    gate = res.get('gate_result') or {}
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'user_question': res.get('user_question'),
        'route': res.get('route'),
        'query_id': res.get('query_id'),
        'match_reason': res.get('match_reason'),
        'retry_used': res.get('retry_used'),
        'escalated': res.get('escalated'),
        'validation_passed': gate.get('passed'),
        'failed_check': gate.get('failed_check'),
        'validation_details': gate.get('details'),
        'confidence': res.get('confidence'),
        'candidate_sql': res.get('candidate_sql'),
        'executed_sql': res.get('executed_sql'),
        'row_count': res.get('row_count'),
        'narrative': res.get('narrative'),
        'generator_model': GEN_MODEL,
        'evaluator_model': EVAL_MODEL,
    }


def record_audit(res: Dict[str, Any]) -> None:
    """Keep an in-session audit trail and (optionally) append it to audit_log.jsonl."""
    record = build_audit_record(res)
    st.session_state.setdefault('audit_log', []).append(record)
    if WRITE_AUDIT_FILE:
        try:
            with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass  # read-only file systems (e.g. some hosts) must not break the app


def run_and_audit(question: str) -> Dict[str, Any]:
    res = run_pipeline(question, conn, verified_query_library, database_schema, verbose=True)
    record_audit(res)
    return res


def render_result(res: Dict[str, Any], key_prefix: str = "res") -> None:
    """Render one pipeline result: answer, route/confidence, SQL, data, validation, trace."""
    log = res['log']

    if log['escalated']:
        st.error("⚠️ Escalated to a human analyst")
        st.write(safe_markdown(log['narrative']))
    else:
        st.subheader("Answer")
        st.write(safe_markdown(log['narrative']))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Route", str(log['route']).title())
    c2.metric("Query ID", log['query_id'] if log['query_id'] else "—")
    conf = log['confidence']
    if isinstance(conf, (int, float)):
        c3.metric("Confidence", f"{conf:.2f}")
    else:
        c3.metric("Confidence", str(conf) if conf else "—")
    c4.metric("Retry used", "Yes" if log['retry_used'] else "No")
    c5.metric("Rows returned", log['row_count'] if log['row_count'] is not None else "—")

    if isinstance(conf, (int, float)):
        st.progress(min(max(float(conf), 0.0), 1.0), text="Validation confidence score")

    st.caption(f"Routing reason: {log['match_reason']}")

    sql_to_show = log['executed_sql'] or log['candidate_sql']
    with st.expander("SQL used" if not log['escalated'] else "Candidate SQL (failed validation)", expanded=True):
        st.code(pretty_sql(sql_to_show), language="sql")

    df = res.get('dataframe')
    if df is not None:
        with st.expander("Raw data returned", expanded=True):
            st.dataframe(df, width="stretch")
            st.download_button(
                "⬇️ Download result as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"{key_prefix}_csv",
            )

    gate = log['gate_result'] or {}
    with st.expander("Validation details"):
        st.json({
            'passed': gate.get('passed'),
            'failed_check': gate.get('failed_check'),
            'details': gate.get('details'),
            'relevance_confidence': gate.get('relevance_confidence'),
            'retry_used': log['retry_used'],
        })

    if SHOW_TRACE and log.get('trace'):
        with st.expander("Pipeline trace"):
            st.code("\n".join(log['trace']), language="text")


# ============================================================================
# MAIN LAYOUT
# ============================================================================
st.title("🏦 Credit Risk Query Engine")
st.caption(
    "Ask routine commercial-lending portfolio questions in plain English. "
    "Recurring questions use pre-approved, verified SQL templates; other questions get generated "
    "read-only SQL that is validated before it runs. Every answer shows the SQL, the raw data and a "
    "confidence score, and is written to an audit trail."
)

# Sidebar: quick reference
with st.sidebar.expander("📚 Database schema"):
    st.code(database_schema, language="text")
with st.sidebar:
    st.markdown("**Reporting dates**")
    st.caption("Provisioning latest: 2025-09-30 · Rating latest: 2025-09-30")
    st.caption("NPA = Substandard, Doubtful, Loss")
    st.caption("Database connection: read-only")

tab_ask, tab_eval, tab_lib, tab_audit = st.tabs(
    ["💬 Ask a question", "🧪 Test cases & evaluation", "📚 Verified query library", "🧾 Audit trail"]
)

# ----------------------------------------------------------------------------
# Tab 1: Ask a question
# ----------------------------------------------------------------------------
with tab_ask:
    # Example questions (test-case questions from test_queries.csv when available)
    default_examples = [
        "What is the total outstanding and NPA exposure in the Real Estate sector?",
        "Show me the DPD aging profile of the portfolio.",
        "What is the average interest rate by sector?",
    ]
    examples = default_examples
    if ground_truth is not None and 'User Query' in ground_truth.columns:
        examples = ground_truth['User Query'].dropna().astype(str).tolist()

    st.markdown("**Try an example**")
    example_choice = st.selectbox(
        "Example questions",
        options=["— choose an example —"] + examples,
        label_visibility="collapsed",
    )

    with st.form("question_form", clear_on_submit=False):
        question = st.text_area(
            "Your question",
            value=example_choice if example_choice != "— choose an example —" else "",
            placeholder="e.g. Which sectors have the highest NPA exposure?",
            height=100,
            key=f"question_input_{example_choice}",
        )
        submitted = st.form_submit_button("Run query", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Routing, validating and running your query..."):
                try:
                    st.session_state['last_result'] = run_and_audit(question.strip())
                except Exception as e:
                    st.session_state['last_result'] = None
                    st.error(f"The pipeline hit an unexpected error: {e}")

    if st.session_state.get('last_result'):
        st.divider()
        render_result(st.session_state['last_result'], key_prefix="ask")

# ----------------------------------------------------------------------------
# Tab 2: Test cases & evaluation against ground truth
# ----------------------------------------------------------------------------
with tab_eval:
    st.subheader("Evaluation against ground truth")
    st.write(
        "Runs every test case from `test_queries.csv` through the pipeline and compares the selected route "
        "and verified query ID with the expected values. Reports **Selected Path Accuracy**, "
        "**Selected Query Accuracy** and **Average Confidence Score**."
    )

    if ground_truth is None:
        st.info(f"`{TEST_CSV_PATH}` was not found, so evaluation is unavailable. "
                "Add the file next to `app.py` or update the path in the sidebar.")
    else:
        with st.expander("Ground-truth test cases", expanded=False):
            st.dataframe(ground_truth, width="stretch")

        if st.button("▶️ Run all test cases", type="primary"):
            test_results = []
            progress = st.progress(0.0, text="Starting...")
            total = len(ground_truth)
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                progress.progress(i / total, text=f"Running test case {i + 1} of {total}: {gt['Test Case']}")
                try:
                    test_results.append(run_and_audit(gt['User Query']))
                except Exception as e:
                    # keep the evaluation going; record the failure as an escalated-style result
                    test_results.append({
                        'log': {'user_question': gt['User Query'], 'route': None, 'query_id': None,
                                'match_reason': None, 'candidate_sql': None, 'gate_result': None,
                                'retry_used': False, 'escalated': True, 'executed_sql': None,
                                'row_count': None, 'confidence': 'ERROR',
                                'narrative': f"Pipeline error: {e}", 'trace': []},
                        'dataframe': None,
                        'user_question': gt['User Query'], 'route': None, 'query_id': None,
                        'match_reason': None, 'candidate_sql': None, 'gate_result': None,
                        'retry_used': False, 'escalated': True, 'executed_sql': None,
                        'row_count': None, 'confidence': 'ERROR',
                        'narrative': f"Pipeline error: {e}", 'trace': [],
                    })
            progress.progress(1.0, text="Done")

            evaluation_rows = []
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = test_results[i]

                evaluation_rows.append({
                    'Test Case': gt['Test Case'],
                    'Expected Route': gt['Expected Route'],
                    'Actual Route': tr['route'],
                    'Route Match': tr['route'] == gt['Expected Route'],
                    'Expected Query ID': gt['Expected Query ID'],
                    'Actual Query ID': tr['query_id'],
                    'Query ID Match': (
                        pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
                    ) or tr['query_id'] == gt['Expected Query ID'],
                    'Confidence': tr['confidence'],
                    'Rows Returned': tr['row_count']
                })

            evaluation_df = pd.DataFrame(evaluation_rows)

            path_accuracy = evaluation_df['Route Match'].mean() * 100

            verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
            query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

            average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

            st.session_state['eval'] = {
                'df': evaluation_df,
                'path_accuracy': path_accuracy,
                'query_accuracy': query_accuracy,
                'average_confidence': average_confidence,
                'results': test_results,
            }

        ev = st.session_state.get('eval')
        if ev:
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{ev['path_accuracy']:.1f}%")
            qa = ev['query_accuracy']
            m2.metric("Selected Query Accuracy", f"{qa:.1f}%" if pd.notna(qa) else "n/a")
            ac = ev['average_confidence']
            m3.metric("Average Confidence Score", f"{ac:.2f}" if pd.notna(ac) else "n/a")

            st.dataframe(ev['df'], width="stretch")

            st.markdown("**Per-test-case detail**")
            for i, (_, gt) in enumerate(ground_truth.iterrows()):
                tr = ev['results'][i]
                with st.expander(f"{gt['Test Case']}: {gt['User Query']}"):
                    render_result(tr, key_prefix=f"eval_{i}")
                    st.markdown("**Ground truth expected answer**")
                    st.write(safe_markdown(str(gt['Expected Answer'])))

# ----------------------------------------------------------------------------
# Tab 3: Verified query library
# ----------------------------------------------------------------------------
with tab_lib:
    st.subheader("Verified Query Template Library")
    st.write(
        f"{len(verified_query_library)} pre-approved SQL templates for recurring credit-risk questions. "
        "These run without modification."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'], language="sql")

# ----------------------------------------------------------------------------
# Tab 4: Audit trail
# ----------------------------------------------------------------------------
with tab_audit:
    st.subheader("Audit trail (this session)")
    audit = st.session_state.get('audit_log', [])
    if not audit:
        st.info("No queries have been run in this session yet.")
    else:
        audit_df = pd.DataFrame(audit)
        st.dataframe(audit_df, width="stretch")
        a1, a2 = st.columns(2)
        a1.download_button(
            "⬇️ Download audit trail (CSV)",
            data=audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_trail.csv",
            mime="text/csv",
        )
        a2.download_button(
            "⬇️ Download audit trail (JSON)",
            data=json.dumps(audit, indent=2, default=str).encode("utf-8"),
            file_name="audit_trail.json",
            mime="application/json",
        )
    if WRITE_AUDIT_FILE:
        st.caption(f"Records are also appended to `{AUDIT_LOG_FILE.name}` next to app.py "
                   "(may be ephemeral on hosted platforms).")
