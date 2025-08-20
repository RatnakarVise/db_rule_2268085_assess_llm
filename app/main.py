from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os, json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# --- Load environment ---
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=dotenv_path)
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

app = FastAPI(title="OSS Note 2268085 Assessment & Remediation Prompt")

# ---- Input models ----
class MRPUsage(BaseModel):
    table: str
    target_type: str
    target_name: str
    used_fields: List[str]
    suggested_fields: List[str]
    suggested_statement: Optional[str] = None


class NoteContext(BaseModel):
    pgm_name: Optional[str] = None
    inc_name: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    mrp_usage: List[MRPUsage] = Field(default_factory=list)


# ---- Summarizer ----
def summarize_context(ctx: NoteContext) -> dict:
    return {
        "unit_program": ctx.pgm_name,
        "unit_include": ctx.inc_name,
        "unit_type": ctx.type,
        "name": ctx.name,
        "mrp_usage": [item.model_dump() for item in ctx.mrp_usage]
    }


# ---- LangChain Prompt ----
SYSTEM_MSG = "You are a precise ABAP reviewer familiar with SAP Note 2268085 who outputs strict JSON only."

USER_TEMPLATE = """
You are evaluating a system context related to SAP OSS Note 2268085 (MRP Live on SAP HANA - MD01N).
We provide:
- program/include/type metadata
- ABAP code with detected MRP related elements

Your job:
1) Provide a concise **assessment**:
   - If transaction MD01 is found, replace with MD01N.
   - If obsolete table MDKP is used, suggest PPH_DBVM / MARC instead (MRP header/attributes moved).
   - If interface IF_EX_MD_CHANGE_MRP_DATA is used, mention it is obsolete as per note 2268085.
   - Merge all these recommendations into suggested_statement** string.

2) Provide an actionable **LLM remediation prompt**:
   - Reference program/include/type/name.
   - Ask to locate all obsolete calls (direct or dynamic SUBMIT/transaction usage).
   - Replace them with new BAPIs/transactions ensuring functional equivalence.
   - Require JSON output with keys: original_code_snippet, remediated_code_snippet, changes[] (line/before/after/reason).

Return ONLY strict JSON:
{{
  "assessment": "<concise note 1803189 impact>",
  "llm_prompt": "<prompt for LLM code fixer>"
}}
   
Unit metadata:
- Program: {pgm_name}
- Include: {inc_name}
- Unit type: {type}
- Unit name: {name}

System context:
{context_json}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_MSG),
    ("user", USER_TEMPLATE),
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
parser = JsonOutputParser()
chain = prompt | llm | parser


def llm_assess(ctx: NoteContext):
    ctx_json = json.dumps(summarize_context(ctx), ensure_ascii=False, indent=2)
    return chain.invoke({
        "context_json": ctx_json,
        "pgm_name": ctx.pgm_name,
        "inc_name": ctx.inc_name,
        "type": ctx.type,
        "name": ctx.name
    })


@app.post("/assess-2268085")
def assess_note_context(ctxs: List[NoteContext]):
    results = []
    for ctx in ctxs:
        try:
            llm_result = llm_assess(ctx)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"LLM call failed: {e}")

        results.append({
            "pgm_name": ctx.pgm_name,
            "inc_name": ctx.inc_name,
            "type": ctx.type,
            "name": ctx.name,
            "code": "",  # keep ABAP code outside response
            "assessment": llm_result.get("assessment", ""),
            "llm_prompt": llm_result.get("llm_prompt", "")
        })

    return results


@app.get("/health")
def health():
    return {"ok": True, "model": OPENAI_MODEL}
