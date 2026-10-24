# FleetCareAI — Intelligent Fleet Maintenance Decision Support System

**Course:** Intelligent Reasoning Systems (IRS)  
**Project Name:** FleetCareAI  
**Group:** Group 47 
**Institution:** National University of Singapore — Institute of Systems Science (NUS-ISS)

---
## SECTION 1: PROJECT TITLE

# FleetCare AI — Intelligent Fleet Maintenance Decision-Support System

**Predict Risk. Explain Decisions. Prioritise Maintenance.**

FleetCare AI is an intelligent fleet-maintenance platform that helps fleet
managers and maintenance personnel identify high-risk vehicles, prioritise
limited inspection capacity, understand predictive results, and retrieve
relevant maintenance knowledge.

The system integrates machine learning, temporal analysis, rule-based
reasoning, SHAP explainability, and retrieval-augmented generation into a
unified web application.

### Core Features

- Fleet-wide risk monitoring and prioritisation
- Vehicle-level risk analysis and prediction history
- Rule-based maintenance actions with traceable evidence
- SHAP-based model explanations
- Maintenance-capacity and inspection-planning summaries
- AI-assisted retrieval of vehicle assessments and maintenance knowledge

---

## SECTION 2: EXECUTIVE SUMMARY

Commercial vehicle fleets generate large volumes of operational data, but
turning that data into timely maintenance decisions remains challenging.
Scheduled inspections and manual monitoring may not adequately distinguish
between vehicles with different operating histories, changing conditions, and
levels of near-term maintenance risk. Fleet operators must determine not only
which vehicles appear risky, but also which vehicles should be reviewed first,
whether the available evidence is stable, and why a particular action has been
recommended.

FleetCare AI addresses this problem through an integrated, human-in-the-loop
decision-support workflow.

The predictive engine transforms cumulative operational readouts and vehicle
specifications into 534 exposure-normalised and temporal features. A specialist
XGBoost–LightGBM ensemble ranks active vehicles by near-term C4 risk, representing
the final zero-to-six-unit period before a recorded maintenance event. A separate
five-class XGBoost model provides the displayed C0–C4 maintenance stage and
coherent Risk6, Risk12, Risk24, and Risk48 probabilities.

An ordered temporal rule engine converts predictive evidence into operational
actions ranging from `CONTINUE` to `PRIORITY_INSPECT`. The rules consider fleet
risk percentiles, recent persistence, deterioration, disagreement between the
specialist models, and the availability of sufficient prediction history.
Unstable assessments are routed to `MANUAL_REVIEW`, while vehicles with
insufficient data receive `COLLECT_MORE_DATA`.

Each assessment is stored in PostgreSQL with its predicted state, risk values,
fleet percentile, matched rule, action reason, temporal evidence, and strongest
SHAP contributions. SHAP explains which anonymised features influenced the C4
XGBoost component, while the rule trace explains why the final operational
action was assigned. These explanations support prioritisation and model
review; they do not constitute confirmed mechanical diagnoses.

The React dashboard provides fleet-level risk summaries, maintenance-planning
indicators, high-priority alerts, and vehicle-level assessment pages. Users can
review current probabilities, historical C4 rankings, model explanations, rule
evidence, and recommended actions.

FleetCare AI also includes a Qwen-powered assistant with Function Calling. For
vehicle-specific questions, the assistant retrieves stored predictions,
actions, histories, and SHAP evidence from PostgreSQL. For maintenance, policy,
and project questions, it searches a curated knowledge base using multilingual
embeddings and pgvector. Retrieved passages retain their citation and
applicability metadata to support evidence-grounded responses.

The system is implemented using React, FastAPI, SQLAlchemy, PostgreSQL,
pgvector, XGBoost, LightGBM, SHAP, and Qwen. Its modular architecture separates
prediction, operational reasoning, data persistence, application services, and
user interaction.

Overall, FleetCare AI demonstrates how predictive modelling, explicit temporal
reasoning, explainability, and knowledge retrieval can be combined to support
transparent fleet-maintenance prioritisation. The system supports professional
judgement and inspection planning; it does not autonomously diagnose faults or
authorise maintenance.

---

## SECTION 3: PROJECT CONTRIBUTION

| Official Full Name | Student ID | Project Contribution |
|---|---|---|
| Zhang Tianqin | A0356380B | Developed the machine-learning pipeline, including temporal feature engineering, imbalance handling, model experimentation and tuning, ensemble design, rule-engine integration, SHAP explainability, and production inference. Designed the database architecture and table schemas for vehicle data, predictions, explanations, and knowledge retrieval. Built the initial RAG prototype and vehicle data-retrieval tool functions, and contributed to frontend and backend improvements. As project leader, managed overall product functionality, system integration, UI/UX, task allocation, team coordination, meetings, and the project timeline.|
| Zhang Yuyue | A0305487X | Developed the React vehicle analysis screens and the FastAPI backend behind them. The frontend covers vehicle search and action filtering, maintenance status, time-window risk readings, the risk-over-time chart, and action explanations with supporting evidence. Implemented the matching endpoints, SQLAlchemy repositories, and service logic, including operational risk curves that assign each readout a calendar time and a predicted risk window. Ordered maintenance actions by current fleet urgency and added API tests for vehicle queries, risk history, and action explanations. |
| Ma Jialing | A0305478X | Designed and established the React frontend and FastAPI backend architecture, adopting a modular API–Service–Repository structure. Developed React frontend features and FastAPI backend APIs for the Fleet Dashboard and AI assistant. Designed PostgreSQL database integration and implemented fleet risk aggregation, maintenance statistics, and vehicle data retrieval using SQLAlchemy. Integrated Qwen 3.5 Flash with Function Calling and pgvector-based RAG to support vehicle risk explanations and maintenance knowledge queries. Supported frontend–backend integration, API testing, debugging, and system demonstrations. |
| Gong Yucheng | A0355325E | Contributed to the initial project idea, business use cases, and refinement of the FleetCareAI solution. Co-developed the predictive maintenance models through data exploration, temporal feature engineering, model implementation, comparative experiments, tuning, and evaluation. Built the RAG knowledge base using official maintenance guidance, Singapore transport policy references, and project documentation; implemented document processing, multilingual embeddings, hybrid retrieval, and source citations, and migrated knowledge storage to PostgreSQL with pgvector. Prepared deployment instructions and interface documentation to support integration by the web team.

**Project Workstreams**

The project includes the following major workstreams:

1. Machine Learning Model Development and Evaluation
2. Temporal Risk Analysis and Rule-Based Reasoning
3. Explainable AI and SHAP Integration
4. Backend API and Database Development
5. Frontend Dashboard and Vehicle Analysis
6. AI Assistant, Function Calling, and RAG Integration
7. System Integration, Testing, Documentation, and Demonstration

The final contribution table should specify each member's actual responsibilities and deliverables.


## SECTION 4: VIDEO OF SYSTEM MODELLING & USE CASE DEMO

FleetCareAI includes two video presentations covering the product experience and the technical implementation.

### 4.1 Product Demonstration Video

<a href="https://youtu.be/C0QNjExkMWU">
  <img
    src="https://github.com/user-attachments/assets/a0923621-f67a-408a-80ae-72742e5ec27a"
    alt="Watch the FleetCare AI Product Demonstration"
    width="600"
  >
</a>

The product demonstration presents FleetCare AI from the perspective of fleet
managers and maintenance staff, including fleet monitoring, maintenance
planning, vehicle-level explanations, and AI-assisted knowledge retrieval.

### 4.2 Technical System Design Video

<a href="https://youtu.be/pnvF9ENN-2Q">
  <img
    src="https://github.com/user-attachments/assets/65de5173-4420-4cad-9121-91a4677b1c6e"
    alt="Watch the FleetCare AI Technical Walkthrough"
    width="600"
  >
</a>

The technical walkthrough explains the system architecture, predictive models,
temporal rule engine, SHAP explainability, PostgreSQL and FastAPI integration,
Qwen Function Calling, RAG knowledge retrieval, and application execution
sequences.


## SECTION 5: USER GUIDE

### 5.1 System Requirements

The installation walkthrough below targets a local macOS development setup:

- Git and access to this repository
- [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
  (Miniconda, Anaconda, or Miniforge); Python 3.12 is specified by the
  backend environment file
- [Node.js **24.x**](https://nodejs.org/en/download) and npm for the frontend
- [Homebrew](https://brew.sh/) and Xcode Command Line Tools for PostgreSQL/pgvector installation
- PostgreSQL **16** with the `vector` extension built for the same server version
- A valid Qwen-compatible API key and an endpoint/model available to its account
- Internet access for dependencies, the database dump, the embedding-model
  download, and Qwen requests

**Setup order:** clone → Python environment → PostgreSQL/pgvector and database
role → `.env` → business dump → knowledge tables/seed → local embedding model
→ backend → frontend → functional checks.

The supplied database contains precomputed assessments and SHAP explanations;
model training or inference is not required to run the web demonstration.

### 5.2 Project Architecture

FleetCareAI follows a modular architecture:

```text
React Frontend
    |
    v
FastAPI Backend
    |
    +-- Dashboard Services
    |       |
    |       +-- SQLAlchemy Repositories
    |               |
    |               +-- PostgreSQL
    |
    +-- AI Assistant
            |
            +-- Qwen LLM
            |
            +-- Function Calling
                    |
                    +-- VehicleAnalysisTool
                    |       |
                    |       +-- PostgreSQL
                    |           (Vehicle Analysis + SHAP)
                    |
                    +-- KnowledgeSearchTool
                            |
                            +-- Multilingual Embedding
                            |
                            +-- pgvector
                                (Maintenance Knowledge)
```

### 5.3 Clone the Repository and Set the Project Path

The commands below use **macOS with Homebrew and Conda**. Run Steps 5.3–5.8
in the same terminal. Keep the backend running in that terminal, then open a
second terminal for Step 5.9.

```bash
git clone https://github.com/TianQinNus/AIS_Project.git
cd AIS_Project
export FLEETCARE_ROOT="$PWD"
```

If the repository is already cloned, enter its root directory and run only
`export FLEETCARE_ROOT="$PWD"`. If the repository requires authentication, use
your authorised GitHub account or clone with
`git clone git@github.com:TianQinNus/AIS_Project.git` instead.

All paths below are quoted so that project folders containing spaces work.

### 5.4 Create the Python Environment

From the repository root:

```bash
conda env create -f "$FLEETCARE_ROOT/SystemCode/backend/environment.yml"
conda activate fleetcare
python --version
python -m pip check
cd "$FLEETCARE_ROOT/SystemCode/backend"
```

Expected: Python 3.12 and `No broken requirements found.` If the `fleetcare`
environment already exists, replace the creation command with:

```bash
conda env update -n fleetcare \
  -f "$FLEETCARE_ROOT/SystemCode/backend/environment.yml" --prune
conda activate fleetcare
```

The supplied environment file contains macOS-specific Conda packages. It is the
backend environment; `SystemCode/requirements.txt` is only a placeholder and
must not be used to install the web application. Re-running predictive-model
inference is a separate workflow described in Section 7.2.

### 5.5 Install PostgreSQL 16 and Prepare an Empty Database

**Install the server and put its matching client tools on PATH:**

```bash
brew install postgresql@16
export PATH="$(brew --prefix postgresql@16)/bin:$PATH"
brew services start postgresql@16
psql --version
pg_isready -h 127.0.0.1 -p 5432
```

Use an up-to-date PostgreSQL 16 installation and client. The readiness command
should report `accepting connections`. These instructions assume that port
5432 is available; stop any conflicting local PostgreSQL service or consistently
use a different port in all database commands and in `.env`.

**Install pgvector for this PostgreSQL 16 server:**

The extension must be installed on the database server before it can be enabled
with SQL. To avoid installing it for a different Homebrew PostgreSQL version,
build it using PostgreSQL 16's `pg_config`:

```bash
export PG_CONFIG="$(brew --prefix postgresql@16)/bin/pg_config"
export FLEETCARE_PGVECTOR_BUILD="$(mktemp -d)"
git clone --depth 1 --branch v0.8.7 \
  https://github.com/pgvector/pgvector.git \
  "$FLEETCARE_PGVECTOR_BUILD/pgvector"
make -C "$FLEETCARE_PGVECTOR_BUILD/pgvector" PG_CONFIG="$PG_CONFIG"
make -C "$FLEETCARE_PGVECTOR_BUILD/pgvector" PG_CONFIG="$PG_CONFIG" install
```

The build requires the Xcode Command Line Tools. If they are missing, run
`xcode-select --install`, complete the installer, and retry the build. If the
last command reports a permission error writing to the PostgreSQL installation,
rerun that installation command with `sudo`, retaining the explicit
`PG_CONFIG="$PG_CONFIG"` argument. See the
[official pgvector installation notes](https://github.com/pgvector/pgvector#installation-notes)
for platform-specific build issues.

**Create the application role, database, and extension:**

Run these commands as the local PostgreSQL administrator. A fresh Homebrew
installation normally provides an administrator role matching your macOS user.
If your server uses another administrator, add `-U YOUR_ADMIN` to these commands.

```bash
createuser --login --pwprompt fleetcare_app
createdb --owner=fleetcare_app fleetcare
psql -X -v ON_ERROR_STOP=1 -d fleetcare \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql -X -d fleetcare \
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

Choose and remember the password prompted by `createuser`; use the same role
and password in `.env`. If this role already exists, reuse its known password.
If the database already exists, proceed only if it has no FleetCare tables;
otherwise choose a new empty database and use its name consistently. Do not
import the dump over an existing populated database.

For the remaining database commands in this terminal, select the application
connection explicitly:

```bash
export PGHOST=127.0.0.1
export PGPORT=5432
export PGUSER=fleetcare_app
export PGDATABASE=fleetcare
psql -X -W -c "SELECT current_user, current_database();"
```

Enter the application-role password when prompted. Expected user/database:
`fleetcare_app` / `fleetcare`.

### 5.6 Configure the Backend Before Importing Data

Keep the active `fleetcare` environment and work from the backend directory:

```bash
cd "$FLEETCARE_ROOT/SystemCode/backend"
cp .env.example .env
```

Run the copy command only when `.env` does not already exist. Open `.env` in a
text editor and replace the placeholders:

```dotenv
DATABASE_URL=postgresql+psycopg://fleetcare_app:YOUR_URL_ENCODED_PASSWORD@127.0.0.1:5432/fleetcare
DASHSCOPE_API_KEY=YOUR_QWEN_API_KEY
LLM_MODEL=qwen3.5-flash
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
DEBUG=false
```

- Use the database role/password created in Step 5.5. Percent-encode reserved
  characters in the password portion of `DATABASE_URL` (for example, `@` becomes
  `%40` and `#` becomes `%23`). Enter the original password at `psql` prompts.
- The endpoint above is for Alibaba Cloud Model Studio's Singapore international
  region. For a key issued by the Qwen AI Platform in China, use
  `QWEN_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1` instead.
  Configure one endpoint matching the key's platform; the selected model must
  be available to that account and region.
- `DATABASE_URL`, `DASHSCOPE_API_KEY`, and `QWEN_BASE_URL` are required by the
  current settings loader, including during knowledge-seed import. Configure
  them **before** running the import script. A real API key is needed to verify
  the assistant; the dashboard itself does not make LLM requests.
- Keep `.env` local; it is ignored by Git. Do not commit credentials. The backend
  loads `.env` relative to the current directory, so run its Python commands
  from `SystemCode/backend`.

Check that the settings load and that Python connects to the intended database
without printing the credentials:

```bash
python -c "from sqlalchemy import text; from app.db.session import engine; c = engine.connect(); print(c.execute(text('SELECT current_user, current_database()')).one()); c.close()"
```

### 5.7 Import Demonstration Data and Prepare Knowledge Retrieval

**1. Download and import the business database.**

Download [fleetcare.sql from Google Drive](https://drive.google.com/file/d/1JMNy12i5nmPfxsfILchxve5c9CMH4eAA/view?usp=sharing)
and save it as `fleetcare.sql` in your Downloads folder (approximately 200 MiB).
It contains vehicle readouts, precomputed model assessments, rule actions, and
SHAP records. Normal web startup does not require running model inference.

From the backend directory, with the database connection variables from Step
5.5 still set:

```bash
psql -X -W -v ON_ERROR_STOP=1 -f "$HOME/Downloads/fleetcare.sql"
```

Change the quoted file path if you saved the download elsewhere. Import into
the empty database prepared in Step 5.5. Stop and resolve any SQL error before
continuing; a partially imported database is not ready for use.

Verify all four business tables, not just that the database exists:

```bash
psql -X -W -v ON_ERROR_STOP=1 -c "
SELECT 'truck' AS table_name, COUNT(*) AS rows FROM public.truck
UNION ALL SELECT 'truck_readout', COUNT(*) FROM public.truck_readout
UNION ALL SELECT 'analysis', COUNT(*) FROM public.analysis
UNION ALL SELECT 'prediction_shap', COUNT(*) FROM public.prediction_shap;"
```

The supplied demonstration snapshot contains 5,045 trucks, 193,095 readouts,
30,265 assessments, and 149,045 SHAP records.

**2. Create the knowledge tables and import the supplied seed.**

These files are included in the repository. The business dump does not contain
the knowledge tables, so this is a required step for document retrieval:

```bash
cd "$FLEETCARE_ROOT/SystemCode/backend"
psql -X -W -v ON_ERROR_STOP=1 -f scripts/create_knowledge_tables.sql
python -m scripts.import_pgvector_seed --file data/pgvector_seed.json.gz
```

Expected importer output:

```text
Import completed successfully.
Documents imported: 5
Chunks processed: 274
Embedding dimension: 384
```

Verify the stored vectors:

```bash
psql -X -W -v ON_ERROR_STOP=1 -c "
SELECT COUNT(*) AS documents FROM public.knowledge_document;
SELECT COUNT(*) AS chunks, MIN(vector_dims(embedding)) AS min_dimension,
       MAX(vector_dims(embedding)) AS max_dimension
FROM public.knowledge_chunk;"
```

Expected: 5 documents, 274 chunks, and dimensions 384 / 384. The seed importer
updates existing records by source/citation ID, so it can be rerun without
creating duplicate citations.

**3. Download and verify the matching embedding model.**

The Python environment from Step 5.4 already includes `sentence-transformers`.
From the backend directory, download the same model used to produce the seed
vectors:

```bash
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); model.save('models/multilingual-minilm')"
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('models/multilingual-minilm', local_files_only=True); print('Embedding dimension:', model.get_sentence_embedding_dimension())"
```

Expected: `Embedding dimension: 384`. Internet access is required for the first
download. The backend reads `SystemCode/backend/models/multilingual-minilm`
directly; downloading the model only to a global cache is insufficient. Model
weights are not tracked in Git. Both the local model and the populated knowledge
tables are needed by `KnowledgeSearchTool`.

### 5.8 Start the Backend — Terminal 1

Keep this process running:

```bash
cd "$FLEETCARE_ROOT/SystemCode/backend"
conda activate fleetcare
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete.` The API is at
<http://127.0.0.1:8000>, with interactive documentation at
<http://127.0.0.1:8000/docs>.

### 5.9 Install and Start the Frontend — Terminal 2

Open a **new terminal**. Replace `/absolute/path/to/AIS_Project` with the actual
repository location; shell variables from Terminal 1 do not carry over:

```bash
cd "/absolute/path/to/AIS_Project/SystemCode/fleetcare-frontend"
node --version
npm --version
npm ci
npm run build
npm run dev -- --host localhost --port 5173 --strictPort
```

Use Node.js **24.x** for this walkthrough. `npm ci` installs the versions in the
committed lockfile; `npm run build` checks TypeScript and the production build
before starting the development server.

Open **<http://localhost:5173>** and keep both terminals running. The frontend's
Vite configuration proxies `/api` requests to `http://127.0.0.1:8000`, so no
frontend `.env` file is needed for this local setup. Leave `VITE_API_BASE_URL`
unset. If port 5173 is occupied, stop the conflicting process before retrying;
`--strictPort` prevents a silent change to an origin missing from CORS settings.

`npm run build` alone does not start a server. This walkthrough uses
`npm run dev`; serving the built files separately requires configuring their
API routing.

### 5.10 Verify the Running Web Product

In a third terminal, run:

```bash
curl --fail-with-body http://127.0.0.1:8000/health
curl --fail-with-body http://127.0.0.1:8000/api/v1/dashboard
curl --fail-with-body 'http://127.0.0.1:8000/api/v1/vehicles?page=1&page_size=1'
curl --fail-with-body http://127.0.0.1:8000/api/v1/vehicles/VHC-24705
```

Expected: HTTP success and JSON data. `/health` returns `{"status":"ok"}`;
the dashboard and vehicle endpoints must also return populated data because
the health endpoint alone does not test the database.

Then check the browser at <http://localhost:5173>:

1. The Dashboard loads fleet totals and priority alerts.
2. Open Vehicles, search for `24705`, and inspect its risk, history, action,
   and SHAP explanation.
3. In the AI Assistant, ask `Explain the current action for vehicle 24705.`
4. Ask `What should I check before driving a heavy goods vehicle?` and check
   that the answer includes retrieved source citations.

The two assistant checks require a working Qwen key/endpoint/model; the document
question additionally verifies the local embedding model and knowledge tables.
An HTTP-level assistant check is also available:

```bash
curl --fail-with-body --max-time 120 \
  http://127.0.0.1:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"What should I check before driving a heavy goods vehicle?"}'
```

Expected: a JSON object with a nonempty `answer`. Review its content and sources
as well as the HTTP status.

### 5.11 Using FleetCareAI

**Fleet Dashboard**

The Dashboard provides fleet-level information, including:

- Active vehicles
- Vehicles at risk
- Average fleet risk score
- Maintenance capacity
- Risk distribution
- Top risk alerts
- Maintenance plan summary

**Vehicle Analysis**

Staff can inspect individual vehicle assessments, including prediction results, risk indicators, maintenance priorities, recommended actions, and SHAP explanations.

**AI Assistant**

The AI Assistant supports two primary question types.

*Vehicle Analysis Example:*

> How has vehicle 18734's risk changed recently?

The assistant retrieves stored vehicle analysis data and explains relevant risk indicators, prediction history, and recommended actions.

*Maintenance Knowledge Example:*

> What should I check before driving a heavy goods vehicle?

The assistant retrieves relevant maintenance reference material and provides supporting sources where available.

The assistant is intended for decision support. It does not replace professional vehicle inspection, qualified mechanical diagnosis, or official regulatory advice.

### 5.12 Troubleshooting and Subsequent Starts

| Symptom | Check / resolution |
|---|---|
| `conda: command not found` or activation fails | Install/initialise Conda for your shell and open a new terminal before Step 5.4. |
| Missing settings / validation error during seed import or startup | Create and complete `backend/.env` first; activate `fleetcare` and run from `SystemCode/backend`. |
| `psql: command not found` | Repeat the PostgreSQL 16 PATH export in Step 5.5 in the current terminal. |
| Connection refused or password authentication failed | Check `pg_isready`, port, database name, role, and password. Match `.env` to the `psql` connection. |
| `extension "vector" is not available` | Build pgvector using the `pg_config` for the running server, then enable it in the correct database as administrator. |
| `invalid command \restrict` when importing the dump | Update the PostgreSQL 16 client and recheck `psql --version`; use a fresh empty database after an incomplete import. |
| Tables already exist during business-data import | Use a new empty database rather than rerunning the dump over populated tables. |
| Dashboard errors or empty data | Check all four business-table counts and the backend logs; `/health` success alone is insufficient. |
| Embedding model not found / knowledge retrieval fails | Verify `backend/models/multilingual-minilm`, the 384-dimensional model, pgvector, and the knowledge-table counts. |
| Assistant authentication/model error | Check the key's platform/region, endpoint, account access, and `LLM_MODEL`, then restart the backend after editing `.env`. |
| Frontend install/build reports an unsupported Node version | Use Node.js 24.x, rerun `npm ci`, and then build again. |
| Browser requests fail while direct API checks succeed | Use `http://localhost:5173`, leave `VITE_API_BASE_URL` unset, and confirm that both servers are running on the documented ports. |

After the first successful setup, do not recreate the database, reimport the
dump, or redownload the embedding model for each run. Start PostgreSQL, activate
`fleetcare`, and repeat the backend and frontend start commands in Steps 5.8 and
5.9. Run `npm ci` again when dependencies change. Stop each web server with
`Ctrl+C`; this leaves the database data intact.


## SECTION 6: PROJECT REPORT

The final FleetCare AI project report is available in this repository and on
Google Drive.

**Final Report (PDF, 30 pages):**

- [View the report in this repository](<ProjectReport/Final Report.pdf>)
- [View the report on Google Drive](https://drive.google.com/file/d/1AdJuVVEpTVI3lskZmlor3J-SzVUQKCPN/view)
- [Download the report from Google Drive](https://drive.google.com/uc?export=download&id=1AdJuVVEpTVI3lskZmlor3J-SzVUQKCPN)

### Report Contents

1. Introduction
2. Project Background and Commercial Context
3. Project Scope and Limitations
4. Data Collection and Preparation
5. System Architecture and Design
6. Predictive Model Implementation and Evaluation
7. Temporal Rule-Based Reasoning
8. Inference, Database Persistence, and SHAP Explainability
9. RAG-Based Assistant
10. Frontend and Backend Application
11. Predictive-Modelling Experiments
12. RAG and System-Integration Evaluation
13. Future Work
14. Appendices and Installation Guide

### Evaluation and Validation

FleetCare AI was evaluated across four connected areas: predictive modelling,
rule-based reasoning, knowledge retrieval, and application integration.

#### Predictive-Model Evaluation

The final C4 ranking ensemble was evaluated on the official Test split,
containing 5,045 vehicles and 60 C4 cases.

At a fixed Top-5% inspection capacity:

- C4 vehicles identified: 20 of 60
- Precision at Top 5%: 7.91%
- Recall at Top 5%: 33.33%
- Average Precision: 0.0552
- ROC–AUC: 0.7907

The original five-class baseline identified four C4 vehicles within the same
Top-5% capacity. These results evaluate fleet prioritisation rather than
confirmed fault diagnosis.

#### Rule-Engine Evaluation

The rule engine was evaluated using stored prediction histories and the final
C4 rankings. The model-defined Top-5% queue contained 253 vehicles and identified
20 C4 cases.

The additional TR2B rescue rule selected 22 vehicles outside this queue and
identified two additional C4 cases. The resulting 275-vehicle inspection scope
contained 22 C4 cases, corresponding to 8.00% precision and 36.67% recall.

Selecting the next 22 vehicles directly from the model ranking produced the same
number of C4 cases. TR2B therefore provides a transparent deterioration-based
escalation condition, but does not demonstrate an equal-capacity improvement
over direct model ranking.

#### RAG Evaluation

The assistant was evaluated using ten document questions and three synthetic
vehicle-assessment cases. The document set included eight answerable questions
and two boundary cases covering unavailable OEM information and an
out-of-domain request.

Controlled knowledge retrieval achieved:

- Macro Precision@5: 0.175
- Macro Recall@5: 0.5313

The integrated assistant achieved:

- Macro Precision@5: 0.150
- Macro Recall@5: 0.4063
- Mean automated response score: 5.40/8
- Mean response time: 4.40 seconds

When retrieved evidence was supplied directly to the language model,
fixed-evidence generation achieved a mean automated score of 7.60/8.

All three synthetic vehicle cases produced the expected tool calls and stored
assessment fields, with a mean automated response score of 7.33/8. HTTP checks
also verified the health endpoint, request validation, and vehicle-assessment
chat workflow.

These results are based on a small local evaluation set and automated scoring.
They demonstrate that the retrieval and tool-calling workflows are operational,
but do not establish production-level answer accuracy. The evaluation also
identified remaining limitations in citation coverage, evidence-to-answer
consistency, and out-of-domain handling.

Refer to the full project report for the experimental methodology, detailed
results, system limitations, and future improvements.


## SECTION 7: TECHNICAL REFERENCE AND LIMITATIONS

### 7.1 Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, TypeScript, Vite |
| Backend | Python, FastAPI |
| ORM | SQLAlchemy |
| Database | PostgreSQL |
| Vector Database Capability | pgvector extension |
| AI Assistant | Qwen 3.5 Flash |
| Agent Integration | Function Calling |
| Embedding Model | Multilingual MiniLM |
| Explainability | SHAP |
| Reasoning | Temporal Analysis and Rule-Based Reasoning |
| API Documentation | FastAPI Swagger UI |
| Version Control | Git and GitHub |

### 7.2 Predictive Inference Workflow

FleetCare AI uses a database-backed inference pipeline to convert vehicle
readouts into maintenance assessments. For each eligible vehicle, the pipeline
selects the latest available readout, reconstructs the frozen 534-feature model
input, and applies the final predictive models.

The XGBoost–LightGBM ensemble ranks near-term C4 risk, while the five-class
XGBoost model provides the predicted C0–C4 maintenance stage and the Risk6,
Risk12, Risk24, and Risk48 values. Recent prediction evidence is then used to
calculate persistence, deterioration, volatility, and model disagreement. The
temporal rule engine converts this evidence into a maintenance action, and SHAP
identifies the strongest feature contributions from the C4 XGBoost component.

The completed assessment, matched rule, supporting evidence, and SHAP
explanations are stored in PostgreSQL for use by the Fleet Dashboard, Vehicle
Analysis pages, and AI Assistant.

The main implementation files are:

- `SystemCode/model/fleetcare_final_inference.py`
- `SystemCode/model/final_model_test54/final_model.joblib`
- `SystemCode/model/final_model_test54/metadata.json`

> **Demonstration data:** The supplied `fleetcare.sql` dump already contains
> the vehicle readouts, model assessments, rule-engine actions, and SHAP
> explanations required for the product demonstration. Re-running model
> inference is not part of the normal application startup process. It is only
> required when new readouts are added, the model or rules are updated, or the
> stored assessments need to be regenerated.

### 7.3 Knowledge Sources

The RAG knowledge base includes selected reference materials from:

- UK Driver and Vehicle Standards Agency (DVSA)
- UK Health and Safety Executive (HSE)
- Singapore Land Transport Authority (LTA)
- FleetCareAI Project Rule Documentation

The documents support general maintenance guidance, regulatory reference, and explanations of project-specific reasoning concepts.

UK maintenance guidance is not presented as Singapore law. Policy applicability must be checked against the relevant vehicle type, registration details, and current official requirements.

### 7.4 Key System Design Principles

**Modular Architecture**

The system separates API routing, business services, repositories, and AI tools to support maintenance and extensibility.

**Explainable Reasoning**

Maintenance decisions include rule identifiers, evidence, and explanations. SHAP provides additional insight into model predictions.

**Grounded AI Assistance**

Vehicle-specific information is retrieved from structured database records, while maintenance knowledge is retrieved from relevant reference documents.

**Human-in-the-Loop Decision Support**

The system provides maintenance recommendations and supporting information. Final inspection and repair decisions remain the responsibility of qualified personnel.

### 7.5 Limitations and Future Work

### Current Limitations

Current limitations include:

- The operational features are anonymised, preventing component-level mechanical interpretation
- Model performance has been evaluated on the SCANIA benchmark rather than a live commercial fleet
- Risk percentiles depend on the composition of the currently eligible fleet
- Vehicles with limited prediction history may not provide enough evidence for temporal reasoning
- Rule thresholds are based on benchmark results and require validation against real maintenance policies and workshop capacity
- The RAG knowledge base currently covers a limited set of maintenance, policy, and project documents
- Citation coverage and evidence-to-answer consistency remain incomplete for some AI-generated responses
- The assistant processes each request independently and does not retain multi-turn conversational memory
- Final inspection and maintenance decisions still require review by qualified personnel

### Future Enhancements

Potential future enhancements include:

- Retraining the predictive models with documented operational features and confirmed maintenance outcomes
- Validating model rankings and rule thresholds using live fleet data
- Expanding the knowledge base with authorised service manuals, maintenance procedures, work orders, and internal policies
- Improving retrieval relevance, citation coverage, and human evaluation of generated responses
- Adding controlled multi-turn conversational memory
- Introducing role-based access control and document-level permissions
- Integrating maintenance scheduling, workshop capacity, and work-order systems
- Monitoring data drift, model performance, action volumes, and fixed-capacity C4 recovery
- Supporting version-controlled model retraining, rule updates, and knowledge-base maintenance
- Deploying the system within a secure and monitored fleet environment

## Acknowledgements

This project was developed as part of the Intelligent Reasoning Systems coursework at the National University of Singapore, Institute of Systems Science.

We thank the teaching team for their guidance and feedback throughout the project.

---

**FleetCareAI — From Predictive Insights to Explainable Maintenance Decisions.**
