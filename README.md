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

**Recommended Development Environment**

- Python 3.12
- Node.js and npm
- PostgreSQL 16
- pgvector extension compatible with the installed PostgreSQL version
- Conda or another Python virtual environment manager
- Qwen-compatible API credentials
- Internet access for remote LLM requests

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

### 5.3 Clone the Repository

```bash
git clone https://github.com/TianQinNus/AIS_Project.git
cd AIS_Project
```

### 5.4 Backend Setup

Navigate to the backend directory:

```bash
cd SystemCode/backend
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate fleetcare
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

### 5.5 PostgreSQL Database Setup

Install and start PostgreSQL 16 on macOS using Homebrew:

```bash
brew install postgresql@16
brew services start postgresql@16
```

Create the database:

```bash
createdb fleetcare
```

The project requires the PostgreSQL `vector` extension for RAG knowledge retrieval.

After installing a pgvector build compatible with PostgreSQL 16, enable the extension:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Business Database

The FleetCare business database stores vehicle records, operational readouts,
model assessments, rule-engine decisions, and SHAP explanations.

The core tables are:

- `public.truck`
- `public.truck_readout`
- `public.analysis`
- `public.prediction_shap`

#### Step 1: Download the Database Dump

Download the PostgreSQL database dump using the link below:

**[Download `fleetcare.sql`](https://drive.google.com/file/d/1JMNy12i5nmPfxsfILchxve5c9CMH4eAA/view?usp=sharing)**

The downloaded file can remain in your Downloads folder. It does not need to be
copied into the GitHub repository.

#### Step 2: Create the Database

If the `fleetcare` database does not already exist, create it:

```bash
createdb fleetcare
```

If the database already exists and is empty, skip this step.

#### Step 3: Import the Database Dump

Replace `/path/to/fleetcare.sql` with the actual location of the downloaded
file:

```bash
psql -X -v ON_ERROR_STOP=1 \
  -d fleetcare \
  -f /path/to/fleetcare.sql
```

For example, on macOS:

```bash
psql -X -v ON_ERROR_STOP=1 \
  -d fleetcare \
  -f ~/Downloads/fleetcare.sql
```

This command reads the downloaded SQL file, creates the FleetCare tables, and
imports the stored data into PostgreSQL.

> **Important:** Importing the dump into a database that already contains the
> FleetCare tables may produce table conflicts. Follow the database reset
> instructions before importing a fresh copy.

#### Step 4: Verify the Import

Check that the vehicle data were imported successfully:

```bash
psql -d fleetcare -c \
  "SELECT COUNT(*) AS vehicles FROM public.truck;"
```

You can also list the imported tables:

```bash
psql -d fleetcare -c "\dt public.*"
```

The business database dump does not include the RAG knowledge tables. Follow the
separate RAG Knowledge Database instructions to create and populate
`public.knowledge_document` and `public.knowledge_chunk`.

**RAG Knowledge Database**

The knowledge retrieval component uses PostgreSQL and pgvector.

Core knowledge tables include:

- `knowledge_document`
- `knowledge_chunk`

The knowledge chunks contain source text, metadata, citations, and 384-dimensional embeddings.

If the corresponding setup scripts and seed files are available:

```bash
psql -X -v ON_ERROR_STOP=1 \
  -d fleetcare \
  -f scripts/create_knowledge_tables.sql
```

```bash
python -m scripts.import_pgvector_seed \
  --file data/pgvector_seed.json.gz
```

Verify the imported knowledge:

```sql
SELECT COUNT(*)
FROM public.knowledge_chunk;
```

The knowledge database must be populated before maintenance knowledge retrieval can function.


### 5.6 Embedding Model Setup

FleetCareAI uses the `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` model to generate 384-dimensional embeddings for semantic knowledge retrieval.

The model weights are not included in the GitHub repository and must be downloaded separately before using the AI Assistant's RAG functionality.

#### Step 1: Navigate to the Backend Directory

```bash
cd SystemCode/backend
```

#### Step 2: Download the Embedding Model

Ensure that `sentence-transformers` is installed:

```bash
python -m pip install sentence-transformers
```

Download the model and save it to the expected local directory:

```bash
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); model.save('models/multilingual-minilm')"
```

An internet connection is required for the initial download.

#### Step 3: Verify the Model Directory

After downloading, the directory structure should resemble:

```text
SystemCode/
└── backend/
    ├── app/
    ├── models/
    │   └── multilingual-minilm/
    │       ├── config.json
    │       ├── modules.json
    │       ├── model.safetensors
    │       ├── tokenizer.json
    │       └── ...
    ├── scripts/
    └── environment.yml
```

The exact files may vary depending on the model version.

#### Step 4: Verify the Embedding Model

From the backend directory, run:

```bash
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('models/multilingual-minilm', local_files_only=True); print('Embedding dimension:', model.get_sentence_embedding_dimension())"
```

Expected output:

```text
Embedding dimension: 384
```

#### Important Notes

- The embedding model is required for `KnowledgeSearchTool`.
- The model must be available locally before starting knowledge retrieval.
- The backend embedding provider must point to `models/multilingual-minilm`.
- Query embeddings must use the same model as the precomputed document embeddings.
- The model weights are not tracked in Git due to their size.
- The PostgreSQL knowledge tables and pgvector embeddings must also be imported separately.

Without the embedding model, the Fleet Dashboard and structured vehicle APIs may still function, but RAG-based maintenance and policy retrieval will be unavailable.


### 5.7 Environment Configuration

Create a `.env` file inside `SystemCode/backend`.

Choose the Qwen endpoint that matches the platform where the API key was
created. Configure **one endpoint only**; API keys from the two platforms
should not be assumed to be interchangeable.

**Option A — Alibaba Cloud Model Studio, Singapore international region:**

```dotenv
DATABASE_URL=postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/fleetcare

DASHSCOPE_API_KEY=YOUR_INTERNATIONAL_DASHSCOPE_API_KEY
LLM_MODEL=qwen3.5-flash
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1

CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

**Option B — Qwen AI Platform, China region:**

```dotenv
DATABASE_URL=postgresql+psycopg://YOUR_USER:YOUR_PASSWORD@localhost:5432/fleetcare

DASHSCOPE_API_KEY=YOUR_CHINA_QWEN_API_KEY
LLM_MODEL=qwen3.5-flash
QWEN_BASE_URL=https://maas.qianwenaiapi.com/compatible-mode/v1

CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

The selected model must be available to the corresponding account and region.
Do not commit the local `.env` file or expose its API key.


### 5.8 Start the Backend

From `SystemCode/backend`:

```bash
uvicorn app.main:app --reload
```

Default local server:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5.9 Start the Frontend

Open a new terminal:

```bash
cd SystemCode/fleetcare-frontend
npm install
npm run dev
```

Open the local URL displayed by Vite, typically:

```text
http://localhost:5173
```

### 5.10 Using FleetCareAI

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


## SECTION 6: PROJECT REPORT

The full FleetCare AI project report is available in the GitHub repository.

**Folder:** `ProjectReport/`

**Report Link:** [Insert final report URL]

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
