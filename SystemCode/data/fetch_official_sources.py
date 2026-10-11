"""Explicit online refresh of allowlisted official originals. Never imported by the agent.
Review source changes before running build_knowledge.py. No generated summaries.
"""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from knowledge_rag import BASE_DIR

SOURCES = [
    ("dvsa", "https://www.gov.uk/guidance/carry-out-daily-heavy-goods-vehicle-hgv-walkaround-checks", "#contents .govspeak", "DVSA", "maintenance", "GB", "Brakes and air build-up"),
    ("hse", "https://www.hse.gov.uk/workplacetransport/vehicles/inspection.htm", "#page-contents", "HSE", "maintenance", "GB", "Planned maintenance"),
    ("lta_inspection", "https://onemotoring.lta.gov.sg/content/onemotoring/home/owning/ongoing-car-costs/inspection.html", ".main_par", "Land Transport Authority Singapore", "policy", "SG", "Vehicle inspection frequency"),
    ("lta_commercial", "https://onemotoring.lta.gov.sg/content/onemotoring/home/driving/commercial-vehicles.html", ".main_par", "Land Transport Authority Singapore", "policy", "SG", "Blind spot safety requirements"),
]


def table_rows(table):
    """Expand row/col spans and repeat header rows on every data row."""
    grid, pending = [], {}
    for ri, tr in enumerate(table.find_all("tr")):
        cells, ci = {}, 0
        for (r,c), value in list(pending.items()):
            if r == ri:
                cells[c] = value
        for cell in tr.find_all(["th", "td"], recursive=False):
            while ci in cells:
                ci += 1
            value = cell.get_text(" ", strip=True)
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            for r in range(ri, ri+rs):
                for c in range(ci, ci+cs):
                    pending[r,c] = value
                    if r == ri:
                        cells[c] = value
            ci += cs
        if cells:
            grid.append([cells.get(c, "") for c in range(max(cells)+1)])
    # LTA frequency table has two header rows; retain both with each row.
    header_count = 2 if grid and any("age" in c.lower() for c in grid[0]) else 1
    header = "\n".join(" | ".join(row) for row in grid[:header_count])
    return [header + "\n" + " | ".join(row) for row in grid[header_count:]] or [header]


def extract(raw, selector):
    soup = BeautifulSoup(raw, "html.parser")
    root = soup.select_one(selector)
    if root is None:
        raise ValueError("Official page layout changed; review the extraction selector")
    for node in root.select("script,style,nav,iframe"):
        node.decompose()
    units, section, block = [], "Introduction", 0
    for node in root.find_all(["h2","h3","h4","p","ul","ol","table"]):
        if node.find_parent(["table","ul","ol"]):
            continue
        text = node.get_text(" ", strip=True)
        if node.name.startswith("h"):
            if text:
                section = text
            continue
        if not text:
            continue
        for content in table_rows(node) if node.name == "table" else [text]:
            block += 1
            unit = {"section": section, "block": block, "text": content}
            if node.name == "table":
                unit["table_subject"] = content.splitlines()[-1].split("|")[0].strip()
            units.append(unit)
    return units


def refresh(root, *, cached=None):
    root = Path(root)
    (root / "official/raw").mkdir(parents=True, exist_ok=True)
    (root / "official/extracted").mkdir(parents=True, exist_ok=True)
    catalog = []
    now = datetime.now(timezone.utc)
    for key,url,selector,publisher,kind,region,expected in SOURCES:
        if cached:
            raw = (Path(cached) / (key+".html")).read_bytes()
            final_url = url
            retrieved = datetime.fromtimestamp((Path(cached) / (key+".html")).stat().st_mtime, timezone.utc)
        else:
            with urlopen(Request(url, headers={"User-Agent": "FleetCareResearch/2.0"}), timeout=45) as response:
                final_url = response.url
                if urlparse(final_url).scheme != "https" or urlparse(final_url).hostname != urlparse(url).hostname:
                    raise ValueError("Unexpected redirect from official publisher")
                raw = response.read(8_000_001)
                if len(raw) > 8_000_000:
                    raise ValueError("Unexpected source size")
            retrieved = datetime.now(timezone.utc)
        units = extract(raw, selector)
        if expected.lower() not in " ".join(u["section"] + " " + u["text"] for u in units).lower():
            raise ValueError("Expected primary-source content missing; reject snapshot")
        original = "official/raw/"+key+".html"
        source = "official/extracted/"+key+".json"
        data = json.dumps({"source_url":url, "units":units}, ensure_ascii=False, indent=2).encode()
        (root/original).write_bytes(raw)
        (root/source).write_bytes(data)
        catalog.append(dict(source=source, original_source=original, title=BeautifulSoup(raw, "html.parser").title.get_text(" ", strip=True),
                            kind=kind, version="Official snapshot "+now.date().isoformat(), reviewed=True,
                            review_level="primary_source_identity_and_extraction_checked", expert_validated=False,
                            applicability=("Singapore policy; apply only after confirming registered vehicle type, age and weight." if region=="SG" else "UK general guidance; not Singapore law or vehicle-specific OEM repair instructions."),
                            source_url=url, final_url=final_url, publisher=publisher, authority="government",
                            evidence_type="official_original", jurisdiction=region, status="active",
                            retrieved_at=retrieved.isoformat(), verified_on=now.date().isoformat(),
                            review_due=(now.date()+timedelta(days=90 if kind=="policy" else 365)).isoformat(),
                            sha256=hashlib.sha256(data).hexdigest(), original_sha256=hashlib.sha256(raw).hexdigest()))
    project = root / "project/fleetcare_policy.md"
    catalog.append(dict(source="project/fleetcare_policy.md", title="FleetCare prototype model boundaries",
                        kind="project", version="Project policy v2", reviewed=True, expert_validated=False,
                        authority="project", publisher="FleetCare Group47", evidence_type="project_policy",
                        jurisdiction="GLOBAL", status="active", applicability="Prototype only; not a government or OEM standard.",
                        verified_on=now.date().isoformat(), review_due=(now.date()+timedelta(days=365)).isoformat(),
                        sha256=hashlib.sha256(project.read_bytes()).hexdigest()))
    (root/"catalog.json").write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"sources":len(catalog), "catalog":str(root/"catalog.json"),
            "next":"Review snapshots and catalog, then run build_knowledge.py. Existing database remains unchanged."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=BASE_DIR/"knowledge")
    args = parser.parse_args()
    print(json.dumps(refresh(args.root), ensure_ascii=False, indent=2))
