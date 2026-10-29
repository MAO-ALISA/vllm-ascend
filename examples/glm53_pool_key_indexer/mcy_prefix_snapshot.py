import json
import os

from cal_prefix_hit_rate import get_prefix_hits_total, get_prefix_queries_total

totals = {"hbm_queries": 0, "hbm_hits": 0, "external_queries": 0, "external_hits": 0}
for port in (9181, 9182):
    queries, external_queries = get_prefix_queries_total(os.environ.get("PREFILL_HOST", "80.48.29.107"), str(port))
    hits, external_hits = get_prefix_hits_total(os.environ.get("PREFILL_HOST", "80.48.29.107"), str(port))
    totals["hbm_queries"] += sum(queries.values())
    totals["hbm_hits"] += sum(hits.values())
    totals["external_queries"] += sum(external_queries.values())
    totals["external_hits"] += sum(external_hits.values())
print(json.dumps(totals, sort_keys=True))
