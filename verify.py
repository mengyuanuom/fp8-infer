"""Validate the published measurement tables using only the standard library."""
import csv
import json
from pathlib import Path

root = Path(__file__).resolve().parent
expected = [128, 256, 512, 1024, 2048, 4096, 8192, 10240]
for name in ['pangu_1b', 'qwen_1_7b']:
    with (root / 'data' / f'{name}.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert [int(r['seq_tokens']) for r in rows] == expected
    for r in rows:
        bf, mx = float(r['bf16_p50_ms']), float(r['mxfp8_p50_ms'])
        assert bf > 0 and mx > 0
        assert 0 <= float(r['bf16_matmul_pct']) <= 100
        assert abs(100 * (bf - mx) / bf - float(r['latency_reduction_pct'])) < .01
g = json.loads((root / 'data' / 'pangu_graph_verification.json').read_text())
assert g['status'] == 'passed' and len(g['cases']) == 16
assert {(r['seq'], r['precision']) for r in g['cases']} == {
    (s, p) for s in expected for p in ['bf16', 'mxfp8']
}
assert all(r['fallbacks'] == 0 and r['prefill_replays'] == 65
           and r['decode_replays'] == 130 for r in g['cases'])
with (root / 'data/bf16_matmul_e2e.csv').open(encoding='utf-8', newline='') as f:
    e2e_rows = list(csv.DictReader(f))
assert [int(r['seq_tokens']) for r in e2e_rows] == expected
for row in e2e_rows:
    for model, source in [('pangu', 'pangu_1b'), ('qwen', 'qwen_1_7b')]:
        numerator = float(row[f'{model}_bf16_matmul_mean_ms'])
        denominator = float(row[f'{model}_bf16_e2e_p50_ms'])
        ratio = float(row[f'{model}_bf16_matmul_e2e_pct'])
        assert 0 < numerator <= denominator
        assert abs(ratio - 100 * numerator / denominator) < 1e-10
        with (root / 'data' / f'{source}.csv').open(encoding='utf-8', newline='') as f:
            original = next(r for r in csv.DictReader(f) if r['seq_tokens'] == row['seq_tokens'])
        assert abs(denominator - float(original['bf16_p50_ms'])) <= .00051
print('PASS: measurement tables, graph records and 16 MatMul/E2E estimates verified.')
