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
print('PASS: 16 measurement rows and 16 PanGu graph replay records verified.')
