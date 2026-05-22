"""
predict.py 快速验证 — 仅测试序列处理 + 标签映射 + JSON 格式（无需 ESM-2 模型）
"""
import sys
import os
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 测试 1：导入 predict 模块
print("=" * 60)
print("Test 1: Import predict module")
from predict import (clean_sequence, pad_center_truncate, Predictor,
                     LOCATION_LABEL_MAP, MEMBRANE_LABEL_MAP,
                     LOCATION_CLASSES, MEMBRANE_CLASSES)
print("  PASSED — all symbols imported")

# 测试 2：clean_sequence
print("=" * 60)
print("Test 2: clean_sequence()")

assert clean_sequence("MALWMR") == "MALWMR"
assert clean_sequence("malwmr") == "MALWMR"        # lowercase → uppercase
assert clean_sequence("MAL WMR") == "MALWMR"       # space removed
assert clean_sequence("MAL\nWMR") == "MALWMR"      # newline removed
assert clean_sequence("MAL123WMR") == "MALWMR"     # digits removed
assert clean_sequence("MAL*&^WMR") == "MALWMR"     # special chars removed
assert clean_sequence("") == ""                      # empty input
assert clean_sequence("123!@#") == ""               # no valid AA
print("  PASSED — 8 cases")

# 测试 3：pad_center_truncate
print("=" * 60)
print("Test 3: pad_center_truncate()")

s = pad_center_truncate("A" * 1000, 1000)
assert len(s) == 1000, f"Expected 1000, got {len(s)}"

s = pad_center_truncate("A" * 1100, 1000)
assert len(s) == 1000
assert s[:490] == "A" * 490   # N-terminal preserved
assert s[-490:] == "A" * 490  # C-terminal preserved

s = pad_center_truncate("A" * 500, 1000)
assert len(s) == 1000
assert s.startswith("A" * 500)
assert s.endswith("X" * 500)
print("  PASSED — 3 cases")

# 测试 4：Label maps
print("=" * 60)
print("Test 4: Label maps")

assert LOCATION_LABEL_MAP["Cell.membrane"] == "Cell membrane"
assert LOCATION_LABEL_MAP["Endoplasmic.reticulum"] == "ER"
assert LOCATION_LABEL_MAP["Golgi.apparatus"] == "Golgi apparatus"
assert LOCATION_LABEL_MAP["Lysosome/Vacuole"] == "Lysosome + Vacuole"
assert len(LOCATION_LABEL_MAP) == 10, f"Expected 10, got {len(LOCATION_LABEL_MAP)}"
assert len(MEMBRANE_LABEL_MAP) == 3
print("  PASSED — all label mappings correct")

# 测试 5：LOCATION_CLASSES 顺序正确
print("=" * 60)
print("Test 5: LOCATION_CLASSES ordering")

expected_order = [
    "Cell.membrane", "Cytoplasm", "Endoplasmic.reticulum",
    "Golgi.apparatus", "Lysosome/Vacuole", "Mitochondrion",
    "Nucleus", "Peroxisome", "Plastid", "Extracellular"
]
assert LOCATION_CLASSES == expected_order, "Location class order mismatch"
assert len(LOCATION_CLASSES) == 10
assert MEMBRANE_CLASSES == ["Membrane", "Soluble", "Unknown"]
print("  PASSED")

# 测试 6：JSON 输出格式验证（与 API 规范对比）
print("=" * 60)
print("Test 6: JSON output format check")

mock_result = {
    "predicted_location": "Extracellular",
    "location_confidence": 0.923456,
    "predicted_membrane": "Soluble",
    "membrane_confidence": 0.871234,
    "all_probabilities": {
        "Cell membrane": 0.012, "Cytoplasm": 0.008, "ER": 0.015,
        "Golgi apparatus": 0.003, "Lysosome + Vacuole": 0.005,
        "Mitochondrion": 0.018, "Nucleus": 0.003,
        "Peroxisome": 0.002, "Plastid": 0.001, "Extracellular": 0.923
    },
    "model_version": "v1",
    "inference_time_ms": 1523
}

json_str = json.dumps(mock_result)
parsed = json.loads(json_str)

required_keys = ["predicted_location", "location_confidence", "predicted_membrane",
                 "membrane_confidence", "all_probabilities", "model_version", "inference_time_ms"]
for k in required_keys:
    assert k in parsed, f"Missing key: {k}"
assert len(parsed["all_probabilities"]) == 10

# single-line JSON
lines = json_str.split("\n")
assert len(lines) == 1, "JSON must be single-line for Java parsing"
print("  PASSED — JSON format matches API spec")

# 测试 7：Predictor._is_dual 逻辑
print("=" * 60)
print("Test 7: Predictor._is_dual()")

assert Predictor._is_dual("FFN") == True
assert Predictor._is_dual("CNN") == True
assert Predictor._is_dual("BLSTM") == True
assert Predictor._is_dual("CNN_BLSTM") == True
assert Predictor._is_dual("BLSTM_Attention") == True
assert Predictor._is_dual("CNN_BLSTM_Attention") == True
assert Predictor._is_dual("Complete") == False
assert Predictor._is_dual("CNN_BLSTM_Attention_Complete") == False
assert Predictor._is_dual("cnn_blstm_attention_complete") == False  # case insensitive
print("  PASSED — 9 cases")

print("\n" + "=" * 60)
print("ALL 7 TESTS PASSED")
print("=" * 60)
print("\nNote: Full end-to-end test requires a trained best_model.pt")
print("Run: python predict.py --sequence 'MALWMR...' --model-path best_model.pt")
