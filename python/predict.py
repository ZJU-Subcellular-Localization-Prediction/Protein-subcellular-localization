"""
推理脚本 — 供 Java 后端通过 ProcessBuilder 调用

用法:
    python predict.py --sequence "MALWMRLLPLL..."
    python predict.py --sequence "MALW..." --model-path ./best_model.pt

输出: 单行 JSON 到 stdout（Java 端解析最后一行以 { 开头的行）
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ---- 常量 ----

LOCATION_CLASSES = [
    "Cell.membrane", "Cytoplasm", "Endoplasmic.reticulum",
    "Golgi.apparatus", "Lysosome/Vacuole", "Mitochondrion",
    "Nucleus", "Peroxisome", "Plastid", "Extracellular"
]

MEMBRANE_CLASSES = ["Membrane", "Soluble", "Unknown"]

LOCATION_LABEL_MAP = {
    "Cell.membrane": "Cell membrane",
    "Cytoplasm": "Cytoplasm",
    "Endoplasmic.reticulum": "ER",
    "Golgi.apparatus": "Golgi apparatus",
    "Lysosome/Vacuole": "Lysosome + Vacuole",
    "Mitochondrion": "Mitochondrion",
    "Nucleus": "Nucleus",
    "Peroxisome": "Peroxisome",
    "Plastid": "Plastid",
    "Extracellular": "Extracellular",
}

MEMBRANE_LABEL_MAP = {"Membrane": "Membrane", "Soluble": "Soluble", "Unknown": "Unknown"}

SEQ_LEN = 1000
ESM_MODEL_NAME = "facebook/esm2_t30_150M_UR50D"

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---- 序列预处理 ----

def clean_sequence(raw: str) -> str:
    """转大写、去空格、去除非氨基酸字符"""
    return "".join(c for c in raw.upper().replace("\n", "").replace(" ", "") if c in VALID_AA)


def pad_center_truncate(sequence: str, target_len: int) -> str:
    """
    中心截断 + 末端 padding，与 extract_features.py 逻辑一致。
    - 太长 → 从中心删除（保护 N 端/C 端信号）
    - 太短 → 末端补 X（ESM tokenizer 会将其映射为 <unk>）
    """
    if len(sequence) > target_len:
        extra = len(sequence) - target_len
        mid = len(sequence) // 2
        start = mid - extra // 2
        end = mid + (extra + 1) // 2
        return sequence[:start] + sequence[end:]
    elif len(sequence) < target_len:
        return sequence + "X" * (target_len - len(sequence))
    return sequence


# ---- 推理核心 ----

class Predictor:
    """预加载模型，执行推理。单次加载可复用。"""

    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载 checkpoint
        if not os.path.exists(model_path):
            alt = os.path.join(os.path.dirname(__file__), model_path)
            if os.path.exists(alt):
                model_path = alt
            else:
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        self.ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.args = self.ckpt["args"]
        self.model_name = self.args.get("model", "CNN_BLSTM_Attention")
        self.is_dual = self._is_dual(self.model_name)

        # 构建模型
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.architectures import create_model

        self.model = create_model(
            self.model_name,
            seq_len=SEQ_LEN,
            n_feat=640,
            n_hid=self.args.get("n_hid", 64),
            n_class=10,
            drop_prob=self.args.get("drop_prob", 0.3),
            n_filt=self.args.get("n_filt", 32),
            drop_hid=self.args.get("drop_hid", 0.3),
        ).to(self.device)
        self.model.load_state_dict(self.ckpt["model_state_dict"])
        self.model.eval()

        # 加载 ESM-2
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(ESM_MODEL_NAME)
        self.esm = AutoModel.from_pretrained(ESM_MODEL_NAME).to(self.device)
        self.esm.eval()

        print(f"[predict] Loaded model={self.model_name} dual={self.is_dual} device={self.device}",
              file=sys.stderr)

    @staticmethod
    def _is_dual(name: str) -> bool:
        name_lower = name.lower().replace("-", "_").replace(" ", "_")
        return not ("complete" in name_lower)

    @torch.no_grad()
    def predict(self, sequence: str) -> dict:
        t_start = time.time()

        # 1. 预处理
        cleaned = clean_sequence(sequence)
        if not cleaned:
            raise ValueError("No valid amino acids in sequence")
        processed = pad_center_truncate(cleaned, SEQ_LEN)

        # 2. ESM-2 embedding
        inputs = self.tokenizer(processed, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        esm_out = self.esm(**inputs)
        # 去掉 BOS (index 0) 和 EOS (index -1)
        embedding = esm_out.last_hidden_state[:, 1:-1, :]  # (1, raw_len, 640)

        # center truncate/pad to SEQ_LEN
        raw_len = embedding.shape[1]
        if raw_len > SEQ_LEN:
            extra = raw_len - SEQ_LEN
            mid = raw_len // 2
            start = mid - extra // 2
            end = mid + (extra + 1) // 2
            embedding = torch.cat([embedding[:, :start, :], embedding[:, end:, :]], dim=1)
        elif raw_len < SEQ_LEN:
            padding = torch.zeros(1, SEQ_LEN - raw_len, 640, device=self.device)
            embedding = torch.cat([embedding, padding], dim=1)
        else:
            pass  # exact match

        # 3. 模型推理
        out = self.model(embedding)

        # 4. 解析输出
        result = {"model_version": "v1"}

        if self.is_dual:
            loc_logits, mem_logits = out
            loc_probs = F.softmax(loc_logits, dim=1).cpu().numpy()[0]
            mem_probs = F.softmax(mem_logits, dim=1).cpu().numpy()[0]

            loc_pred = int(torch.argmax(loc_logits, dim=1).item())
            mem_pred = int(torch.argmax(mem_logits, dim=1).item())

            loc_key = LOCATION_CLASSES[loc_pred]
            mem_key = MEMBRANE_CLASSES[mem_pred]

            result["predicted_location"] = LOCATION_LABEL_MAP.get(loc_key, loc_key)
            result["location_confidence"] = round(float(loc_probs[loc_pred]), 6)
            result["predicted_membrane"] = MEMBRANE_LABEL_MAP.get(mem_key, mem_key)
            result["membrane_confidence"] = round(float(mem_probs[mem_pred]), 6)

            result["all_probabilities"] = {
                LOCATION_LABEL_MAP.get(k, k): round(float(p), 6)
                for k, p in zip(LOCATION_CLASSES, loc_probs)
            }
        else:
            loc_logits = out
            loc_probs = F.softmax(loc_logits, dim=1).cpu().numpy()[0]
            loc_pred = int(torch.argmax(loc_logits, dim=1).item())
            loc_key = LOCATION_CLASSES[loc_pred]

            result["predicted_location"] = LOCATION_LABEL_MAP.get(loc_key, loc_key)
            result["location_confidence"] = round(float(loc_probs[loc_pred]), 6)
            result["predicted_membrane"] = None
            result["membrane_confidence"] = None

            result["all_probabilities"] = {
                LOCATION_LABEL_MAP.get(k, k): round(float(p), 6)
                for k, p in zip(LOCATION_CLASSES, loc_probs)
            }

        elapsed = int((time.time() - t_start) * 1000)
        result["inference_time_ms"] = elapsed

        return result


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(description="Protein subcellular localization inference")
    parser.add_argument("--sequence", required=True, help="Amino acid sequence")
    parser.add_argument("--model-path", default="best_model.pt",
                        help="Path to trained model checkpoint")
    args = parser.parse_args()

    predictor = Predictor(args.model_path)
    result = predictor.predict(args.sequence)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
