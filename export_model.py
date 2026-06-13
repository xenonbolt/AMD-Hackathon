import platform
# Monkeypatch platform.win32_ver to bypass Windows WMI query errors/hangs
platform.win32_ver = lambda release='', version='', csd='', ptype='': ('10', '10.0.22621', '', 'Multiprocessor Free')

import argparse
import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("export_model")

def export_and_merge(model_id: str, adapter_path: str, output_dir: str):
    """
    Loads base model in FP16/FP32 precision (unquantized), loads the PEFT adapter,
    merges the adapter weights, and saves the self-contained merged model offline.
    """
    logger.info(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # Determine dtype (bfloat16 or float16/float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32
        
    logger.info(f"Loading base model on {device} with dtype: {torch_dtype}")
    # Note: We must NOT load the model in 4-bit/8-bit quantization because we cannot merge weights in QLoRA directly.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    logger.info(f"Overlaying PEFT adapter from path: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging adapter weights into the base model...")
    # This combines the PEFT parameters back into the base model architecture weights
    merged_model = model.merge_and_unload()
    
    logger.info(f"Saving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Model export and merge successfully completed!")
    logger.info(f"The offline self-contained Hugging Face model folder is saved at: {os.path.abspath(output_dir)}")
    logger.info("You can load it completely offline using:")
    logger.info(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}', local_files_only=True)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge PEFT Adapters into Base Model and Save Offline")
    parser.add_argument("--model_id", type=str, required=True, help="Base model identifier (e.g. bigcode/starcoder2-3b)")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the trained LoRA adapter directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the merged self-contained model")
    
    args = parser.parse_args()
    export_and_merge(args.model_id, args.adapter_path, args.output_dir)


