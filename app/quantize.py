import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

# 1. Load your trained model
model_path = "./final_model3"  # Your current model folder
print(f"Loading model from {model_path}...")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# 2. Apply Dynamic Quantization
print("Quantizing model...")
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # Quantize only linear layers (safe for BERT)
    dtype=torch.qint8
)

# 3. Save the Quantized Model
save_path = "./quantized_model"
os.makedirs(save_path, exist_ok=True)

# Save state dict (weights)
torch.save(quantized_model.state_dict(), os.path.join(save_path, "quantized_bert.pth"))
# Save config and tokenizer so we can reload it easily
model.config.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Success! Quantized model saved to {save_path}")
print(f"Original Size: {os.path.getsize(os.path.join(model_path, 'pytorch_model.bin')) / 1e6:.2f} MB")
print(f"Quantized Size: {os.path.getsize(os.path.join(save_path, 'quantized_bert.pth')) / 1e6:.2f} MB")