import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# Limpieza inicial
torch.cuda.empty_cache()

print("1. Cargando modelo y tokenizer...")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
new_model_name = "qwen_tutor_algoritmos_v2"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" 

# Modelo (CPU + float32)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32, 
    device_map="cpu"
)

# Datos
dataset = load_dataset(
    "json",
    data_files="tutor_programacion.jsonl",
    split="train"
)

# --- CAMBIO CRÍTICO: FORMATEO MANUAL ---
# En lugar de una función compleja para el trainer, formateamos el dataset AHORA.
def apply_chat_template(example):
    # Recuperamos los campos con seguridad
    instruction = example.get("instruction", example.get("prompt", ""))
    response = example.get("response", "")
    
    # Creamos el texto final EXACTAMENTE como lo quiere el modelo
    text = (
        f"### Pregunta:\n{instruction}\n\n"
        f"### Respuesta esperada:\n{response}{tokenizer.eos_token}"
    )
    # Devolvemos un diccionario con la columna 'text' ya lista
    return {"text": text}

print("Aplicando formato al dataset manualmente...")
# Aplicamos la función fila por fila (batched=False para evitar líos de listas)
dataset = dataset.map(apply_chat_template)

# --- CONFIGURACIÓN ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,   
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
    output_dir="./resultados_qwen",
    num_train_epochs=11,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,   
    
    # --- AJUSTES CPU/WINDOWS ---
    bf16=False,
    fp16=False,             
    use_cpu=True,           
    logging_steps=1,
    save_strategy="no",
    report_to="none",
    optim="adamw_torch",
    
    # --- ANTI-CONGELAMIENTO ---
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    
    packing=False,
    dataset_text_field="text" # Le decimos: "Usa la columna 'text' que creé arriba"
)

# Inicializar Trainer
# NOTA: Ya no pasamos 'formatting_func' porque ya formateamos el dataset arriba
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer, 
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args
)

print("2. Entrenando... (Paciencia, CPU trabajando)")
trainer.train()

print("3. Guardando modelo...")
trainer.model.save_pretrained(new_model_name)
tokenizer.save_pretrained(new_model_name)

print(" Entrenamiento terminado")