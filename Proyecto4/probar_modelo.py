import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# --- CONFIGURACIÓN ---
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "qwen_tutor_algoritmos_v2" # La carpeta donde se guardó el entrenamiento

print("1. Cargando el modelo base...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32, # Importante para el CPU
    device_map="cpu"
)

print("2. Cargando tus aprendizajes (LoRA)...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model = model.merge_and_unload() # Fusionamos para que sea un solo modelo rápido
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

def preguntar(pregunta):
    # Usamos EXACTAMENTE el mismo formato con el que entrenamos
    # Si cambiamos esto, el modelo se confunde.
    prompt = f"### Pregunta:\n{pregunta}\n\n### Respuesta esperada:\n"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("\nGenerando respuesta...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,    # Longitud de la respuesta
            temperature=0.7,       # Creatividad (0.7 es equilibrado)
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1 # Para que no repita frases como loro
        )
    
    respuesta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Limpieza para mostrar solo la parte nueva
    respuesta_limpia = respuesta_completa.split("### Respuesta esperada:\n")[-1]
    
    print("-" * 30)
    print(f"TUTOR: {respuesta_limpia.strip()}")
    print("-" * 30)

# Bucle de chat
print("\nTutor Qwen cargado. Escribe 'salir' para terminar.")
while True:
    user_input = input("\nTú: ")
    if user_input.lower() in ['salir', 'exit', 'quit', 'bye', 'adiós', 'huir', 'terminar']:
        break
    preguntar(user_input)