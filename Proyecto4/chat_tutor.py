import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings

# --- CONFIGURACIÓN ---
base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
lora_model_name = "qwen_tutor_algoritmos_v2"
device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.filterwarnings("ignore", category=UserWarning)

print(f"Cargando el Tutor Inteligente en {device}...")
print("Espere unos segundos...")

# 1. Cargar Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2. Cargar Modelo Base
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Cargar LoRA (modelo entrenado)
model = PeftModel.from_pretrained(base_model, lora_model_name)
model.eval()

print("\n" + "=" * 50)
print("Qwen Tutor de Algoritmos")
print(" Escribe 'salir' para terminar.")
print("=" * 50 + "\n")


# -------------------------------------------------
# FUNCIÓN CLAVE: LIMPIEZA Y CORTE DE RESPUESTA
# -------------------------------------------------
def limpiar_respuesta(texto):
    texto = texto.strip()


    if "\n" in texto:
        texto = texto.split("\n")[0]


    if "." in texto:
        texto = texto.split(".")[0].strip() + "."

    return texto


def generar_respuesta(pregunta):
    prompt = f"### Pregunta:\n{pregunta}\n\n### Respuesta esperada:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=30,     
            do_sample=False,       
            num_beams=1,
            repetition_penalty=1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    texto = tokenizer.decode(out[0], skip_special_tokens=True)


    respuesta = texto.split("### Respuesta esperada:")[-1].strip()


    respuesta = limpiar_respuesta(respuesta)

    return respuesta


# --- BUCLE INTERACTIVO ---
while True:
    try:
        usuario = input(">> Pregunta: ")

        if usuario.lower() in ["salir", "exit", "quit"]:
            print("Guardando sesión... ¡Adiós!")
            break

        if not usuario.strip():
            continue

        print("Pensando...", end="\r")
        respuesta = generar_respuesta(usuario)

        print(f" Qwen: {respuesta}\n")

    except KeyboardInterrupt:
        print("\nSaliendo...")
        break
    except Exception as e:
        print(f"Error crítico: {e}")
