from huggingface_hub import login

print("🔒 Iniciando sesión en Hugging Face...")
token = input("👉 Pega tu token (hf_...) y dale Enter: ")

login(token=token)
print("✅ ¡Login exitoso! Ya puedes entrenar.")