# chat_inference.py
import torch
from transformers import AutoTokenizer
from training_script import OPTForCausalLM
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "trained_model"

# تحميل النموذج والتوكنيزر
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = OPTForCausalLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=768,
    num_heads=12,
    num_layers=6,
    ff_dim=3072,
    max_len=128
).to(device)
model.load_state_dict(torch.load(os.path.join(model_dir, "pytorch_model.bin")))
model.eval()

# الدالة الرئيسية للدردشة مع حفظ السياق
def chat_loop():
    print("ابدأ الدردشة. اكتب 'خروج' لإنهاء الجلسة.")
    context = ""  # لحفظ سياق المحادثة
    while True:
        user_input = input("أنت: ")
        if user_input.strip().lower() in ["خروج", "exit", "quit"]:
            break

        # تحديث السياق مع إدخال المستخدم
        context += f"\nأنت: {user_input}\nروبوت:"

        input_ids = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            for _ in range(50):
                output = model(input_ids)
                logits = output[:, -1, :]
                next_token = torch.argmax(logits, dim=-1).unsqueeze(0)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break

        response = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        # استخراج الرد الأخير فقط
        bot_reply = response.split("روبوت:")[-1].strip().split("أنت:")[0].strip()

        print("روبوت:", bot_reply)
        context += f" {bot_reply}"

if __name__ == "__main__":
    chat_loop()
