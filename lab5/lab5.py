import sys
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Установка случайного seed для обеспечения воспроизводимости результатов
np.random.seed(42)
torch.manual_seed(42)

# Загрузка модели GPT-2 и токенизатора
model_name = "sberbank-ai/rugpt3large_based_on_gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Функция генерации текста
def generate(
        model, tok, text,
        do_sample=True, max_length=100, repetition_penalty=5.0,
        top_k=50, top_p=0.95, temperature=1,
        num_beams=None,
        no_repeat_ngram_size=3
):
    # Кодирование входного текста в формат, подходящий для модели
    input_ids = tok.encode(text, return_tensors="pt")

    # Генерация текста с использованием модели
    out = model.generate(
        input_ids,
        max_length=max_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_k=top_k, top_p=top_p, temperature=temperature,
        num_beams=num_beams, no_repeat_ngram_size=no_repeat_ngram_size
    )
    # Декодирование сгенерированного текста и возврат результата
    return list(map(tok.decode, out))

# Функция для создания текста-подсказки
def create_prompt():
  prompt = ( "Когда туристы приезжают в город, они часто решают, как лучше всего передвигаться по городу и найти лучшие места для посещения. "
        "Некоторые предпочитают пользоваться различными транспортными услугами, чтобы быстрее добраться до достопримечательностей. "
        "В городе также есть много интересных мест, которые туристы могут увидеть. В конечном итоге, "
)
  return prompt

def main():
    # Создание текста-подсказки
    prompt = create_prompt()

    # Генерация текста
    generated_text = generate(
        model=model, tok=tokenizer, text=prompt,
        max_length=800,  # Увеличение длины сгенерированного текста
        repetition_penalty=2.0, top_k=50, top_p=0.95, temperature=0.7
    )

    # Вывод сгенерированного текста
    print("Сгенерированный текст:")
    print(generated_text[0])

# Вызов основной функции
if __name__ == "__main__":
    main()
