import torch
from transformers import BertTokenizer, BertForMaskedLM

#слова «работает» и «идёт»
# Загрузка предобученной модели BERT и токенизатора
model_name = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

def get_top_predictions(sentence, target_words, top_k=10):
    # Токенизация предложения
    inputs = tokenizer(sentence, return_tensors='pt')
    mask_token_index = torch.where(inputs.input_ids == tokenizer.mask_token_id)[1].item()

    # Получение предсказаний
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Извлечение top_k наиболее вероятных слов для [MASK]
    mask_logits = logits[0, mask_token_index, :]
    top_k_ids = torch.topk(mask_logits, top_k).indices.tolist()
    top_k_words = [tokenizer.decode([word_id]).strip() for word_id in top_k_ids]

    # Выводим результаты
    print(f"Топ {top_k} предсказанных слов для [MASK]:")
    print(top_k_words)

    # Проверяем, содержатся ли целевые слова в предсказаниях
    for target_word in target_words:
        if target_word in top_k_words:
            print(f"Целевое слово '{target_word}' содержится в топ-{top_k} предсказаниях.")
        else:
            print(f"Целевое слово '{target_word}' не содержится в топ-{top_k} предсказаниях.")

def main():
    sentence = "Эта программа [MASK] в любое время."

    target_words = ["работает", "идёт"]

    # Запуск предсказания
    get_top_predictions(sentence, target_words=target_words, top_k=10)

if __name__ == "__main__":
    main()
