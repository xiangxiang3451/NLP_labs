import re
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
import pymorphy3

# nltk.download('punkt')  # Загрузка токенизатора (раскомментируйте при первом запуске)

def main():
    # Чтение файла
    with open('lab1.txt', 'r', encoding='utf-8') as file:
        text = file.read()

    # Токенизация текста
    words = word_tokenize(text)

    # Создание морфологического анализатора
    morphy = pymorphy3.MorphAnalyzer()
    result = []
    left_word = {'word': '', 'POS': '', 'gender': '', 'number': '', 'case': '', 'idx': -2}

    # Обработка каждого слова
    for i in tqdm(range(len(words))):
        m = morphy.parse(words[i])[0]  # Получение морфологического разбора слова
        tag = m.tag
        # Проверка на существительное или прилагательное
        if tag.POS in ['NOUN', 'ADJF']:
            # Условие на совпадение форм слов
            if (i - left_word['idx'] == 1 and left_word['gender'] == tag.gender and
                left_word['number'] == tag.number and left_word['case'] == tag.case and
                left_word['POS'] != tag.POS):
                result.append(left_word['word'] + ' ' + m.normal_form)  # Сохранение совпадающих слов
            left_word['idx'] = i  # Обновление индекса текущего слова
            left_word['word'] = m.normal_form  # Сохранение нормальной формы слова
            left_word['POS'] = tag.POS  # Сохранение части речи
            left_word['gender'] = tag.gender  # Сохранение рода
            left_word['number'] = tag.number  # Сохранение числа
            left_word['case'] = tag.case  # Сохранение падежа

    # Удаление дубликатов
    result1 = list(set(result))

    # Вывод результата
    print(result1)

if __name__ == "__main__":
    main()
