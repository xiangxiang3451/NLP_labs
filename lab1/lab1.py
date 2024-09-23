import pymorphy3
import re

# Инициализация морфологического анализатора
morph = pymorphy3.MorphAnalyzer()

def tokenize(text):
    """Разделяет текст на предложения и слова."""
    sentences = re.split(r'[.!?]', text)  # Разделение по предложениям
    words_in_sentences = [re.findall(r'\w+', sentence) for sentence in sentences]  # Извлечение слов
    return words_in_sentences  # Возвращает список слов, разделённых по предложениям

def is_adjective_or_noun(word):
    """Проверяет, является ли слово существительным или прилагательным."""
    parsed_word = morph.parse(word)[0]  # Парсинг слова
    # Проверка, является ли тег слова существительным или прилагательным
    return 'NOUN' in parsed_word.tag or 'ADJF' in parsed_word.tag or 'ADJS' in parsed_word.tag

def check_gender_number_case(word1, word2):
    """Проверяет, совпадают ли род, число и падеж двух слов."""
    word1_parsed = morph.parse(word1)[0].tag  # Парсинг первого слова
    word2_parsed = morph.parse(word2)[0].tag  # Парсинг второго слова
    # Возвращает булево значение, указывающее, равны ли род, число и падеж
    return (word1_parsed.gender == word2_parsed.gender and
            word1_parsed.number == word2_parsed.number and
            word1_parsed.case == word2_parsed.case)

def extract_adjective_noun_pairs(text):
    """Извлекает пары прилагательных и существительных из текста и печатает их."""
    all_sentences = tokenize(text)  # Разделение текста на предложения

    for words in all_sentences:
        if not words:  # Пропуск пустых предложений
            continue

        lemmas = []  # Список для хранения основ существительных и прилагательных
        for word in words:
            if is_adjective_or_noun(word):  # Проверка, является ли слово существительным или прилагательным
                lemmas.append(morph.parse(word)[0].normal_form)  # Добавление основы

        matching_pairs = []  # Список для хранения совпадающих пар прилагательных и существительных
        for i in range(len(lemmas) - 1):
            current_word = lemmas[i]  # Текущее слово
            next_word = lemmas[i + 1]  # Следующее слово
            # Проверка, являются ли текущее и следующее слова существительными или прилагательными, и совпадают ли род, число и падеж
            if (is_adjective_or_noun(current_word) and
                    is_adjective_or_noun(next_word) and
                    check_gender_number_case(current_word, next_word)):
                matching_pairs.append((current_word, next_word))  # Добавление пары

        for pair in matching_pairs:
            print(" ".join(pair))  # Печать пары

        print()  # Разделение вывода предложений

def main():
    """Основная функция для загрузки текста и извлечения пар прилагательных и существительных."""
    filename = 'lab1.txt'  # Указание имени текстового файла
    with open(filename, 'r', encoding='utf-8') as f:  # Открытие файла с кодировкой utf-8
        text = f.read()  # Чтение содержимого файла

    extract_adjective_noun_pairs(text)  # Извлечение пар прилагательных и существительных

if __name__ == "__main__":
    main()  # Выполнение основной функции
