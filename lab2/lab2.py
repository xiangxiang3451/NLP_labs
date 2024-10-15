import gensim
import re
#слова: спад,бум
def main():
    # Определение списка позитивных слов
    pos = ["подорожание_NOUN", "ажиотаж_NOUN"]
    neg = []  # Список негативных слов пустой

    # Загрузка модели cbow
    word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)


    # Получение 10 наиболее похожих слов на линейную комбинацию
    dist = word2vec.most_similar(positive=pos, topn=10)

    # Использование регулярных выражений для извлечения слов в формате "_NOUN"
    pat = re.compile("(.*)_NOUN")

    # Вывод слов, которые соответствуют формату "_NOUN"
    for i in dist:
        e = pat.match(i[0])
        if e is not None:
            print(e.group(1))

if __name__ == '__main__':
    main()