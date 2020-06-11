
from nltk.corpus import wordnet as wn

#nltk.download('wordnet')

# 특정 단어의 최상위 노드까지의 경로
def hypernyms(word):
    current_node = wn.synsets(word)[0]
    yield current_node

    while True:
        try:
            current_node = current_node.hypernyms()[0]
            yield current_node
        except IndexError:
            break
def get_hypernyms(word):
    for h in hypernyms(word):
        print(h)

#get_hypernyms('policeman')

get_hypernyms('student')