import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np





def main():
    text =  ("<b>Lorem ipsum dolor :)           <br> sit amet,5 consectetur; adipiscing elit. Sed 3 eget mattis sem. ;)"
            "Mauris ;(       egestas erat quam, <div>           6:< ut faucibus eros congue :> et. <a>In blandit, 0 mi eu porta;"
            "lobortis, tortor :-)<a> 32 nisl facilisis leo, at ;< tristique       augue risus 69 eu risus ;-).")

    print(allClearing(text))

def allClearing(text: str) -> str:
    text = removeEmocji(text)
    text = changeLower(text)
    text = removeNumber(text)
    text = removeHtml(text)
    text = removeSpaces(text)
    return text

def removeEmocji(text: str) -> str:
    emotkiwTexcie = re.findall('([:;]+[)(<-]+)',text)
    text = re.sub('([:;]+[)(<-]+)','',text)
    listTostr = ''.join([str(elem) for elem in emotkiwTexcie])
    textComine = text + listTostr
    return textComine

def removeHtml(text: str) -> str:
   text = re.sub('<[^>]*>','',text)
   return text

def changeLower(text: str) -> str:
    text = re.sub('[A-Z]','',text.lower())
    return text

def removeNumber(text: str) -> str:
    text = re.sub('\d','',text)
    return text

def removeIP(text: str) -> str:
    text = re.sub('[^\w\s]','',text)
    return text

def removeSpaces(text: str) -> str:
    text = re.sub('\s ','',text)
    return text

def remove_stop_words(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    list_of_words = text
    return [word for word in list_of_words if word not in stop_words]


def steming(text: list) -> str:
    porter = PorterStemmer()
    return [porter.stem(word) for word in text]


def bag_of_words(words: list)-> dict:
    bow = {}
    for word in words:
        if word not in bow.keys():
            bow[word] = 1
        else:
            bow[word] += 1
    return  bow

def text_tokenizer(text: str)-> list:
    text_Comb = text
    text_Comb = allClearing(text_Comb)
    text_Comb_list = text_Comb.split(" ")
    text_Comb_list = steming(text_Comb_list)
    text_Comb_list = remove_stop_words(text_Comb_list)

    return[word for word in text_Comb_list if len(word)>3]


def top_tokens(list_of_tokens: list, token_words: list, how_many: int = 10) -> list:
    working_list = list_of_tokens.copy()
    result = []
    for i in range(how_many):
        token_index = np.argmax(working_list)
        result.append(token_words[token_index])
        working_list[token_index] = 0
    return result


def top_documents(list_of_documents: list, how_many: int = 10) -> list:
    working_list = list_of_documents.copy()
    result = []
    for i in range(how_many):
        token_index = np.argmax(working_list)
        result.append(token_index)
        working_list[token_index] = 0
    return result

if __name__ == "__main__":
    main()