################
# fun command line antonym fetcher using beautiful soup
################


import requests
from bs4 import BeautifulSoup



def get_thesarus_html_soup(word):
    root = 'http://www.thesaurus.com/browse/'

    r = requests.get(url = root + word)
    soup = BeautifulSoup(r.text, 'html.parser')
    
    return soup



def print_antonyms(word):
    soup = get_thesarus_html_soup(word)
    antonyms = soup.find_all('a', {'class':'css-1usnxsl e1s2bo4t1'})
    
    if len(antonyms) > 0:
        print('\nAntonyms for "', word, '":', sep='')
        for antonym in antonyms:
            print(antonym.string)
    else:
        print('\nNo strong antonyms')



if __name__ == '__main__':
    word = input('What word should I fetch antonyms from Thesaurus.com for?\n')
    print_antonyms(word)