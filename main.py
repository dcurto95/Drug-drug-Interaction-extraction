import os
import re
import string
import xml.etree.ElementTree as ET
from nltk.parse.corenlp import CoreNLPDependencyParser
from collections import Counter

import nltk
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from nltk import word_tokenize, QuadgramCollocationFinder
from nltk.corpus import stopwords


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()

def get_sentence_info(child):
    return child.get('id'), child.get('text')

def get_sentence_entities_info(child):
    return child.get('id'), (child.get('charOffset')).split("-")


def chem_tokenize(text):
    cwt = ChemWordTokenizer()
    tokens = cwt.tokenize(text)
    token_indexs = cwt.span_tokenize(text)
    tokenized_info = []
    for token_index, token in zip(token_indexs, tokens):
        tokenized_info.append((token, token_index[0], token_index[1] - 1))
    return tokenized_info


def tokenize(text):
    tokenized_sent = word_tokenize(text)
    tokenized_info = []
    current_index = 0

    for word in tokenized_sent:

        if not re.match("[" + string.punctuation + "]", word):
            for match in re.finditer(word, text):
                if match.start() >= current_index:
                    tokenized_info.append((word, match.start(), match.end() - 1))
                    current_index = match.end() - 1
                    break
    return tokenized_info


def evaluate(inputdir, outputfile):
    return os.system("java -jar ../eval/evaluateDDI.jar " + inputdir + " ../output/" + outputfile)


if __name__ == '__main__':
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')

    my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    my_tree, = my_parser.raw_parse("Hello, my name is David")

    #Process each file in the directory
    for filename in os.listdir(input_directory):
        #Parse XML file
        root = parse_xml(input_directory + filename)
        print(" - File:", filename)

        for child in root:
            sid, text = get_sentence_info(child)
            entities = {}

            for entity in child.findall('entity'):
                id = entity.get('id')
                offset = (entity.get('charOffset')).split("-")
                entities[id] = offset

            #TODO: Tokenize, tag and parse sentence
            #analysis = analyze(stext)
            token_list = chem_tokenize(text)

            for pair in child.findall('pair'):
                id_e1 = pair.get('e1')
                id_e2 = pair.get('e2')
                #TODO: Check interaction
                #(is_ddi,ddi_type) = check_interaction(analysis, entities, id_e1, id_e2)


            #output_entities(sid, entities, output_file)

    # Close the file
    output_file.close()
    print(evaluate(input_directory, output_file_name))
