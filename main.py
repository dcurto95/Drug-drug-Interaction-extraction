import os
import re
import string
import xml.etree.ElementTree as ET
from nltk.parse.corenlp import CoreNLPDependencyParser
from collections import Counter
import pandas as pd
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


def offset_to_int(offset):
    start = [int(i) for i in offset[0]]
    end = [int(i) for i in offset[1]]

    return start, end

def get_training_statistic():
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')


    types = []
    drugs_interact = []
    sentences = []
    distance = []
    #Process each file in the directory
    for filename in os.listdir(input_directory):
        #Parse XML file
        print(filename)
        root = parse_xml(input_directory + filename)
        for child in root:
            sid, text = get_sentence_info(child)
            entities = {}
            for entity in child.findall('entity'):
                id = entity.get('id')
                char_offset = entity.get('charOffset').split(";")
                offset = []
                for i in char_offset:
                    splited_offset = i.split("-")
                    splited_offset = [int(of) for of in splited_offset]
                    offset.append(splited_offset)
                entities[id] = offset
            for pair in child.findall('pair'):
                id_e1 = pair.get('e1')
                id_e2 = pair.get('e2')
                ddi = pair.get('ddi')
                type = pair.get('type')

                if ddi == "true":
                    types.append(type)
                    print(entities[id_e1])
                    print(entities[id_e2])
                    offset_1 = offset_to_int(entities[id_e1])
                    offset_2 = offset_to_int(entities[id_e2])
                    drug_1 = text[offset_1[0]:offset_1[1]+1]
                    drug_2 = text[offset_2[0]:offset_2[1]+1]
                    drugs_interact.append((drug_1, drug_2))
                    distance.append(offset_2[1] - offset_1[0])
                    sentence = text[offset_1[0]:offset_2[1]]
                    sentences.append(sentence)

    df = pd.DataFrame(list(zip(types, drugs_interact, sentences, distance)),
                      columns=['Type', 'Drug_Interact', 'Sentence', 'Distance'])

    df.to_csv('analysis.csv', index=False)

if __name__ == '__main__':
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')

    get_training_statistic()
    #my_parser = CoreNLPDependencyParser(url="http://localhost:9000")
    #my_tree, = my_parser.raw_parse("Hello, my name is David")

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
