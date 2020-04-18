import os
import re
import string
import xml.etree.ElementTree as ET
import  pandas as pd
import numpy as np
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from nltk import word_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser


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


def get_training_statistic():
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')

    types = []
    drugs_interact = []
    sentences = []
    distance = []
    count_words_between = []
    # Process each file in the directory
    for filename in os.listdir(input_directory):
        # Parse XML file
        root = parse_xml(input_directory + filename)
        print(" - File:", filename)

        for child in root:
            sid, text = get_sentence_info(child)
            entities = {}
            if not text:
                continue
            for entity in child.findall('entity'):
                id = entity.get('id')
                offset = entity.get('charOffset')
                if ';' in offset:
                    offset = offset.split(";")
                else:
                    offset = [offset]
                ent_offset = []
                for off in offset:
                    ent_offset.append(tuple([int(i) for i in off.split("-")]))
                entities[id] = np.asarray(ent_offset)

            for pair in child.findall('pair'):
                id_e1 = pair.get('e1')
                id_e2 = pair.get('e2')
                ddi = pair.get('ddi')
                type = pair.get('type')

                if ddi == "true":
                    types.append(type)
                    print(entities[id_e1])
                    print(entities[id_e2])
                    offset_1 = entities[id_e1][0]
                    offset_2 = entities[id_e2][0]
                    drug_1 = drug_2 = ""
                    end = 0
                    start = offset_1[0]
                    if len(offset_1) > 2:
                        drug_1 = text[offset_1[0]:offset_1[1]+1] + text[offset_1[2]:offset_1[3]+1]
                    else:
                        drug_1 = text[offset_1[0]:offset_1[1] + 1]
                    if len(offset_2) > 2:
                        drug_2 = text[offset_2[0]:offset_2[1]+1] + text[offset_2[2]:offset_2[3]+1]
                        end = offset_2[3]
                    else:
                        drug_2 = text[offset_2[0]:offset_2[1] + 1]
                        end = offset_2[1]
                    drugs_interact.append((drug_1, drug_2))
                    distance.append(end-start)
                    sentence = text[start:end]
                    count_words_between.append(len(sentence.split()) - 2)
                    sentences.append(sentence)


    df = pd.DataFrame(list(zip(types, drugs_interact, sentences, distance, count_words_between)),
                      columns=['Type', 'Drug_Interact', 'Sentence', 'Distance', 'CountWordsBetween'])

    df.to_csv('analysis.csv', index=False)

def add_offset_to_tree(parse):
    word_count = 0
    for key in range(len(parse.nodes)):
        value = parse.nodes[key]

        if value['word'] and value['rel'] != 'punct':
            parse.nodes[key]['start_off'] = word_count
            parse.nodes[key]['end_off'] = len(value['word']) - 1 + word_count
            word_count += len(value['word']) + 1

        elif value['rel'] == 'punct':
            parse.nodes[key]['start_off'] = word_count
            parse.nodes[key]['end_off'] = word_count + 1
            word_count += 1

    return parse


def analyze(stext):
    parser = CoreNLPDependencyParser(url="http://localhost:9000")

    iterator = parser.raw_parse(stext)
    parse = next(iterator)

    parse = add_offset_to_tree(parse)
    return parse


def check_dependency(analysis, tokens_info, e2_start_off):
    # Deps_types: effect, int, mechanis, advice and null

    for token_info in tokens_info:
        if 'start_off' in analysis.nodes[token_info['head']] and \
                analysis.nodes[token_info['head']]['start_off'] == e2_start_off:
            return 1, 'head'
        for dep_rel, value in token_info['deps'].items():
            for dependency in value:
                if 'start_off' in analysis.nodes[dependency] and \
                        analysis.nodes[dependency]['start_off'] == e2_start_off:
                    return 1, 'deps'
    return 0, 'null'


def check_interaction(analysis, entities, id_e1, id_e2):
    e1_off = entities[id_e1]
    e2_off = entities[id_e2]

    is_ddi = 0
    ddi_type = 'null'

    for word_index in range(len(analysis.nodes)):
        token_info = analysis.nodes[word_index]
        tokens_info = []

        for (e1_start_off, e1_end_off), (e2_start_off, e2_end_off) in zip(e1_off, e2_off):
            if 'start_off' in token_info and token_info['start_off'] == e1_start_off:
                tokens_info.append(token_info)
                while token_info['end_off'] < e1_end_off:
                    word_index += 1
                    token_info = analysis.nodes[word_index]
                    tokens_info.append(token_info)

                is_ddi, ddi_type = check_dependency(analysis, tokens_info, e2_start_off)

            elif 'start_off' in token_info and token_info['start_off'] == e2_start_off:
                tokens_info.append(token_info)
                while token_info['end_off'] < e2_end_off:
                    word_index += 1
                    token_info = analysis.nodes[word_index]
                    tokens_info.append(token_info)

                is_ddi, ddi_type = check_dependency(analysis, tokens_info, e1_start_off)

        if is_ddi:
            return is_ddi, ddi_type

    return 0, 'null'


if __name__ == '__main__':
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')
    get_training_statistic()


    # Process each file in the directory
    for filename in os.listdir(input_directory):
        # Parse XML file
        root = parse_xml(input_directory + filename)
        print(" - File:", filename)

        for child in root:
            sid, text = get_sentence_info(child)
            entities = {}
            if not text:
                continue
            for entity in child.findall('entity'):
                id = entity.get('id')
                offset = entity.get('charOffset')
                if ';' in offset:
                    offset = offset.split(";")
                else:
                    offset = [offset]
                ent_offset = []
                for off in offset:
                    ent_offset.append(tuple([int(i) for i in off.split("-")]))
                entities[id] = np.asarray(ent_offset)

            analysis = analyze(text)
            # token_list = chem_tokenize(text)

            for pair in child.findall('pair'):
                id_e1 = pair.get('e1')
                id_e2 = pair.get('e2')

                # TODO: Add rules in Check interaction
                (is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2)

                print("|".join([sid, id_e1, id_e2, str(is_ddi), ddi_type]), file=output_file)

            # output_entities(sid, entities, output_file)

    # Close the file
    output_file.close()
    print(evaluate(input_directory, output_file_name))
