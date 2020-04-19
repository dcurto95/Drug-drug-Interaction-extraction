import os
import re
import string
import xml.etree.ElementTree as ET

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


def add_offset_to_tree(parse, text, offset=0):
    for key in range(len(parse.nodes)):
        value = parse.nodes[key]

        if value['word'] and value['rel'] != 'punct':
            start = text.find(value['word'], offset)
            parse.nodes[key]['start_off'] = start
            if len(value['word']) > 1:
                parse.nodes[key]['end_off'] = len(value['word']) - 1 + start
            else:
                parse.nodes[key]['end_off'] = len(value['word']) + start
            offset = start + len(value['word'])

        elif value['rel'] == 'punct':
            parse.nodes[key]['start_off'] = offset
            parse.nodes[key]['end_off'] = offset + 1
            offset += 1

    return parse


def analyze(stext):
    parser = CoreNLPDependencyParser(url="http://localhost:9000")

    if '\r\n' in stext:
        stext = stext.replace('\r\n', '  ')
    iterator = parser.raw_parse(stext)
    parse = next(iterator)

    parse = add_offset_to_tree(parse, stext)

    return parse


def basic_rules(word, lemma):
    advise_list = ['can', 'could', 'may', 'might', 'will', 'shall', 'should', 'ought', 'must', 'would']
    advise_string = ' '.join(advise_list)
    effect_list = ['administer', 'potentiate', 'prevent', 'effect', 'cause']
    effect_string = ' '.join(effect_list)
    mechanism_list = ['reduce', 'increase', 'decrease']
    mechanism_string = ' '.join(mechanism_list)
    int_list = ['interact', 'interaction', 'interfere']
    int_string = ' '.join(int_list)

    if word in int_list:
        return 'int'
    if word in advise_list:
        return 'advise'
    if word in effect_list:
        return 'effect'
    if word in mechanism_list:
        return 'mechanism'
    return None


def check_dependency(analysis, tokens_info, e2_start_off, first_index, second_index, truth_ddi, dic, good_dic):
    # Deps_types: effect, int, mechanis, advise and null
    if truth_ddi == 'false':
        dic[second_index - first_index] = dic.get(second_index - first_index, 0) + 1
    else:
        good_dic[second_index - first_index] = good_dic.get(second_index - first_index, 0) + 1

    for token_info in tokens_info:
        for index in range(len(analysis.nodes)):
            if index < first_index:
                # Search before
                # category = basic_rules(analysis.nodes[index]['word'])
                # if category is not None:
                #     return 1, category
                pass

            if first_index <= index <= second_index:
                # Search between words
                category = basic_rules(analysis.nodes[index]['word'].lower(), analysis.nodes[index]['lemma'])
                if category is not None:
                    return 1, category
            if index > second_index:
                # Search after words
                # category = basic_rules(analysis.nodes[index]['word'])
                # if category is not None:
                #     return 1, category
                pass

        if 'start_off' in analysis.nodes[token_info['head']] and \
                analysis.nodes[token_info['head']]['start_off'] == e2_start_off:
            return 0, 'null'
        for dep_rel, value in token_info['deps'].items():
            for dependency in value:
                if 'start_off' in analysis.nodes[dependency] and \
                        analysis.nodes[dependency]['start_off'] == e2_start_off:
                    return 0, 'null'
    return 0, 'null'


def find_second_entity(analysis, word_index, e2_start_off):
    index = word_index

    while index < len(analysis.nodes):
        if 'start_off' in analysis.nodes[index] and (analysis.nodes[index]['start_off'] == e2_start_off or (
                analysis.nodes[index]['start_off'] < e2_start_off <= analysis.nodes[index]['end_off'])):
            return index
        index += 1
    raise Exception("Entity not found")


def check_interaction(analysis, entities, id_e1, id_e2, truth_ddi, dic, good_dic):
    e1_off = entities[id_e1]
    e2_off = entities[id_e2]

    for word_index in range(len(analysis.nodes)):
        token_info = analysis.nodes[word_index]
        entity_list_tokens = []

        for (e1_start_off, e1_end_off), (e2_start_off, e2_end_off) in zip(e1_off, e2_off):
            if 'start_off' in token_info and (token_info['start_off'] == e1_start_off or (
                    token_info['start_off'] < e1_start_off <= token_info['end_off'])):
                # Start offset matches or start offset is inside token
                entity_list_tokens.append(token_info)
                aux_word_index = word_index
                # If entity is longer than token add to token list
                while token_info['end_off'] < e1_end_off:
                    aux_word_index += 1
                    token_info = analysis.nodes[aux_word_index]
                    entity_list_tokens.append(token_info)
                second_index = find_second_entity(analysis, word_index, e2_start_off)
                return check_dependency(analysis, entity_list_tokens, e2_start_off, word_index, second_index, truth_ddi,
                                        dic, good_dic)

    return 0, 'null'


if __name__ == '__main__':
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')

    dic = {}
    good_dic = {}

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
                (is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2, pair.get('ddi'), dic, good_dic)

                print("|".join([sid, id_e1, id_e2, str(is_ddi), ddi_type]), file=output_file)

            # output_entities(sid, entities, output_file)
    dic = {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}
    print("DICT:", dic)
    good_dic = {k: v for k, v in sorted(good_dic.items(), key=lambda item: item[1], reverse=True)}
    print("Good DICT:", good_dic)
    # Close the file
    output_file.close()
    print(evaluate(input_directory, output_file_name))
