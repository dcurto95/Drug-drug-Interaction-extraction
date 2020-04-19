import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from nltk import CoreNLPDependencyParser


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


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()


def get_sentence_info(child):
    return child.get('id'), child.get('text')


def check_interaction(analysis, entities, id_e1, id_e2):
    e1_off = entities[id_e1].get('charOffset')
    e2_off = entities[id_e2].get('charOffset')

    e1_off = split_offset(e1_off)
    e2_off = split_offset(e2_off)

    tokens_info_e1 = []
    tokens_info_e2 = []
    for word_index in range(len(analysis.nodes)):
        token_info = analysis.nodes[word_index]

        for (e1_start_off, e1_end_off), (e2_start_off, e2_end_off) in zip(e1_off, e2_off):
            if 'start_off' in token_info and token_info['start_off'] == e1_start_off:
                tokens_info_e1.append(token_info)
                while token_info['end_off'] < e1_end_off:
                    word_index += 1
                    token_info = analysis.nodes[word_index]
                    tokens_info_e1.append(token_info)

            if 'start_off' in token_info and token_info['start_off'] == e2_start_off:
                tokens_info_e2.append(token_info)
                while token_info['end_off'] < e2_end_off:
                    word_index += 1
                    token_info = analysis.nodes[word_index]
                    tokens_info_e2.append(token_info)

    return tokens_info_e1, tokens_info_e2


def split_offset(offset):
    if ';' in offset:
        offset = offset.split(";")
    else:
        offset = [offset]
    ent_offset = []
    for off in offset:
        ent_offset.append(tuple([int(i) for i in off.split("-")]))

    return np.asarray(ent_offset)


if __name__ == '__main__':
    input_directory = '../data/Train/'
    parser = CoreNLPDependencyParser(url="http://localhost:9000")

    sentences = {}
    all_entities = {}
    all_roots = {}
    all_parse = {}
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
                entities[entity.get('id')] = entity
            sentences[sid] = text
            all_entities[sid] = entities
            iterator = parser.raw_parse(text)
            parse = next(iterator)
            parse = add_offset_to_tree(parse)
            all_parse[sid] = parse
            all_roots[sid] = str(parse.root)
    # Using readlines()
    file1 = open('goldDDI.txt', 'r')
    Lines = file1.readlines()

    truth = {}

    # Strips the newline character
    for line in Lines:
        value = line.split("|")
        if value[0] not in truth:
            truth[value[0]] = []
        tokens_e1, tokens_e2 = check_interaction(all_parse[value[0]], all_entities[value[0]], value[1], value[2])
        aux = ("Sentence: " + sentences[value[0]],
               "Entity attributes: " + str(all_entities[value[0]][value[1]].attrib),
               "Dependency attributes: " + str(tokens_e1),
               "ROOT: " + all_roots[value[0]],
               "Entity_2 attributes: " + str(all_entities[value[0]][value[2]].attrib),
               "Dependency_2 attributes: " + str(tokens_e2))
        truth[value[0]].append((aux, value[-1]))

    # Using readlines()
    file1 = open('../output/task9.2_out_1.txt', 'r')
    Lines = file1.readlines()

    output = {}
    wrong_entities = []
    new_sent = ""
    matched_entities = []
    missing = []

    # Strips the newline character
    for line in Lines:
        value = line.split("|")
        if value[0] not in output:
            output[value[0]] = []
        tokens_e1, tokens_e2 = check_interaction(all_parse[value[0]], all_entities[value[0]], value[1], value[2])

        aux = (("Sentence: " + sentences[value[0]],
               "Entity attributes: " + str(all_entities[value[0]][value[1]].attrib),
               "Dependency attributes: " + str(tokens_e1),
               "ROOT: " + all_roots[value[0]],
               "Entity_2 attributes: " + str(all_entities[value[0]][value[2]].attrib),
               "Dependency_2 attributes: " + str(tokens_e2)), value[-1])

        output[value[0]].append(aux)

        if new_sent != value[0] and new_sent != "":
            if new_sent in truth:
                missing += [item for item in truth[new_sent] if item not in matched_entities]
            matched_entities = []
            new_sent = value[0]

        if new_sent == "":
            new_sent = value[0]
        if value[0] in truth and aux in truth[value[0]]:
            matched_entities.append(aux)
        else:
            wrong_entities.append(aux)

    missing_dict = {}
    for ent, type in missing:
        if type[:-1] not in missing_dict:
            missing_dict[type[:-1]] = []
        missing_dict[type[:-1]].append(ent)

    wrong_entities_dict = {}
    for ent, type in wrong_entities:
        if type[:-1] not in wrong_entities_dict:
            wrong_entities_dict[type[:-1]] = []
        wrong_entities_dict[type[:-1]].append(ent)
    file = open("missing_pairs.txt", 'w')
    print("MISSING:\n", json.dumps(missing_dict, indent=4), file=file)
    print("\n\n", file=file)
    print("WRONG:\n", json.dumps(wrong_entities_dict, indent=4), file=file)
