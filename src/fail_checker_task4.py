import json
import os
import xml.etree.ElementTree as ET

import numpy as np
from nltk import CoreNLPDependencyParser

import analyze_pairs


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


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()


def get_sentence_info(child):
    return child.get('id'), child.get('text')


def find_second_entity(analysis, word_index, e2_start_off):
    index = word_index

    while index < len(analysis.nodes):
        if 'start_off' in analysis.nodes[index] and (analysis.nodes[index]['start_off'] == e2_start_off or (
                analysis.nodes[index]['start_off'] < e2_start_off <= analysis.nodes[index]['end_off'])):
            return index
        index += 1
    raise Exception("Entity not found")


def find_common_ancestor(analysis, first_index, second_index):
    visited_first = [first_index]
    visited_second = [second_index]

    while not (analysis.root['address'] in visited_first and analysis.root['address'] in visited_second):
        head = analysis.nodes[first_index]['head']
        if head is not None:
            visited_first.append(head)
            first_index = head
        head = analysis.nodes[second_index]['head']
        if head is not None:
            visited_second.append(head)
            second_index = head
        intersection = list(set(visited_first) & set(visited_second))
        if intersection:
            if analysis.nodes[intersection[0]]['tag'][0] == 'V':
                return intersection[0]

    return analysis.root['address']


def check_interaction(analysis, entities, id_e1, id_e2):
    e1_off = entities[id_e1].get('charOffset')
    e2_off = entities[id_e2].get('charOffset')

    e1_off = split_offset(e1_off)
    e2_off = split_offset(e2_off)

    tokens_info_e1 = []
    tokens_info_e2 = []
    common_ancestor_index = 0
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
                common_ancestor_index = find_common_ancestor(analysis, word_index, second_index)
    return tokens_info_e1, tokens_info_e2, analysis.nodes[common_ancestor_index]


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

    # Process each file in the directory
    for index_file, filename in enumerate(os.listdir(input_directory)):
        # Parse XML file
        root = parse_xml(input_directory + filename)
        print(" - File:", filename, "(", index_file + 1, "out of ", len(os.listdir(input_directory)), ")")

        for child in root:
            sid, text = get_sentence_info(child)
            entities = {}
            if not text:
                continue
            for entity in child.findall('entity'):
                entities[entity.get('id')] = entity
            sentences[sid] = text
            all_entities[sid] = entities
            parse = analyze(text)
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

        aux = (("Sentence: " + sentences[value[0]],
                "Entity attributes: " + str(all_entities[value[0]][value[1]].attrib),
                "ROOT: " + all_roots[value[0]],
                "Entity_2 attributes: " + str(all_entities[value[0]][value[2]].attrib)), value[-1])
        truth[value[0]].append(aux)

    # Using readlines()
    file1 = open('../output/task9.2_train-out_1.txt', 'r')
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

        aux = (("Sentence: " + sentences[value[0]],
                "Entity attributes: " + str(all_entities[value[0]][value[1]].attrib),
                "ROOT: " + all_roots[value[0]],
                "Entity_2 attributes: " + str(all_entities[value[0]][value[2]].attrib)), value[-1])

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
    file = open("missing_pairs(task4).txt", 'w')
    print("MISSING:\n", json.dumps(missing_dict, indent=4), file=file)
    print("\n\n", file=file)
    print("WRONG:\n", json.dumps(wrong_entities_dict, indent=4), file=file)
