import os
import re
import string
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from chemdataextractor.nlp.tokenize import ChemWordTokenizer
from nltk import word_tokenize
from nltk.parse.corenlp import CoreNLPDependencyParser


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()


def get_sentence_info(child):
    return child.get('id'), child.get('text')


def evaluate(inputdir, outputfile):
    return os.system("java -jar ../eval/evaluateDDI.jar " + inputdir + " ../output/" + outputfile)


def create_model(features_file_name):
    return os.system("ubuntu run ./megam-64.opt -quiet -nc -nobias multiclass ../features/" + features_file_name + " > ../models/model.dat")


def predict(features_file_name, test_name, model_file_name="../models/model.dat"):
    return os.system(
        "ubuntu run ./megam-64.opt -nc -nobias -predict " + model_file_name + " multiclass ../features/" + features_file_name + " > ../output/" + test_name)


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
        for index in range(1, len(analysis.nodes)):
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


def find_common_verb_ancestor(analysis, first_index, second_index):
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
        if intersection and analysis.nodes[intersection[0]]['tag'][0] == 'V':
            return intersection[0]

    return analysis.root['address']


def check_common_ancestor(ancestor_token):
    # if ancestor_token['lemma'] in ['report', 'interaction', 'suggest']:
    #     return 1, 'int'
    if ancestor_token['lemma'] in ['approach', 'recommend', 'contraindicate']:
        return 1, 'advise'
    return 0, 'null'


def check_interaction(analysis, entities, id_e1, id_e2, truth_ddi, dic, good_dic):
    inbetween_text = extract_inbetween_text(analysis, entities, id_e1, id_e2)
    (is_ddi, ddi_type) = rules_without_dependency(inbetween_text)

    if is_ddi:
        return is_ddi, ddi_type
    if analysis.root['lemma'] in ['advise', 'recommend', 'contraindicate', 'suggest']:
        return 1, 'advise'
    if analysis.root['lemma'] in ['enhance', 'inhibit', 'block', 'produce']:
        return 1, 'effect'
    # TODO: NO BORRAR UTIL PER DEPENDENCY TREE
    e1_off = entities[id_e1]
    e2_off = entities[id_e2]

    for word_index in range(1, len(analysis.nodes)):
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
                common_ancestor_index = find_common_verb_ancestor(analysis, word_index, second_index)
                return check_common_ancestor(analysis.nodes[common_ancestor_index])
    #             return check_dependency(analysis, entity_list_tokens, e2_start_off, word_index, second_index, truth_ddi,
    #                                     dic, good_dic)

    return 0, 'null'


def extract_inbetween_text(analysis, entities, id_e1, id_e2):
    index_drug1 = index_drug2 = 0
    sentece_analysis = [None] * (len(analysis.nodes) - 1)
    for i_node in range(1, len(analysis.nodes)):
        current_node = analysis.nodes[i_node]
        address = current_node["address"]
        word = current_node["word"]
        start = current_node["start_off"]
        end = current_node["end_off"]
        end_drug1 = entities[id_e1][0][1] if len(entities[id_e1][0] == 2) else entities[id_e1][0][3]
        if end == end_drug1:
            index_drug1 = i_node - 1
        if start == entities[id_e2][0][0]:
            index_drug2 = i_node - 1
        sentece_analysis[address - 1] = word
    inbetween_text = ' '.join(sentece_analysis[index_drug1:index_drug2])
    return inbetween_text


def rules_without_dependency(sentence):
    features = []

    features.append("Contains_effect=" + str("effect" in sentence))
    features.append("Contains_should=" + str("should" in sentence))
    features.append("Contains_mechanism_word=" + str("increase" in sentence or "decrease" in sentence or "reduce" in sentence))
    features.append("Contains_interact=" + str("interact" in sentence))

    return features


def extract_features(analysis, entities, id_e1, id_e2):
    features = []

    inbetween_text = extract_inbetween_text(analysis, entities, id_e1, id_e2)
    rule_features = rules_without_dependency(inbetween_text)

    features.extend(rule_features)
    features.append("lemma_advise=" + str(analysis.root['lemma'] in ['advise', 'recommend', 'contraindicate', 'suggest']))
    features.append("lemma_effect=" + str(analysis.root['lemma'] in ['enhance', 'inhibit', 'block', 'produce']))

    return features


def save_features(features_file_name, input_directory, training=False):
    features_file = open('../features/' + features_file_name + '.txt', 'w+')
    features_file_without_sent = open('../features/' + features_file_name + '.features', 'w+')

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

            for pair in child.findall('pair'):
                id_e1 = pair.get('e1')
                id_e2 = pair.get('e2')
                ddi_type = pair.get('type') if pair.get('type') is not None else 'null'

                features = extract_features(analysis, entities, id_e1, id_e2)

                print("\t".join([sid, id_e1, id_e2, ddi_type, "\t".join(features)]), file=features_file)
                if training:
                    print("\t".join([ddi_type, "\t".join(features)]), file=features_file_without_sent)
                else:
                    print("\t".join(features), file=features_file_without_sent)
    features_file.close()


def parse_features(features_file):
    lines = features_file.readlines()

    sent_info = []
    sent_features = []
    # Strips the newline character
    for line in lines:
        value = line.split("\t")
        sid, id_e1, id_e2, ddi_type, features = value[0], value[1], value[2], value[3], value[4:]
        sent_info.append((sid, id_e1, id_e2))
        sent_features.append(features)

    return sent_info, sent_features


def create_features():
    # Save Train features
    features_file_name = "train_features"
    input_directory = '../data/Train/'
    save_features(features_file_name, input_directory, training=True)

    # Save Devel features
    features_file_name = "devel_features"
    input_directory = '../data/Devel/'
    save_features(features_file_name, input_directory)

    # Save Test-DDI features
    features_file_name = "test_features"
    input_directory = '../data/Test-DDI/'
    save_features(features_file_name, input_directory)


def parse_prediction(prediction_file):
    lines = prediction_file.readlines()

    predictions = [line.split('\t')[0] for line in lines]

    return predictions


if __name__ == '__main__':
    # Create_features -> Train -> Predict_Devel or Predict_Test
    stage = 'Predict_Devel'

    output_file_name = "task9.2_out_2.txt"

    if stage == 'Create_features':
        create_features()

    elif stage == 'Train':
        features_file_name = "train_features.features"
        features_file = open('../features/' + features_file_name, 'r')

        create_model(features_file_name)

    elif stage == 'Predict_Devel':
        input_directory = '../data/Devel/'
        output_file_name = "task9.2_devel-out_1.txt"
        output_file = open('../output/' + output_file_name, 'w+')

        features_file_name = "devel_features.txt"
        features_file = open('../features/' + features_file_name, 'r')
        features_without_sent_file_name = "devel_features.features"

        sentences_info, sent_features = parse_features(features_file)
        features_file.close()

        prediction_file_name = "Devel.test"
        predict(features_without_sent_file_name, prediction_file_name)
        prediction_file = open('../output/' + prediction_file_name, 'r')

        predictions = parse_prediction(prediction_file)

        for prediction, (sid, id_e1, id_e2) in zip(predictions, sentences_info):
            is_ddi = 1 if prediction != 'null' else 0
            print("|".join([sid, id_e1, id_e2, str(is_ddi), prediction]), file=output_file)

        # Close the file
        output_file.close()
        print(evaluate(input_directory, output_file_name))

    elif stage == 'Predict_Test':
        input_directory = '../data/Test-DDI/'
        output_file_name = "task9.2_test-out_1.txt"
        output_file = open('../output/' + output_file_name, 'w+')

        features_file_name = "test_features.txt"
        features_file = open('../features/' + features_file_name, 'r')
        features_without_sent_file_name = "test_features.features"

        sentences_info, sent_features = parse_features(features_file)
        features_file.close()

        prediction_file_name = "Test.test"
        predict(features_without_sent_file_name, prediction_file_name)
        prediction_file = open('../output/' + prediction_file_name, 'r')

        predictions = parse_prediction(prediction_file)

        for prediction, (sid, id_e1, id_e2) in zip(predictions, sentences_info):
            is_ddi = 1 if prediction != 'null' else 0
            print("|".join([sid, id_e1, id_e2, str(is_ddi), prediction]), file=output_file)

        # Close the file
        output_file.close()
        print(evaluate(input_directory, output_file_name))
