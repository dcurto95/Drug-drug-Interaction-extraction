import os
import pickle
import xml.etree.ElementTree as ET

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import numpy as np
from nltk.parse.corenlp import CoreNLPDependencyParser
from sklearn.tree import DecisionTreeClassifier


def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()


def get_sentence_info(child):
    return child.get('id'), child.get('text')


def evaluate(inputdir, outputfile):
    return os.system("java -jar ../eval/evaluateDDI.jar " + inputdir + " ../output/" + outputfile)


def create_model(features_file_name):
    return os.system(
        "ubuntu run ./megam-64.opt -quiet -nc -nobias multiclass ../features/" + features_file_name + " > ../models/model.dat")


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


def basic_features(in_between_text, feature_priority):
    advise_list = ['can', 'could', 'may', 'might', 'will', 'shall', 'should', 'ought', 'must', 'would']
    effect_list = ['administer', 'potentiate', 'prevent', 'effect', 'cause']
    mechanism_list = ['reduce', 'increase', 'decrease']
    int_list = ['interact', 'interaction', 'interfere']

    times = {'int': 0, 'advise': 0, 'effect': 0, 'mechanism': 0}
    for word in in_between_text.split():
        if word in int_list:
            times['int'] = times.get('int', 0) + 1
        if word in advise_list:
            times['advise'] = times.get('advise', 0) + 1
        if word in effect_list:
            times['effect'] = times.get('effect', 0) + 1
        if word in mechanism_list:
            times['mechanism'] = times.get('mechanism', 0) + 1

    return [(feature_priority, key + "=" + str(value)) for key, value in times.items()]


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
            return intersection[0], visited_first, visited_second

    return analysis.root['address'], visited_first, visited_second


def get_vertical_order_features(analysis, first_index, second_index, feature_priority):
    original_first = first_index
    original_second = second_index

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

        if original_first in visited_second:
            return [(feature_priority, "e1_over_e2")]
        if original_second in visited_first:
            return [(feature_priority, "e2_over_e1")]

    return [(feature_priority, "no_realtion")]


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


def rules_without_dependency(sentence, feature_priority):
    features = []

    features.append((feature_priority, "Contains_effect=" + str("effect" in sentence.lower())))
    features.append((feature_priority, "Contains_should=" + str("should" in sentence.lower())))
    features.append((feature_priority,
                     "Contains_mechanism_word=" + str(
                         "increase" in sentence or "decrease" in sentence or "reduce" in sentence.lower())))
    features.append((feature_priority, "Contains_interact=" + str("interact" in sentence.lower())))

    return features


def get_common_ancestor_features(common_ancestor_node, feature_priority, text=""):
    common_ancestor_features = []

    common_ancestor_features.append(
        (feature_priority, text + "ancestor_lemma=" + common_ancestor_node['lemma'].lower()))
    common_ancestor_features.append((feature_priority, text + "ancestor_word=" + common_ancestor_node['word'].lower()))
    common_ancestor_features.append((feature_priority, text + "ancestor_tag=" + common_ancestor_node['tag'].lower()))
    common_ancestor_features.append(
        (feature_priority, text + "ancestor_subtag=" + common_ancestor_node['tag'].lower()[0]))
    common_ancestor_features.append((feature_priority, text + "ancestor_rel=" + common_ancestor_node['rel'].lower()))

    advise_list = ['can', 'could', 'may', 'might', 'will', 'shall', 'should', 'ought', 'must', 'would']
    effect_list = ['administer', 'potentiate', 'prevent', 'effect', 'cause']
    mechanism_list = ['reduce', 'increase', 'decrease']
    int_list = ['interact', 'interaction', 'interfere']
    #
    # common_ancestor_features.append(
    #     (feature_priority, text + "ancestor_int_lemma=" + common_ancestor_node['lemma'].lower() in int_list))
    # common_ancestor_features.append(
    #     (feature_priority, text + "ancestor_effect_lemma=" + common_ancestor_node['lemma'].lower() in effect_list))
    # common_ancestor_features.append(
    #     (feature_priority, text + "ancestor_advise_lemma=" + common_ancestor_node['lemma'].lower() in advise_list))
    # common_ancestor_features.append((
    #     feature_priority, text + "ancestor_mechanism_lemma=" + common_ancestor_node['lemma'].lower() in mechanism_list))

    return common_ancestor_features


def get_dependency_path(analysis, first_route, second_route, common_ancestor_index):
    path = "path="
    rel_path = "rel_path="

    distance = 0
    route = first_route[:first_route.index(common_ancestor_index) + 1] + second_route[
                                                                         :second_route.index(common_ancestor_index)][
                                                                         ::-1]
    for index in route:
        distance += 1
        path += ' ' + analysis.nodes[index]['lemma']
        rel_path += ' ' + analysis.nodes[index]['rel']

    return path, rel_path, distance


def get_entities_info(analysis, first_index, second_index, feature_priority):
    entities_features = [(feature_priority, "e1_lemma=" + analysis.nodes[first_index]['lemma']),
                         (feature_priority, "e1_rel=" + analysis.nodes[first_index]['rel']),
                         (feature_priority, "e1_tag=" + analysis.nodes[first_index]['tag']),
                         (feature_priority, "e1_subtag=" + analysis.nodes[first_index]['tag'][0]),
                         (feature_priority, "e2_lemma=" + analysis.nodes[second_index]['lemma']),
                         (feature_priority, "e2_rel=" + analysis.nodes[second_index]['rel']),
                         (feature_priority, "e2_tag=" + analysis.nodes[second_index]['tag']),
                         (feature_priority, "e2_subtag=" + analysis.nodes[second_index]['tag'][0])]

    return entities_features


def analyze_dependency_tree(analysis, entities, id_e1, id_e2, feature_priority):
    dependency_features = []

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
                dependency_features.append((0, "equal=" + str(word_index == second_index)))
                order_features = get_vertical_order_features(analysis, word_index, second_index, feature_priority + 0.2)
                dependency_features.extend(order_features)

                common_ancestor_index, first_path, second_path = find_common_ancestor(analysis, word_index,
                                                                                      second_index)
                common_verb_ancestor_index = find_common_verb_ancestor(analysis, word_index, second_index)

                common_ancestor_features = get_common_ancestor_features(analysis.nodes[common_ancestor_index],
                                                                        feature_priority + 0.1)
                dependency_features.extend(common_ancestor_features)
                common_ancestor_features = get_common_ancestor_features(analysis.nodes[common_verb_ancestor_index],
                                                                        feature_priority + 0.3, text="verb_")
                dependency_features.extend(common_ancestor_features)

                # path, rel_path, distance = get_dependency_path(analysis, first_path, second_path, common_ancestor_index)
                # dependency_features.append((feature_priority + 0.4, path))
                # dependency_features.append((feature_priority + 0.4, rel_path))
                # dependency_features.append((feature_priority + 0.4, "distance=" + str(distance)))

                dependency_features.append(
                    (feature_priority, "head_nmod=" + str(analysis.nodes[word_index]['rel'] == 'nmod')))
                dependency_features.append((feature_priority,
                                            "int_nmod=" + str(analysis.nodes[word_index]['rel'] == 'nmod' and
                                                              analysis.nodes[common_ancestor_index]['lemma'] in [
                                                                  'interact', 'interaction',
                                                                  'implication'] and (analysis.nodes[second_index][
                                                                                          'rel'] == 'conj'))))

                entities_features = get_entities_info(analysis, word_index, second_index, feature_priority + 0.5)
                dependency_features.extend(entities_features)

                return dependency_features
    return dependency_features


def extract_features(analysis, entities, id_e1, id_e2):
    features = []

    inbetween_text = extract_inbetween_text(analysis, entities, id_e1, id_e2)
    rule_features = rules_without_dependency(inbetween_text, 1)
    features.extend(rule_features)

    basic_features_list = basic_features(inbetween_text, 2)
    features.extend(basic_features_list)

    features.append((2, "lemma_root=" + analysis.root['lemma']))
    features.append((2, "lemma_tag=" + analysis.root['tag']))

    advise_list = ['can', 'could', 'may', 'might', 'will', 'shall', 'should', 'ought', 'must', 'would']
    effect_list = ['administer', 'potentiate', 'prevent', 'effect', 'cause']
    mechanism_list = ['reduce', 'increase', 'decrease']
    int_list = ['interact', 'interaction', 'interfere']

    features.append(
        (2, "lemma_advise=" + str(
            analysis.root['lemma'] in ['advise', 'recommend', 'contraindicate', 'suggest', 'can', 'could', 'may',
                                       'might', 'will', 'shall', 'should', 'ought', 'must', 'would'])))
    features.append((2, "lemma_effect=" + str(
        analysis.root['lemma'] in ['enhance', 'inhibit', 'block', 'produce', 'administer', 'potentiate', 'prevent',
                                   'effect', 'cause'])))
    features.append(
        (2, "lemma_int=" + str(analysis.root['lemma'] in int_list)))
    features.append((2, "lemma_mechanism=" + str(analysis.root['lemma'] in mechanism_list)))

    dependency_features = analyze_dependency_tree(analysis, entities, id_e1, id_e2, 3)
    features.extend(dependency_features)

    features = sorted(features, key=lambda x: x[0])
    return np.asarray(features)[:, 1]


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


def save_features_quick(features_file_name, input_directory, training=False):
    features_file = open('../features/' + features_file_name + '.txt', 'w+')
    features_file_without_sent = open('../features/' + features_file_name + '.features', 'w+')

    name = input_directory.split("/")[-2]
    pickle_in = open("../pickle/" + name + ".pickle", "rb")
    sentence_list = pickle.load(pickle_in)

    for sid, entities, analysis, id_e1, id_e2, ddi_type in sentence_list:
        features = extract_features(analysis, entities, id_e1, id_e2)

        print("\t".join([sid, id_e1, id_e2, ddi_type, "\t".join(features)]), file=features_file)
        if training:
            print("\t".join([ddi_type, "\t".join(features)]), file=features_file_without_sent)
        else:
            print("\t".join(features), file=features_file_without_sent)
    features_file.close()


class DepTree:
    def __init__(self, analysis):
        self.nodes = dict(analysis.nodes.items())
        self.root = analysis.root


def load_pickle_sentences(input_directory):
    sentence_list = []
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

            analysis = DepTree(analysis)
            for pair in child.findall('pair'):
                id_e1 = pair.get('e1')
                id_e2 = pair.get('e2')
                ddi_type = pair.get('type') if pair.get('type') is not None else 'null'

                sentence_list.append((sid, entities, analysis, id_e1, id_e2, ddi_type))
    name = input_directory.split("/")[-2]
    pickle_out = open("../pickle/" + name + ".pickle", "wb")
    pickle.dump(sentence_list, pickle_out)
    pickle_out.close()


def parse_features(features_file):
    lines = features_file.readlines()

    sent_info = []
    gold = []
    sent_features = []
    # Strips the newline character
    for line in lines:
        value = line.split("\t")
        sid, id_e1, id_e2, ddi_type, features = value[0], value[1], value[2], value[3], value[4:]
        sent_info.append((sid, id_e1, id_e2))
        sent_features.append(features)
        gold.append(ddi_type)

    return sent_info, gold, sent_features


def create_features(quick):
    if quick:
        print("Creating features... QUICK")
        # Save Train features
        features_file_name = "train_features"
        input_directory = '../data/Train/'
        save_features_quick(features_file_name, input_directory, training=True)

        # Save Train features
        features_file_name = "predict_train_features"
        input_directory = '../data/Train/'
        save_features_quick(features_file_name, input_directory)

        # Save Devel features
        features_file_name = "devel_features"
        input_directory = '../data/Devel/'
        save_features_quick(features_file_name, input_directory)

        # Save Test-DDI features
        features_file_name = "test_features"
        input_directory = '../data/Test-DDI/'
        save_features_quick(features_file_name, input_directory)
    else:
        print("Creating features... SLOW")
        # Save Train features
        features_file_name = "train_features"
        input_directory = '../data/Train/'
        save_features(features_file_name, input_directory, training=True)

        # Save Train features
        features_file_name = "predict_train_features"
        input_directory = '../data/Train/'
        save_features(features_file_name, input_directory)

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


def create_SVC_model(features, gold):
    # clf = DecisionTreeClassifier()
    clf = SVC(gamma='auto')

    encoders = []
    features = np.asarray(features)
    for i, col in enumerate(features.T):
        le = LabelEncoder()
        le.fit(np.unique(col))
        features[:, i] = le.transform(features[:, i])
        encoders.append(le)

    clf.fit(features, gold)
    pickle_out = open("../pickle/model.pickle", "wb")
    pickle.dump(clf, pickle_out)
    pickle_out.close()

    pickle_out = open("../pickle/encoders.pickle", "wb")
    pickle.dump(encoders, pickle_out)
    pickle_out.close()


def svc_predict(sent_features):
    pickle_in = open("../pickle/model.pickle", "rb")
    clf = pickle.load(pickle_in)

    pickle_in = open("../pickle/encoders.pickle", "rb")
    encoders = pickle.load(pickle_in)

    features = np.asarray(sent_features)
    for i, (col, le) in enumerate(zip(features.T, encoders)):
        missing_labels = list(set(np.concatenate((le.classes_, np.unique(col)))) - set(le.classes_))
        if missing_labels:
            le.classes_ = np.concatenate((le.classes_, missing_labels))
        features[:, i] = le.transform(features[:, i])
    return clf.predict(features)


def execute_stage(stage, quick=False):
    if stage == 'Create_features':
        create_features(quick)

    elif stage == 'Train':
        features_file_name = "train_features.txt"
        features_file = open('../features/' + features_file_name, 'r')
        sentences_info, gold, sent_features = parse_features(features_file)
        features_file.close()
        # create_model(features_file_name)
        create_SVC_model(sent_features, gold)

    elif stage == 'Predict_Train':
        input_directory = '../data/Train/'
        output_file_name = "task9.2_train-out_1.txt"
        output_file = open('../output/' + output_file_name, 'w+')

        features_file_name = "predict_train_features.txt"
        features_file = open('../features/' + features_file_name, 'r')
        # features_without_sent_file_name = "predict_train_features.features"

        sentences_info, gold, sent_features = parse_features(features_file)
        features_file.close()

        # prediction_file_name = "Train.test"
        # predict(features_without_sent_file_name, prediction_file_name)
        # prediction_file = open('../output/' + prediction_file_name, 'r')
        #
        # predictions = parse_prediction(prediction_file)

        predictions = svc_predict(sent_features)

        for prediction, (sid, id_e1, id_e2) in zip(predictions, sentences_info):
            is_ddi = 1 if prediction != 'null' else 0
            print("|".join([sid, id_e1, id_e2, str(is_ddi), prediction]), file=output_file)

        # Close the file
        output_file.close()
        print(evaluate(input_directory, output_file_name))

    elif stage == 'Predict_Devel':
        input_directory = '../data/Devel/'
        output_file_name = "task9.2_devel-out_1.txt"
        output_file = open('../output/' + output_file_name, 'w+')

        features_file_name = "devel_features.txt"
        features_file = open('../features/' + features_file_name, 'r')
        features_without_sent_file_name = "devel_features.features"

        sentences_info, gold, sent_features = parse_features(features_file)
        features_file.close()

        # prediction_file_name = "Devel.test"
        # predict(features_without_sent_file_name, prediction_file_name)
        # prediction_file = open('../output/' + prediction_file_name, 'r')
        #
        # predictions = parse_prediction(prediction_file)

        predictions = svc_predict(sent_features)

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

        sentences_info, gold, sent_features = parse_features(features_file)
        features_file.close()

        # prediction_file_name = "Test.test"
        # predict(features_without_sent_file_name, prediction_file_name)
        # prediction_file = open('../output/' + prediction_file_name, 'r')
        #
        # predictions = parse_prediction(prediction_file)

        predictions = svc_predict(sent_features)

        for prediction, (sid, id_e1, id_e2) in zip(predictions, sentences_info):
            is_ddi = 1 if prediction != 'null' else 0
            print("|".join([sid, id_e1, id_e2, str(is_ddi), prediction]), file=output_file)

        # Close the file
        output_file.close()
        print(evaluate(input_directory, output_file_name))


def load_pickles():
    input_directory = '../data/Train/'
    load_pickle_sentences(input_directory)

    input_directory = '../data/Devel/'
    load_pickle_sentences(input_directory)

    input_directory = '../data/Test-DDI/'
    load_pickle_sentences(input_directory)


if __name__ == '__main__':
    # Create_features -> Train -> Predict_Train, Predict_Devel or Predict_Test

    # If full train process wants to be done:
    stages = ['Create_features', 'Train', 'Predict_Train', 'Predict_Devel', 'Predict_Test']
    # stages = ['Predict_Train']

    # load_pickles()
    # print("PICKLES LOADED")

    for stage in stages:
        print("------------------" + stage + "------------------")
        execute_stage(stage, quick=True)
