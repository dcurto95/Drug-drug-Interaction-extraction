import os
import xml.etree.ElementTree as ET
import numpy as np
from nltk.parse.corenlp import CoreNLPDependencyParser

def parse_xml(file):
    tree = ET.parse(file)
    return tree.getroot()

def get_sentence_info(child):
    return child.get('id'), child.get('text')

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
                return analysis.nodes[intersection[0]]["lemma"], analysis.nodes[intersection[0]]["tag"]

    return 'None', 'None'


def check_common_ancestor(ancestor_token):

    if ancestor_token['lemma'] in ['approach', 'recommend', 'contraindicate']:
        return 1, 'advise'
    return 0, 'null'

def check_interaction(analysis, entities, id_e1, id_e2):

    inbetween_text = extract_inbetween_text(analysis, entities, id_e1, id_e2)
    (is_ddi, ddi_type) = rules_without_dependency(inbetween_text)

    e1_off = entities[id_e1]
    e2_off = entities[id_e2]

    nodes_drug1, nodes_drug2 = extract_drug_nodes(analysis, e1_off, e2_off)
    drug1, drug2 = extract_drug_names(entities, id_e1, id_e2, nodes_drug1, nodes_drug2)
    head_drug1, head_drug2 = extract_head_drugs_info(analysis, nodes_drug1, nodes_drug2)
    distance = extract_distance_between_drugs(nodes_drug1, nodes_drug2)

    if drug1 == drug2:
        return 0, 'null'

    common_ancestor_info = find_common_ancestor(analysis, nodes_drug1[0]["address"], nodes_drug2[0]["address"])
    if common_ancestor_info[0] in ['interact', 'interaction']:
        if common_ancestor_info[1] in ['NN', 'VB', 'NNS', 'VBP']:
            return 1, 'int'

    if analysis.root['lemma'] in ['advise', 'recommend', 'contraindicate', 'suggest']:
        return 1, 'advise'

    if analysis.root['lemma'] in ['enhance', 'inhibit', 'block', 'produce']:
        return 1, 'effect'

    if is_ddi:
        return is_ddi, ddi_type

    return 0, 'null'


def extract_head_drugs_info(analysis, nodes_drug1, nodes_drug2):
    drug1_node = nodes_drug1[0]
    drug2_node = nodes_drug2[0]
    counter_d1 = counter_d2 = 0
    head1_found = head2_found = False
    while not head1_found or not head2_found:
        head_drug1 = analysis.nodes[drug1_node["head"]]
        head_drug2 = analysis.nodes[drug2_node["head"]]
        lemma_pos_d1 = (head_drug1["lemma"], head_drug1["tag"])
        lemma_pos_d2 = (head_drug2["lemma"], head_drug2["tag"])
        if head_drug1 in nodes_drug1:
            counter_d1 = counter_d1 + 1
            drug1_node = nodes_drug1[counter_d1]
        else:
            head1_found = True
        if head_drug2 in nodes_drug2:
            counter_d2 = counter_d2 + 1
            drug2_node = nodes_drug2[counter_d2]
        else:
            head2_found = True

    return lemma_pos_d1, lemma_pos_d2


def extract_drug_names(entities, id_e1, id_e2, nodes_drug1, nodes_drug2):
    drug1 = extract_drug_name(entities, id_e1, nodes_drug1)
    drug2 = extract_drug_name(entities, id_e2, nodes_drug2)
    return drug1, drug2


def extract_inbetween_text(analysis, entities, id_e1, id_e2):
    index_drug1 = index_drug2 = 0
    sentece_analysis = [None] * (len(analysis.nodes) - 1)
    for i_node in range(1, len(analysis.nodes)):
        current_node = analysis.nodes[i_node]
        address = current_node["address"]
        word = current_node["word"]
        start = current_node["start_off"]
        end = current_node["end_off"]
        end_drug1 = entities[id_e1][0][1] if len(entities[id_e1] == 1) else entities[id_e1][len(entities[id_e1]) - 1][1]
        if end == end_drug1:
            index_drug1 = i_node - 1
        if start == entities[id_e2][0][0]:
            index_drug2 = i_node - 1
        sentece_analysis[address - 1] = word
    inbetween_text = ' '.join(sentece_analysis[index_drug1:index_drug2])
    return inbetween_text


def extract_drug_nodes(analysis, e1_off, e2_off):
    nodes_drug1 = []
    nodes_drug2 = []

    start_drug1 = e1_off[0][0]
    start_drug2 = e2_off[0][0]
    end_drug1 = e1_off[0][1] if len(e1_off) < 2 else e1_off[len(e1_off) - 1][1]
    end_drug2 = e2_off[0][1] if len(e2_off) < 2 else e2_off[len(e2_off) - 1][1]

    for i_node in range(1, len(analysis.nodes)):
        current_node = analysis.nodes[i_node]
        start = current_node["start_off"]
        end = current_node["end_off"]

        if start == start_drug1 or (start < start_drug1 <= end):
            nodes_drug1.append(current_node)
        if start == start_drug2 or (start < start_drug2 <= end):
            nodes_drug2.append(current_node)
        if start != start_drug1:
            if end == end_drug1 or (start < end_drug1 <= end) or(start_drug1 < end < end_drug1):
                nodes_drug1.append(current_node)
        if start != start_drug2:
            if end == end_drug2 or (start < end_drug2 <= end) or(start_drug2 < end < end_drug2):
                nodes_drug2.append(current_node)

    return nodes_drug1, nodes_drug2


def extract_drug_name(entities,id, nodes_drug):
    drug = []
    for node in nodes_drug:
        if len(entities[id]) < 2:
            drug.append(node["word"])
        else:
            if not (node["start_off"] > entities[id][0][1] and node["end_off"] < entities[id][1][0]):
                drug.append(node["word"])
    return " ".join(drug)

def rules_without_dependency(sentence):
    is_ddi = 0
    ddi_type = "null"

    effect_list = ['administer', 'potentiate', 'prevent', 'effect', 'cause']
    effect_list = ['administer', 'potentiate', 'prevent', 'effect', 'cause']

    if "effect" in sentence:
        is_ddi = 1
        ddi_type = "effect"
    if "should" in sentence:
        is_ddi = 1
        ddi_type = "advise"
    if "increase" in sentence or "decrease" in sentence or "reduce" in sentence:
        is_ddi = 1
        ddi_type = "mechanism"

    return is_ddi, ddi_type

def extract_distance_between_drugs(nodes_drug1, nodes_drug2):

    max_d1 = max([i["address"] for i in nodes_drug1])
    min_d1 = min([i["address"] for i in nodes_drug2])
    distance = abs(min_d1 - max_d1)
    return distance

if __name__ == '__main__':
    output_file_name = "task9.2_out_1.txt"
    input_directory = '../data/Train/'

    output_file = open('../output/' + output_file_name, 'w+')

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

                (is_ddi, ddi_type) = check_interaction(analysis, entities, id_e1, id_e2)

                print("|".join([sid, id_e1, id_e2, str(is_ddi), ddi_type]), file=output_file)

    # Close the file
    output_file.close()
    print(evaluate(input_directory, output_file_name))

