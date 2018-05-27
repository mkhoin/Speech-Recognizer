
# coding: utf-8
import os
import sys
from math import log
import numpy as np
from collections import Counter
incrementer = 1.0
incrementer2 = 0.00000000001

def getFileContents(filename):
    data = None
    with open(filename, 'r') as f:
        data = f.readlines()
    return data


def getFileFromCommandLine():
    filename = sys.argv[1]
    return getFileContents(filename)


def splitWordTag(word_tag_pair):
    splitted = word_tag_pair.split('/')
    tag = splitted[-1]
    word = '/'.join(splitted[:-1])
    return word, tag


def getUniqueTags(tagged_data):
    tags = {}
    for line in tagged_data:
        word_tag_pairs = line.strip().split(' ')
        for word_tag_pair in word_tag_pairs:
            word, tag = splitWordTag(word_tag_pair)
            if tag in tags.keys():
                tags[tag] += 1
            else:
                tags[tag] = 1
    return tags


def getOpenProbabilities(tagged_data, all_tags_dict):
    global incrementer2
    sentences_count = len(tagged_data)
    open_tag_count_dict = {}
    for line in tagged_data:
        first_word_tag_pairs = line.strip().split(' ')[0]
        word, tag = splitWordTag(first_word_tag_pairs)
        if tag in open_tag_count_dict.keys():
            open_tag_count_dict[tag] += 1
        else:
            open_tag_count_dict[tag] = 1
    
    #increment all existing tags count to one
    open_tag_count_dict.update((tag, occurances + incrementer2) for tag, occurances in open_tag_count_dict.items())
    sentences_count += (sentences_count*incrementer2)
    
    #add one to non-opening tags
    for tag in all_tags_dict.keys():
        try:
            val = open_tag_count_dict[tag]
        except KeyError as e:
            open_tag_count_dict[tag] = incrementer2
            sentences_count += incrementer2
    
    open_tag_count_dict.update((tag, (occurances*1.0)/sentences_count) for tag, occurances in open_tag_count_dict.items())
    return open_tag_count_dict


def getCloseProbabilities(tagged_data, all_tags_dict):
    global incrementer2
    sentences_count = len(tagged_data)
    close_tag_count_dict = {}
    for line in tagged_data:
        last_word_tag_pairs = line.strip().split(' ')[-1]
        word, tag = splitWordTag(last_word_tag_pairs)
        if tag in close_tag_count_dict.keys():
            close_tag_count_dict[tag] += 1
        else:
            close_tag_count_dict[tag] = 1
            
    #increment all existing tags count by one
    close_tag_count_dict.update((tag, occurances + incrementer2) for tag, occurances in close_tag_count_dict.items())
    
    sentences_count += (sentences_count*incrementer2)
    
    #add one to non-closing tags
    for tag in all_tags_dict.keys():
        try:
            val = close_tag_count_dict[tag]
        except KeyError as e:
            close_tag_count_dict[tag] = incrementer2
            sentences_count += incrementer2
            
    close_tag_count_dict.update((tag, (occurances*1.0)/sentences_count) for tag, occurances in close_tag_count_dict.items())
    return close_tag_count_dict


def buildTransitionMatrix(tagged_data, tags_dict):
    global incrementer
    tags = tags_dict.keys()
    tags.sort()
    
    tags_index_dict = {}
    tags_index_dict_reverse = {}
    for index, tag in enumerate(tags):
        tags_index_dict[tag] = index
        tags_index_dict_reverse[index] = tag
    
    tag_count = len(tags)
    
    feature_tags = {'PAGE_SEP' : [], 'URLS' : [], 'NUMERICS' : []}
    feature_counts = {'PAGE_SEP' : 0, 'URLS' : 0, 'NUMERICS' : 0}
    
    #Change this line to np.ones for add 1 smoothing
    transition_matrix = np.zeros(shape=(tag_count, tag_count))
    
    for line in tagged_data:
        prev_tag = None
        word_tag_pairs = line.strip().split(' ')
        
        for word_tag_pair in word_tag_pairs:
            word, tag = splitWordTag(word_tag_pair)
            
            if any(word.lower().endswith(last) for last in ('.com', '.net', '.org', '.edu')) or word.startswith('http') or word.startswith('www.'):
                feature_tags['URLS'].append(tag)
                feature_counts['URLS'] += 1

            elif word.count('=') > 10 or word.count('_') > 10 or word.count('*') > 10 or word.count('-') > 10 or word.count('+') > 10:
                feature_tags['PAGE_SEP'].append(tag)
                feature_counts['PAGE_SEP'] += 1
                
            elif [char.isdigit() for char in word].count(True) * 1.0 > len(word) * 0.4:
                feature_tags['NUMERICS'].append(tag)
                feature_counts['NUMERICS'] += 1
            else:
                pass
            
            if prev_tag is not None:
                transition_matrix[tags_index_dict[prev_tag]][tags_index_dict[tag]] += 1
            
            prev_tag = tag
    
    new_feature_tags = { 'PAGE_SEP' : 'xAMITx', 'URLS' : 'xAMITx', 'NUMERICS' : 'xAMITx' }
    try:
        for feature in feature_tags:
            possible_tags = feature_tags[feature]
            possible_tags_counter = Counter(possible_tags)
            possible_tags_ct =  possible_tags_counter.most_common(1)
            if len(possible_tags) > 0:
                best_possible_tag, tag_count = possible_tags_ct[0]
                if tag_count > feature_counts[feature] * 0.35:
                    new_feature_tags[feature] = best_possible_tag
    except:
        print "Fat raha hai"
        pass        
    transition_matrix = transition_matrix + incrementer
    
    probability_transition_matrix = transition_matrix/transition_matrix.sum(axis=1, keepdims=True)
    
    # print "Transition Values aree NaN : ", np.argwhere(np.isnan(probability_transition_matrix))
#     probability_transition_matrix[np.isnan(probability_transition_matrix)] = incrementer
#     probability_transition_matrix = np.log(probability_transition_matrix)
    return probability_transition_matrix.tolist(), tags_index_dict, tags_index_dict_reverse, new_feature_tags
        

def getUniqueWords(tagged_data):
    words = []
    for line in tagged_data:
        word_tag_pairs = line.strip().split(' ')
        
        for word_tag_pair in word_tag_pairs:
            word, tag = splitWordTag(word_tag_pair)
            words.append(word)
    return list(set(words))


def computeEmissionProbabilities(tagged_data, tags_dict):
    global incrementer
    tags = tags_dict.keys()
    tags.sort()
    
    words = getUniqueWords(tagged_data)
    words.sort()
    
    tags_index_dict = {}
    for index, tag in enumerate(tags):
        tags_index_dict[tag] = index
        
    words_index_dict = {}
    words_index_dict_reverse = {}
    for index, word in enumerate(words):
        words_index_dict[word] = index
        words_index_dict_reverse[index] = word
    
    tag_count = len(tags)
    word_count = len(words)
    
    # word_count + 1 => Last column for unseen words
    emission_matrix = np.zeros(shape=(tag_count, word_count + 1))
    
    for line in tagged_data:
        prev_tag = None
        word_tag_pairs = line.strip().split(' ')
        
        for word_tag_pair in word_tag_pairs:
            word, tag = splitWordTag(word_tag_pair)
            
            emission_matrix[tags_index_dict[tag]][words_index_dict[word]] += 1
            
            prev_tag = tag
    #increment 1 in all the elements so that the last col for unseen words have non zero values
#     emission_matrix = emission_matrix + incrementer
    probability_emission_matrix = emission_matrix/emission_matrix.sum(axis=1, keepdims=True)
    # print "Emission Values are NaN : ", np.argwhere(np.isnan(probability_emission_matrix))
    probability_emission_matrix[np.isnan(probability_emission_matrix)] = incrementer
#     probability_emission_matrix = np.log(probability_emission_matrix)
    return probability_emission_matrix.tolist(), tags_index_dict, words_index_dict, words_index_dict_reverse


def printEmissionProbabilities(count):
    counter = 0
    global probability_emission_matrix, tags_index_dict, words_index_dict
    word_count = len(words_index_dict.keys())
    tag_count = len(tags_index_dict.keys())
    for word, word_index in words_index_dict.iteritems():
        for tag, tag_index in tags_index_dict.iteritems():
            if probability_emission_matrix[tag_index][word_index] != 0:
                print tag, " => ", word, ' => ', probability_emission_matrix[tag_index][word_index]
                counter += 1
                if counter > count:
                    return


def writeModelToFile(probability_transition_matrix, opening_probabilities, closing_probabilities, probability_emission_matrix, tags_index_dict, words_index_dict, new_feature_tags, most_used_tag):
    total_tags = len(tags_index_dict.keys())
    total_words = len(words_index_dict.keys())
        
    lineCounter = 8
    text = ''
    
    text += '---------------------TransitionMatrix---------------------' + '\n'
    lineCounter += 1
    tr_start_line_number = lineCounter
    tr_end_line_number = tr_start_line_number
    for row in range(len(probability_transition_matrix)):
        row_text = ''
        for col in range(len(probability_transition_matrix[0])):
            row_text += str(probability_transition_matrix[row][col]) + '\t'
        row_text = row_text.strip()
        text += row_text + '\n'
        tr_end_line_number += 1
    
    text += '---------------------EmissionMatrix---------------------' + '\n'
    
    em_start_line_number = tr_end_line_number + 1
    em_end_line_number = em_start_line_number
    for row in range(len(probability_emission_matrix)):
        row_text = ''
        for col in range(len(probability_emission_matrix[0])):
            row_text += str(probability_emission_matrix[row][col]) + '\t'
        row_text = row_text.strip()
        text += row_text + '\n'
        em_end_line_number += 1
        
    text += '---------------------OpeningClosingProbabilities---------------------' + '\n'
    
    oc_start_line_number = em_end_line_number + 1
    oc_end_line_number = oc_start_line_number
    for tag in opening_probabilities:
        tag_details = tag + '\t' + str(opening_probabilities[tag]) + '\t' + str(closing_probabilities[tag]) + '\t' + str(tags_index_dict[tag]) + '\n'
        text += tag_details
        oc_end_line_number += 1
    
    text += '---------------------Words---------------------' + '\n'
    
    wi_start_line_number = oc_end_line_number + 1
    wi_end_line_number = wi_start_line_number
        
    for word in words_index_dict:
        word_details = word + '\t' + str(words_index_dict[word]) + '\n'
        text += word_details
        wi_end_line_number += 1
        
    text += '---------------------AdditionalFeatures---------------------' + '\n'
        
    af_start_line_number = wi_end_line_number + 1
    af_end_line_number = af_start_line_number
    
    for feature_name in new_feature_tags.keys():
        text += feature_name + '\t' + new_feature_tags[feature_name] + '\n'
        af_end_line_number += 1

    text += '---------------------MostUsedTag---------------------' + '\n'

    mut_start_line_number = af_end_line_number + 1
    mut_end_line_number = mut_start_line_number

    text += most_used_tag + '\n'

    mut_end_line_number += 1
    
    header = ''
    header += 'total_tags:' + str(total_tags) + '\n'
    header += 'total_words:' + str(total_words) + '\n'
    header += 'tranistion_matrix:' + str(tr_start_line_number) + ':' + str(tr_end_line_number) + '\n'
    header += 'emission_matrix:' + str(em_start_line_number) + ':' + str(em_end_line_number) + '\n'
    header += 'open_close_probabilities:' + str(oc_start_line_number) + ':' + str(oc_end_line_number) + '\n'
    header += 'word_indexes:' + str(wi_start_line_number) + ':' + str(wi_end_line_number) + '\n'
    header += 'additional_features:' + str(af_start_line_number) + ':' + str(af_end_line_number) + '\n'
    header += 'most_used_tag:' + str(mut_start_line_number) + ':' + str(mut_end_line_number) + '\n'
    
    text = header + text
    filename = 'hmmmodel.txt'
    with open(filename, 'w') as output_file:
        output_file.write(text)



if __name__ == '__main__':
    tagged_data = getFileFromCommandLine()
    # tagged_data = getFileContents('data/en_train_tagged.txt')
    tags_dict = getUniqueTags(tagged_data)

    opening_probabilities = getOpenProbabilities(tagged_data, tags_dict)
    closing_probabilities = getCloseProbabilities(tagged_data, tags_dict)

    probability_transition_matrix, tags_index_dict, tags_index_dict_reverse, new_feature_tags = buildTransitionMatrix(tagged_data, tags_dict)

    probability_emission_matrix, tags_index_dict, words_index_dict, words_index_dict_reverse = computeEmissionProbabilities(tagged_data, tags_dict)

    most_used_tag = max(tags_dict, key=tags_dict.get)

    writeModelToFile(probability_transition_matrix, opening_probabilities, closing_probabilities, probability_emission_matrix, tags_index_dict, words_index_dict, new_feature_tags, most_used_tag)
    # print "Done"
