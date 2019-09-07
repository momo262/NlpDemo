
def add_ngram(sequences, token_indice, ngram_range=2):
    new_sequences = []
    for input_list in sequences:
        new_list = []
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
new_sequences = add_ngram(sequences, token_indice, ngram_range=2)
print(new_sequences)

