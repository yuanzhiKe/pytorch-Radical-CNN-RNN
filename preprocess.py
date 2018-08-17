import jaconv, os, pickle
import collections
from janome.tokenizer import Tokenizer as janome_tokenizer


def flatten(x):
    if not isinstance(x, str):
        if isinstance(x, collections.Iterable):
            return [a for i in x for a in flatten(i)]
        else:
            return [x]
    return [x]


def strip_ideographic_desctription(text):
    # Ideographic Description Characters, U+2FF0 - U+2FFF
    ideographic__description__characters = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"
    translator = str.maketrans("", "", ideographic__description__characters)
    return text.translate(translator)


def get_all_character_radical(filename="IDS-UCS-Basic.txt", ideographic_desc_chars=True,
                              expand_radicals=False, eoc=True):
    """
    get_all_character_radical
    :param filename: CHISE character information database file
    :param ideographic_desc_chars: if the dict contain ideographic description chracters
    :param expand_radicals: if expand radicals to finer-grained radicals
                            note that: such radical in the database is often phonetic.
                                       in some scenes, they should not be split into smaller ones as the semantic ones.
    :param eoc: if add <eoc> in the end of the radical list of each char.
                For variable length RNN, if I do not want to add it in later processing.
    :return:
    """
    radicals = ['<pad>', '<unk>', '<eoc>']  # pad, unknown, end of character
    characters = ['<pad>', '<unk>', '<eow>']  # actually chars. pad, unknown, end of word
    character_radical = {}
    print('Loading...')
    for i, line in enumerate(open(filename, "r").readlines()):
        if i % 1000 == 0:
            print(i, "lines processed.")
        if line[0] != "U":  # not start with U+XXXX means it is not a character
            continue
        line = line.split()
        character = line[1]
        components = line[2]
        if not ideographic_desc_chars:
            components = strip_ideographic_desctription(components)
        radical_list = []
        """
        tolerate unprintable radicals
        TODO: need to test :
        U+4E54  乔      ⿱夭&CDP-89AB;
        U+4E55  乕      ⿸𠂆&CDP-89ED;
        U+4E4C  乌      ⿹&CDP-89DE;一
        """
        unprintable_radical = ''
        for radical in components:
            if radical == ';':
                radical_list.append(unprintable_radical)
                unprintable_radical = ''  # reset
            elif radical == '\n':
                continue
            elif (radical == "&") or (unprintable_radical != ''):
                unprintable_radical += radical
            else:
                radical_list.append(radical)
        characters.append(character)
        character_radical[character] = radical_list
        if len(radical_list) == 1 and radical_list[0] == character and character not in radicals:
            radicals.append(character)  # add simple characters (not compound) into the radical list

    def map_radicals(some_radicals):
        """
        this sub function build radical vocab and update character_radical to idx2idx
        if expand_radicals is True, it expand every compound glyphs into simple ideograms recursively.
        :param some_radicals: raw radicals
        :return: radical idx
        """
        radical_x = []
        for r in some_radicals:
            if expand_radicals:
                # in this case, the radical vocab only contains non-compound symbols,
                # i.e. the list defined before this sub function
                if r in radicals:
                    # if r is non-compound symbols which is contained in the list defined before this function
                    radical_x.append(r)
                else:
                    if r in characters:
                        # in this case, r is compound
                        radical_x.append(map_radicals(character_radical[r]))
                    else:
                        # in this case, r is an ideographic desc char or an unprintable radical
                        radicals.append(r)  # r should be in the radical vocab too
                        radical_x.append(r)
            else:
                # in this case, they are what they are in the database
                if r not in radicals:
                    radicals.append(r)
                radical_x.append(r)
        return radical_x

    new_character_radical = {}  # use new container for final output to avoid bugs such as duplicating <eoc>, etc.
    # char_id to raw_radicals -> char_id to radical_x
    for i_character, i_radical in character_radical.items():
        if i_character == '<pad>':
            new_character_radical[i_character] = ['<pad>']
        b_list = map_radicals(i_radical)
        b_list = flatten(b_list)
        new_character_radical[i_character] = b_list
        if eoc:
            new_character_radical[i_character].append('<eoc>')
    return characters, radicals, new_character_radical


def get_all_character(filename="IDS-UCS-Basic.txt"):
    chars = []

    for i, line in enumerate(open(filename, "r").readlines()):
        if line[0] != "U":  # not start with U+XXXX means it is not a character
            continue
        line = line.split()
        char = line[1]
        chars.append(char)
    return chars


def basic_preprocess(text):
    # convert digital number and latin to hangaku
    text = jaconv.z2h(text, kana=False, digit=True, ascii=True)
    # convert kana to zengaku
    text = jaconv.h2z(text, kana=True, digit=False, ascii=False)
    # convert kata to hira
    text = jaconv.kata2hira(text)
    # lowercase
    text = text.lower()
    return text


class Vocab:
    """
    The tool class to store the vocabs and process text
    """

    def __init__(self, dataset, max_character_length=8, expand=False, keep_unseen_CHISEinfo=True, char_padding=True):
        """
        Constructor.
        :param dataset: the dataset to parse
        :param max_character_length: the length of the radical sequence of each character
        :param expand: if expand expandable radicals (for example: 吾 in 語). Note: for phonetic radicals, it is bad.
        :param keep_unseen_CHISEinfo:
        :param char_padding:
        """
        super(Vocab, self).__init__()
        self.tokenizer = janome_tokenizer()
        if expand:
            dbfilename = 'CHISE_Basic_Expand.pkl'
        else:
            dbfilename = 'CHISE_Basic_noExpand.pkl'
        if os.path.isfile(dbfilename):
            characters, radicals, character_radical = pickle.load(open(dbfilename, 'rb'))
        else:
            characters, radicals, character_radical \
                = get_all_character_radical(filename="IDS-UCS-Basic.txt", ideographic_desc_chars=True,
                                            expand_radicals=expand, eoc=False)
        assert max_character_length > 2  # comp_with < 3 is meaningless
        self.max_character_length = max_character_length
        # build and merge data vocab with local vocab
        self.characters, self.words, self.radicals, self.character_radical \
            = self.get_vocabs(dataset, characters, radicals, character_radical,
                              keep_unseen_CHISEinfo=keep_unseen_CHISEinfo, char_padding=char_padding)

    def get_vocabs(self, dataset, world_characters, world_radicals, world_character_radical, keep_unseen_CHISEinfo,
                   char_padding):
        """
        Read data set. build data set vocabs with existing local own vocabs.
        For characters and radicals: keep the ones appeared in the data set, drop the others
        For unseen characters in the CHISE character information db, add them.
        :param dataset: the text data set. should be a python iterator
        :param world_characters: the characters read from the CHISE character information db
        :param world_radicals:  the radicals read from the CHISE character information db
        :param world_character_radical: the char-radical mapping read from the CHISE character information db
        :return: vocab of characters, words, radicals; and char2radical mapping character_radical
        """
        characters = ['<pad>', '<unk>', '<eow>', '<eos>']
        words = ['<eos>']
        radicals = ['<pad>', '<unk>', '<eoc>']
        character_radical = {}
        for sentence in dataset:
            # tolerate list, tuple, dict input
            if not isinstance(sentence, str):
                if isinstance(sentence, list) or isinstance(sentence, tuple):
                    for item in sentence:
                        if isinstance(item, str):
                            sentence = item
                            break
                    if not isinstance(sentence, str):  # check if sentence is replaced with a string item
                        raise Exception("No sentence contained in: ", sentence)
                elif isinstance(sentence, dict):
                    for key, item in sentence.items():
                        if isinstance(item, str):
                            sentence = item
                            break
                    if not isinstance(sentence, str):  # check if sentence is replaced with a string item
                        raise Exception("No sentence contained in: ", sentence)
                else:
                    raise Exception("Unsupported data type: ", sentence.__class__)
            # now sentence is string
            # parse charcters
            for c in sentence:
                if c not in characters:
                    characters.append(c)
            # parse words. using janome with mecab-ipadic-2.7.0-20070801 as the lexicon
            for word in self.tokenizer.tokenize(sentence, wakati=True):
                if word not in words:
                    words.append(word)
        if keep_unseen_CHISEinfo:
            # keep unseen character_radical information for processing unseen characters
            # pad everything
            for character, radicals in world_character_radical.items():
                if char_padding:
                    character_radical[character] = self.pad_sequence(radicals, self.max_character_length, '<eoc>')
                else:
                    character_radical[character] = radicals
                    if character_radical[character][-1] != '<eoc>':
                        character_radical[character].append('<eoc>')
            # add the characters not in CHISE db
            for character in characters:
                if character not in character_radical.keys():
                    if char_padding:
                        character_radical[character] = self.pad_sequence([character], self.max_character_length,
                                                                         '<eoc>')
                    else:
                        character_radical[character] = [character, '<eoc>']
        else:
            # keep only the seen characters' char2radical mapping
            for character in characters:
                if character in world_character_radical.keys():
                    char_radicals = world_character_radical[character]
                    if char_padding:
                        character_radical[character] = self.pad_sequence(char_radicals, self.max_character_length,
                                                                         '<eoc>')
                    else:
                        character_radical[character] = char_radicals
                        if character_radical[character][-1] != '<eoc>':
                            character_radical[character].append('<eoc>')
                else:
                    if char_padding:
                        character_radical[character] = self.pad_sequence([character], self.max_character_length,
                                                                         '<eoc>')
                    else:
                        character_radical[character] = [character, '<eoc>']
        # collect radicals
        for _, local_radicals in character_radical.items():
            for local_radical in local_radicals:
                if local_radical not in radicals:
                    radicals.append(local_radical)
        # Fill <pad>'s radicals with <pad>. Otherwise, it will be like ['<pad>', '<eoc>', '<pad>', ...]
        # as we want assign zero vector to <pad>, non-zero vector to <eoc>,
        # ['<pad>', '<eoc>', '<pad>', ...] will make word-level padding non-zero.
        character_radical['<pad>'] = ['<pad>'] * self.max_character_length
        character_radical['<eow>'] = ['<eow>'] + ['<pad>'] * (self.max_character_length - 1)
        character_radical['<eos>'] = ['<eos>'] + ['<pad>'] * (self.max_character_length - 1)
        return characters, words, radicals, character_radical

    def pad_sequence(self, sequence, max_length, end_symbol, pad=None):
        if pad is None:
            pad = '<pad>'
        if isinstance(sequence, list):
            # pad the radical sequence to the same size
            if len(sequence) < max_length:
                if sequence[-1] != end_symbol:
                    sequence.append(end_symbol)
                sequence += [pad] * (max_length - len(sequence))
            elif len(sequence) > max_length:
                sequence = sequence[:max_length - 1] + [end_symbol]
            else:
                if sequence[-1] != end_symbol:
                    sequence[-1] = end_symbol
            return sequence
        else:
            raise Exception("Function pad_character expects list.")

    def text2radicalIdx(self, text):
        """
        convert a sentence or a word to seq of radicals
        Note that character-level padding has been done in get_vocab
        Note that <pad> is not constrained at index 0. please use radicals.index('<pad>') characters.index('<pad>')
             similarly, please use radicals.index('<eoc>') characters.index('<eoc>'), etc.
        :param text: a sentence or a word
        :return: radical sequence
        """
        assert self.character_radical is not None
        result = []
        for char in text:
            try:
                radicals = self.character_radical[char]
            except KeyError:
                print("Unseen Character: ", char)
                radicals = self.character_radical['<unk>']  # assign to unknown characters
            result += radicals
        return [self.radicals.index(x) for x in result]

    def sen2cr(self, sentence, hier=True):
        """
        parse and process sentence to words, which characters is a sequence of radicals.
        everything is padded instead of variable length, however, with <eow> symbol
        ex. [[1,2,3], [1,2,3], [1,2,3]]
             w1,      w2,      w3
        :param sentence: a sentence
        :return: char sequence each char is represented by a radical sequence, with <eow> symbol
        """
        # TODO: ONGOING
        pass

    def sen2wcr(self, sentence, hier=True):
        """
        parse and process sentence to words, which characters is a sequence of radicals.
        the sequence of each character is stored seperately.
        everything is padded instead of variable length, however, with <eow> symbol
        ex. [[1,2,3], [1,2,3], [1,2,3]]
             w1,      w2,      w3
        :param sentence: a sentence
        :return: char sequence each char is represented by a radical sequence, with <eow> symbol
        """
        # TODO: Can be implemented by detecting <eoc> in the result of sen2wr's hier mode
        pass

    def sen2wr(self, sentence, max_sentence_length=None, max_word_length=None, hier=True):
        """
        parse and process sentence to words, which words is a sequence of radicals.
        everything is padded instead of variable length, however, with <eow> symbol
        if hier is true, output ex. [[1,2,3], [1,2,3], [1,2,3]]
                                      w1,      w2,      w3
        else, output flat ex. [1, 2, 3, 1, 2, 3, 1, 2, 3]
                          w1,      w2,      w3
        :param sentence: a sentence
        :param hier: return hierarchical list or flat one
        :return: word sequence each word is represented by a radical sequence, with <eow> symbol
        """
        # tolerate list, tuple, dict input
        if not isinstance(sentence, str):
            if isinstance(sentence, list) or isinstance(sentence, tuple):
                for item in sentence:
                    if isinstance(item, str):
                        sentence = item
                        break
                if not isinstance(sentence, str):  # check if sentence is replaced with a string item
                    raise Exception("No sentence contained in: ", sentence)
            elif isinstance(sentence, dict):
                for key, item in sentence.items():
                    if isinstance(item, str):
                        sentence = item
                        break
                if not isinstance(sentence, str):  # check if sentence is replaced with a string item
                    raise Exception("No sentence contained in: ", sentence)
            else:
                raise Exception("Unsupported data type: ", sentence.__class__)
        # now sentence is string
        # parse words. using janome with mecab-ipadic-2.7.0-20070801 as the lexicon
        if hier:
            # Hierarchical seq
            if max_word_length is None:
                if max_sentence_length is None:
                    # no word padding and no sentence padding
                    return [self.text2radicalIdx(word) for word in self.tokenizer.tokenize(sentence, wakati=True)]
                else:
                    # TODO: only sentence padding
                    assert isinstance(max_sentence_length, int)
                    raise Exception("Only max_sentence_length has not been implemented yet.")
            else:
                assert isinstance(max_word_length, int)
                if max_sentence_length is None:
                    # TODO: only word padding
                    raise Exception("Only max_word_length has not been implemented yet.")
                else:
                    # word padding + sentence padding
                    result = []
                    # word padding
                    for word in self.tokenizer.tokenize(sentence, wakati=True):
                        result.append(self.pad_sequence(self.text2radicalIdx(word),
                                                        max_word_length * self.max_character_length,
                                                        self.radicals.index('<eow>'), pad=self.radicals.index('<pad>')))
                    # sentence padding
                    eos_word = [self.radicals.index('<eos>')] + \
                               [self.radicals.index('<pad>')] * (max_word_length * self.max_character_length - 1)
                    pad_word = [self.radicals.index('<pad>')] * (max_word_length * self.max_character_length)
                    return self.pad_sequence(result, max_sentence_length, eos_word, pad=pad_word)

        else:
            if max_sentence_length is None or max_word_length is None:
                raise Exception('Flat mode requires max_sentence_length and max_word_length')
            result = []
            for word in self.tokenizer.tokenize(sentence, wakati=True):
                result += self.pad_sequence(self.text2radicalIdx(word), max_word_length * self.max_character_length,
                                            self.radicals.index('<eow>'), pad=self.radicals.index('<pad>'))
            return self.pad_sequence(result, max_sentence_length * max_word_length * self.max_character_length,
                                     self.radicals.index('<eos>'), pad=self.radicals.index('<pad>'))
