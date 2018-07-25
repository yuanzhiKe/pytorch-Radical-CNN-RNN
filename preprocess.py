import pickle, jaconv
import collections


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def strip_ideographic_desctription(text):
    # Ideographic Description Characters, U+2FF0 - U+2FFF
    ideographic__description__characters = "⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻"
    translator = str.maketrans("", "", ideographic__description__characters)
    return text.translate(translator)


def get_all_character_bukken(filename="IDS-UCS-Basic.txt", ideographic_desc=True):
    bukkens = []
    characters = []  # actually chars
    character_bukken = {}
    for i, line in enumerate(open(filename, "r").readlines()):
        if line[0] != "U":  # not start with U+XXXX means it is not a character
            continue
        line = line.split()
        character = line[1]
        components = line[2]
        if not ideographic_desc:
            components = strip_ideographic_desctription(components)
        bukken = []
        while ";" in components:
            bukken.append(components[:components.find(";") + 1])
            components = components[components.find(";") + 1:]
        while len(components) > 1:
            bukken.append(components[0])
            components = components[1:]
        bukken.append(components)
        characters.append(character)
        character_bukken[characters.index(character)] = bukken
        if len(bukken) == 1 and bukken[0] == character:
            bukkens.append(character)

    def expand_bukken(bukken):
        expanded_bukken = []
        for b in bukken:
            if b in bukkens:
                expanded_bukken.append(bukkens.index(b))
            else:
                if b in characters:
                    expanded_bukken.append(expand_bukken(character_bukken[characters.index(b)]))
                else:
                    bukkens.append(b)  # will contain Ideographic Description Characters if ideographic_desc = True
                    expanded_bukken.append(bukkens.index(b))
        return expanded_bukken

    for i_character, i_bukken in character_bukken.items():
        b_list = expand_bukken(i_bukken)
        b_list = flatten(b_list)
        character_bukken[i_character] = b_list
    return characters, bukkens, character_bukken


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


def text_to_comps(text, character_vocab, character_bukken, addition_translate, comp_width, skip_unknown=False, transform=None):
    # TODO: ONGOING
    text = basic_preprocess(text)
    # expanding every character with 3 components
    ch2id = {}
    for i, w in enumerate(full_vocab):
        ch2id[w] = i
    int_text = []
    # print(text)
    for c in text:
        # print(c)
        try:
            i = ch2id[c]
        except KeyError:
            # print("Unknown Character: ", c)
            if skip_unknown:
                continue  # skip unknown characters
            else:
                i = 1  # assign to unknown characters
        # print(i)
        if real_vocab_number < i < preprocessed_char_number:
            comps = chara_bukken_revised[i]
            if shuffle == "flip":
                comps = comps[::-1]
            # print(comps)
            if len(comps) >= comp_width:
                int_text += comps[:comp_width]
            else:
                int_text += comps + [0] * (comp_width - len(comps))
        else:
            if shuffle=="random":
                if i<real_vocab_number:
                    i = (i+20)%real_vocab_number
            int_text += [i] + [0] * (comp_width - 1)
    return int_text


def get_vocabs(dataset):
    # TODO: ONGOING
    pass