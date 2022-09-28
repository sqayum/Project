import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import unicodedata
from functools import reduce
import re

from nltk.probability import FreqDist
from nltk.tokenize import NLTKWordTokenizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer



CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "can not",
    "'cause": "because",
    "cha": "you",
    "coulda": "could have",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "dunno": "do not know",
    "gimme": "give me",
    "gonna": "going to",
    "gotta": "got to",
    "gotcha": "got you",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "imma": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "kinda": "kind of",
    "lemme": "let me",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "not've": "not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "outta": "out of",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shoulda": "should have",
    "should've": "should have",
    "shouldn't": "should not",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wanna": "want to",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "woulda": "would have",
    "would've": "would have",
    "wouldn't": "would not",
    "tryna": "trying to",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "u": "you",
    "ya": "you"}

CONVERSATIONAL_ABBREVIATION_MAP = {
    "afaik": "as far as i know",
    "bc": "because",
    "bfn": "bye for now",
    "brb": "be right back",
    "btw": "by the way",
    "dm": "direct message",
    "dyk": "did you know",
    "fomo": "fear of missing out",
    "fb": "facebook",
    "fml": "fuck my life",
    "fr": "for real",
    "ftf": "face to face",
    "ftl": "for the loss",
    "ftw": "for the win",
    "fwd": "forward",
    "fwiw": "for what it is worth",
    "fyi": "for your information",
    "gtg": "got to go",
    "gtr": "got to run",
    "hifw": "how i feel when",
    "hmb": "hit me back",
    "hmu": "hit me up",
    "hth": "hope that helps",
    "idc": "i do not care",
    "idk": "i do not know",
    "ikr": "i know right",
    "ily": "i love you",
    "imho": "in my humble opinion",
    "imo": "in my opinion",
    "irl": "in real life",
    "jk": "just kidding",
    "lmao": "laughing my ass off",
    "lmk": "let me know",
    "lol": "laughing out loud",
    "nbd": "no big deal",
    "nm": "not much",
    "nfw": "no fucking way",
    "nsfw": "not safe for work",
    "nvm": "nevermind",
    "omfg": "oh my fucking God",
    "omg": "oh my God",
    "omw": "on my way",
    "ppl": "people",
    "rly": "really",
    "rofl": "rolling on the floor laughing",
    "sfw": "safe for work",
    "smh": "shaking my head",
    "stfu": "shut the fuck up",
    "tbh": "to be honest",
    "tfw": "that feeling when",
    "tgif": "thank God its Friday",
    "tmi": "too much information",
    "tldr": "too long did not read",
    "wbu": "what about you",
    "wtf": "what the fuck",
    "wth": "what the hell",
    "ty": "thank you",
    "txt": "text",
    "yolo": "you only live once",
    "yw": "your welcome",
    "zomg": "oh my God"}

TWITTER_TECHNICAL_ABBREVIATION_MAP = {
    "DM": "direct message",
    "CT": "cuttweet",
    "RT": "retweet",
    "MT": "modified tweet",
    "HT": "hat tip",
    "CC": "carbon-copy",
    "CX": "correction",
    "FB": "Facebook",
    "LI": "LinkedIn",
    "YT": "YouTube"}


def print_completion_message(*, start_msg=None, end_msg=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal start_msg
            nonlocal end_msg
            arg_str = ', '.join([str(arg) for arg in args] + [f"{key}={value}" for key, value in kwargs.items()])
            if start_msg is None:
                start_msg = f"EXECUTING <{func.__name__}({arg_str})> ......"
            print(f"{start_msg} ......", end='')
            result = func(*args, **kwargs)
            if end_msg is None:
                end_msg = f" COMPLETE"
            print(end_msg)
            return result
        return wrapper
    return decorator



def _gather_tokens(tokenized_documents, *, by_document=False):
    for token_list in tokenized_documents:
        if by_document:
            token_list = set(token_list)
        for token in token_list:
            yield token


def _get_document_frequencies(corpus,
                                   *,
                                   document_col,
                                   label_col,
                                   label_name=None,
                                   N=None):

    if label_name is None:
        tokenized_documents = corpus[document_col]
    else:
        tokenized_documents = corpus.loc[corpus[label_col] == label_name, document_col].reset_index(drop=True)

    if isinstance(tokenized_documents.values[0], str):
        tokenized_documents = tokenized_documents.apply(str.split)

    gathered_tokens = _gather_tokens(tokenized_documents, by_document=True)
    frequency_df = pd.DataFrame(FreqDist(gathered_tokens).items(), columns=['word', 'frequency'])
    frequency_df.sort_values('frequency', inplace=True)
    if N is None:
        N = corpus.shape[0]
    total_num_tokens = frequency_df["frequency"].sum()
    tokens = frequency_df["word"].tail(N)
    normalized_frequencies = [round((frequency / total_num_tokens), 3) for frequency in frequency_df["frequency"].tail(N)]

    return {token: frequency for token, frequency in zip(tokens, normalized_frequencies)}


def plot_document_frequencies(corpus,
                                  *,
                                  document_col,
                                  label_col,
                                  label_name=None,
                                  N=20,
                                  figsize=None,
                                  filepath=None):

    frequency_dict = _get_document_frequencies(corpus,
                                                      document_col=document_col,
                                                      label_col=label_col,
                                                      label_name=label_name,
                                                      N=N)
    if figsize is None:
        fig, ax = plt.subplots(figsize=(15,15))
    else:
        fig, ax = plt.subplots(figsize=figsize)
    tokens = list(frequency_dict.keys())
    normalized_frequencies = list(frequency_dict.values())
    ax.barh(tokens, normalized_frequencies)
    ax.set(title=f'Normalized Document Frequencies ({label_name})')
    fig.tight_layout()

    if filepath is not None:
        fig.savefig(filepath)


def plot_label_frequencies(corpus,
                              *,
                              document_col,
                              label_col,
                              figsize=(8,7),
                              barwidth=0.5,
                              filepath=None):

    label_counts = corpus.groupby(label_col).count()[document_col]
    label_frequencies = label_counts.apply(lambda count: count / label_counts.sum())

    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 15

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(label_frequencies.index, label_frequencies.values, width=barwidth)
    ax.set(title="Label Frequencies (Normalized)");

    if filepath is not None:
        fig.savefig(filepath)


def average_token_length(corpus, *, document_col):
    if isinstance(corpus[document_col][0], str):
        tokenized_documents = corpus[document_col].apply(str.split)
    else:
        tokenized_documents = corpus[document_col]
    return round(tokenized_documents.map(len).mean(), 2)



def get_english_stopwords():
    stopwords = set()
    with open("data/english_stopwords.txt") as file_iter:
        for word in file_iter.readlines():
            stopwords.add(word.strip())
    return stopwords


def get_corpus_stopwords(corpus,
                             *,
                             document_col,
                             label_col,
                             threshold):

    stopwords_by_label = {}
    for label_name in corpus[label_col].unique():
        label_frequency_dict = _get_document_frequencies(corpus,
                                                                 document_col=document_col,
                                                                 label_col=label_col,
                                                                 label_name=label_name)

        label_stopwords = {word for word in label_frequency_dict if label_frequency_dict[word] >= threshold}
        stopwords_by_label[label_name] = label_stopwords
    return reduce(lambda x,y: x&y, stopwords_by_label.values())


def regex_scan(corpus,
                 *,
                 col_name,
                 pattern,
                 flags=None,
                 append_to_corpus=False,
                 new_col_name=None):

    if flags is None:
        pattern = re.compile(pattern)
    else:
        pattern = re.compile(pattern, flags=flags)
    result_col = corpus[col_name].str.contains(pattern, na=False)
    if not append_to_corpus:
        return result_col
    else:
        assert isinstance(new_col_name, str)
        corpus[new_col_name] = result_col



@print_completion_message(start_msg="Stripping accents")
def strip_accents(corpus, *, document_col):
    def _strip_accents(text):
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    corpus[document_col] = corpus[document_col].apply(_strip_accents)


@print_completion_message(start_msg="Stripping newline characters")
def strip_newline_characters(corpus, *, document_col):
    newline_character_regex = r"\n+"
    newline_character_pattern = re.compile(newline_character_regex)
    def _strip_newline_characters(text):
        return newline_character_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_newline_characters)


@print_completion_message(start_msg="Stripping Twitter handles")
def strip_twitter_handles(corpus, *, document_col):
    twitter_handle_regex = r"@\w+"
    twitter_handle_pattern = re.compile(twitter_handle_regex)
    def _strip_twitter_handles(text):
        return twitter_handle_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_twitter_handles)


@print_completion_message(start_msg="Stripping Twitter hashtags")
def strip_twitter_hashtags(corpus, *, document_col):
    twitter_hashtag_regex = r"#\w+"
    twitter_hashtag_pattern = re.compile(twitter_hashtag_regex)
    def _strip_twitter_hashtags(text):
        return twitter_hashtag_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_twitter_hashtags)


@print_completion_message(start_msg="Stripping Twitter technical abbreviations")
def strip_twitter_technical_abbreviations(corpus, *, document_col):
    twitter_technical_abbreviation_regex = r"(\bRT\b)|(\bCT\b)|(\bDM\b)|(\bMT\b)|(\bHT\b)|(\bCC\b)|(\bCX\b)|(\bFB\b)|(\bLI\b)|(\bYT\b)"
    twitter_technical_abbreviation_pattern = re.compile(twitter_technical_abbreviation_regex)
    def _strip_twitter_technical_abbreviations(text):
        return twitter_technical_abbreviation_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_twitter_technical_abbreviations)


@print_completion_message(start_msg="Stripping web addresses")
def strip_web_addresses(corpus, *, document_col):
    web_address_regex = r"(https?\:\/\/\S*)|(w{3}\.\S+\.(com|org|gov|edu)\S*)|(\S*\.(com|org|gov|edu))"
    web_address_pattern = re.compile(web_address_regex, flags=re.IGNORECASE)
    def _strip_web_addresses(text):
        return web_address_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_web_addresses)

@print_completion_message(start_msg="Stripping HTML entities")
def strip_html_entities(corpus, *, document_col):
    html_entity_regex = r"(&\S+;)+"
    html_entity_pattern = re.compile(html_entity_regex)
    def _strip_html_entities(text):
        return html_entity_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_html_entities)


@print_completion_message(start_msg="Expanding contractions")
def expand_contractions(corpus, *, document_col, contraction_mapping=CONTRACTION_MAP):
    SKIP_CONTRACTIONS = ["'cause", "she'd", "she'll", "he'll", "it's", "we'd", "we'll", "we're"]
    contraction_map_items = CONTRACTION_MAP.copy().items()
    for contraction, expanded_contraction in contraction_map_items:
        if contraction not in SKIP_CONTRACTIONS:
            CONTRACTION_MAP.update({contraction.replace('\'', ''): expanded_contraction})
    def _expand_contractions(text):
        def expand_match(contraction):
            match = contraction.group(0)
            expanded_match = contraction_mapping[match.lower()]
            if match[0].isupper():
                expanded_match = expanded_match[0].upper() + expanded_match[1:]
            return expanded_match
        contractions_regex = "(" + '|'.join('\\b' + contraction + "\\b" for contraction in sorted(contraction_mapping.keys(), key=lambda x: len(x))) + ")"
        contractions_pattern = re.compile(contractions_regex)
        expanded_text = contractions_pattern.sub(expand_match, text)
        if contractions_pattern.search(expanded_text) is None:
            return expanded_text
        return contractions_pattern.sub(expand_match, expanded_text)
    corpus[document_col] = corpus[document_col].apply(_expand_contractions)


@print_completion_message(start_msg="Stripping non-alphabetical characters")
def strip_special_characters(corpus, *, document_col):
    special_character_regex = r"[^a-zA-Z\s]+"
    special_character_pattern = re.compile(special_character_regex)
    def _strip_special_characters(text):
        return special_character_pattern.sub(' ', text)
    corpus[document_col] = corpus[document_col].apply(_strip_special_characters)


@print_completion_message(start_msg="Expanding abbreviations")
def expand_abbreviations(corpus, *, document_col, abbreviation_mapping=CONVERSATIONAL_ABBREVIATION_MAP):
    def _expand_abbreviations(text):
        def expand_match(abbreviation):
            match = abbreviation.group(0)
            expanded_match = abbreviation_mapping[match.lower()]
            return expanded_match
        abbreviations_regex = "(" + '|'.join('\\b' + abbreviation + "\\b" for abbreviation in abbreviation_mapping.keys()) + ")"
        abbreviations_pattern = re.compile(abbreviations_regex)
        expanded_text = abbreviations_pattern.sub(expand_match, text)
        return expanded_text
    corpus[document_col] = corpus[document_col].apply(_expand_abbreviations)


@print_completion_message(start_msg="Tokenizing")
def tokenize(corpus, *, document_col, tokenizer):
    tokenizer = tokenizer(reduce_len=True, preserve_case=False)
    def _tokenize(text):
        return tokenizer.tokenize(text)
    corpus[document_col] = corpus[document_col].apply(_tokenize)


@print_completion_message(start_msg="Removing single character tokens")
def remove_single_character_tokens(corpus, *, document_col):
    def _remove_single_character_tokens(tokens):
        return list(filter(lambda token: len(set(token)) > 1, tokens))
    corpus[document_col] = corpus[document_col].apply(_remove_single_character_tokens)


@print_completion_message(start_msg="Removing stopwords")
def remove_stopwords(corpus,
                        *,
                        document_col,
                        label_col,
                        remove_corpus_stopwords,
                        corpus_stopword_threshold):

    stopwords = get_english_stopwords()

    if label_col is not None and remove_corpus_stopwords:
        corpus_stopwords = get_corpus_stopwords(corpus,
                                                       document_col=document_col,
                                                       label_col=label_col,
                                                       threshold=corpus_stopword_threshold)
        stopwords = stopwords.union(corpus_stopwords)
    def _remove_stopwords(tokens):
        return list(filter(lambda token: token not in stopwords, tokens))
    corpus[document_col] = corpus[document_col].apply(_remove_stopwords)


@print_completion_message(start_msg="Removing common names")
def remove_common_names(corpus, *, document_col, N=1500):
    common_names = set()
    with open("data/common_names.txt") as file_iter:
        for _ in range(N):
            common_names.add(next(file_iter).lower().strip())
    def _remove_commmon_names(tokens):
        return list(filter(lambda token: token not in common_names, tokens))
    corpus[document_col] = corpus[document_col].apply(_remove_commmon_names)


@print_completion_message(start_msg="Lemmatizing")
def lemmatize(corpus, *, document_col):
    lemmatizer = WordNetLemmatizer()
    def _lemmatize(tokens):
        def get_wordnet_tag(treebank_tag):
            if treebank_tag == 'ADJ':
                return wordnet.ADJ
            elif treebank_tag == 'VERB':
                return wordnet.VERB
            elif treebank_tag == 'ADV':
                return wordnet.ADV
            else:
                return wordnet.NOUN
        tagged_tokens = pos_tag(tokens, tagset="universal")
        return [lemmatizer.lemmatize(token[0], get_wordnet_tag(token[1])) for token in tagged_tokens]
    corpus[document_col] = corpus[document_col].apply(_lemmatize)

@print_completion_message(start_msg="Removing empty token lists")
def remove_empty_token_lists(corpus, *, document_col):
    corpus = corpus.loc[corpus[document_col].map(len) > 0]
    corpus.reset_index(drop=True, inplace=True)
    return corpus

def normalize_corpus(corpus,
                        *,
                        document_col,
                        label_col=None,
                        remove_corpus_stopwords=False,
                        corpus_stopword_threshold=0.0001):

    if isinstance(corpus[document_col][0], list):
        corpus[document_col] = corpus[document_col].apply(' '.join)
    if remove_corpus_stopwords:
        assert label_col is not None
    strip_accents(corpus, document_col=document_col)
    strip_newline_characters(corpus, document_col=document_col)
    strip_twitter_handles(corpus, document_col=document_col)
    strip_twitter_hashtags(corpus, document_col=document_col)
    strip_twitter_technical_abbreviations(corpus, document_col=document_col)
    strip_web_addresses(corpus, document_col=document_col)
    strip_html_entities(corpus, document_col=document_col)
    expand_contractions(corpus, document_col=document_col)
    strip_special_characters(corpus, document_col=document_col)
    expand_abbreviations(corpus, document_col=document_col)
    tokenize(corpus, document_col=document_col)
    remove_single_character_tokens(corpus, document_col=document_col)

    remove_common_names(corpus, document_col=document_col)
    lemmatize(corpus, document_col=document_col)
    remove_stopwords(corpus,
                        label_col=label_col,
                        document_col=document_col,
                        remove_corpus_stopwords=remove_corpus_stopwords,
                        corpus_stopword_threshold=corpus_stopword_threshold)
    corpus = remove_empty_token_lists(corpus, document_col=document_col)
    corpus.rename(columns={document_col: "tokens"}, inplace=True)
    return corpus