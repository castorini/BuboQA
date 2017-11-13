import unicodedata
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

def processed_text(text):
    text = text.replace('\\\\','')
    #stripped = strip_accents(text.lower())
    stripped = text.lower()
    toks = tokenizer.tokenize(stripped)
    return " ".join(toks)

def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')

def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        in_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    if in_str == 'fb:m.07s9rl0':
        in_str = 'fb:m.02822'
    if in_str == 'fb:m.0bb56b6':
        in_str = 'fb:m.0dn0r'
    # Manual Correction
    if in_str == 'fb:m.01g81dw':
        in_str = 'fb:m.01g_bfh'
    if in_str == 'fb:m.0y7q89y':
        in_str = 'fb:m.0wrt1c5'
    if in_str == 'fb:m.0b0w7':
        in_str = 'fb:m.0fq0s89'
    if in_str == 'fb:m.09rmm6y':
        in_str = 'fb:m.03cnrcc'
    if in_str == 'fb:m.0crsn60':
        in_str = 'fb:m.02pnlqy'
    if in_str == 'fb:m.04t1f8y':
        in_str = 'fb:m.04t1fjr'
    if in_str == 'fb:m.027z990':
        in_str = 'fb:m.0ghdhcb'
    if in_str == 'fb:m.02xhc2v':
        in_str = 'fb:m.084sq'
    if in_str == 'fb:m.02z8b2h':
        in_str = 'fb:m.033vn1'
    if in_str == 'fb:m.0w43mcj':
        in_str = 'fb:m.0m0qffc'
    if in_str == 'fb:m.07rqy':
        in_str = 'fb:m.0py_0'
    if in_str == 'fb:m.0y9s5rm':
        in_str = 'fb:m.0ybxl2g'
    if in_str == 'fb:m.037ltr7':
        in_str = 'fb:m.0qjx99s'
    return in_str

def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri