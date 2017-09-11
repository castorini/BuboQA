from torchtext import data
import os

def my_tokenizer():
    return lambda text: [tok for tok in text.split()]

class SimpleQADataset(data.TabularDataset):

    dirname = 'data'

    @classmethod
    def splits(cls, text_field, label_field, root='./data',
               train='train.txt', validation='valid.txt', test='test.txt'):
        prefix_name = 'annotated_fb_entity_'
        return super(SimpleQADataset, cls).splits(
            os.path.join(root, prefix_name), train, validation, test,
            format='TSV', fields=[('question', text_field), ('label', label_field)]
        )
