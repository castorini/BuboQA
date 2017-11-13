from torchtext import data

class SQdataset(data.TabularDataset):
    @classmethod
    def splits(cls, text_field, label_field,
               train='train.txt', validation='valid.txt', test='test.txt'):
        path = '../../data/processed_simplequestions_dataset/'
        return super(SQdataset, cls).splits(
            path, train, validation, test,
            format='TSV', fields=[('id', None), ('sub', None), ('entity', None), ('relation', label_field),
                                  ('obj', None), ('text', text_field), ('ed', None)]
        )