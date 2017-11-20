from torchtext import data

class SQdataset(data.TabularDataset):
    @classmethod
    def splits(cls, text_field, label_field,
               train='train.txt', validation='valid.txt', test='test.txt'):
        path = '../../data/processed_simplequestions_dataset/'
        return super(SQdataset, cls).splits(
            path=path, train=train, validation=validation, test=test,
            format='TSV', fields=[('id', None), ('sub', None), ('entity', None), ('relation', None),
                                  ('obj', None), ('text', text_field), ('ed', label_field)]
        )
