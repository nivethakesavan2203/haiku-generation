import pandas as pd
import haiku_generation.src.utils.preprocessing as prepro


class HaikuDataset():
    """
    haiku dataset -- works on two types of datasets:

    - the first one is a fairly messy csv file;
    - the other one is from kaggle and much cleaner
    """
    def __init__(self, filename, is_kaggle=False):
        super(HaikuDataset, self).__init__()

        if not is_kaggle:
            df = pd.read_csv(filename, names=['haiku'])
        else:
            df = pd.read_csv(filename, names=['nan', 'first', 'second', 'third', 'source', 'hash'])
            df['haiku'] = df['first'] + ' ' + df['second'] + ' ' + df['third']
            df['haiku'] = df['haiku'].astype(str)

        # do preprocessing here:


        df['haiku'] = df['haiku'].apply(lambda x: prepro.preprocess_text(x))
        # df['haiku'] = df['haiku'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
        # df['haiku'] = df['haiku'].apply(lambda x: x.replace('\n', ' '))
        self.df = df.dropna()

    def __len__(self):
        """
        gets the length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, index):
        """
        gets an item from the haikus dataset.
        """
        poem = self.df['haiku'].values[index]
        # # clean up and parse the poem
        # cleaned_poem = re.sub(' +', ' ', poem)
        # word_list = re.sub("[^\w]", " ",  cleaned_poem).split()
        return poem

    def get_all_poems(self):
        return self.df['haiku'].values


if __name__ == '__main__':
    # the below line is the Kaggle dataset
    haikus = HaikuDataset('/home/peasant98/Documents/haiku-generation/haiku_generation/datasets/all_haiku.csv', is_kaggle=True)

    # the below line is the "non-kaggle" dataset
    # haikus = HaikuDataset('src/datasets/Haikus1.csv', is_kaggle=False)

    # __len__ in action
    print(len(haikus))

    # __getitem__ in action
    for i in range(100):
        print(haikus[i])

    print(haikus.get_all_poems())