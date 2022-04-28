import pickle
from recording_language_classification import Recording_language_classification as Rec


path_to_example = 'ar_100.pkl'
def get_packs():
    with open(path_to_example, 'rb') as f:
        pack = pickle.load(f)
        return pack[2]



def main():
    sample = get_packs()
    ans = Rec.get_string_of_ans(sample)
    print(ans)


if __name__ == '__main__':
    main()


