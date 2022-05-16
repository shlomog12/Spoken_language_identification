
from language_inference import Recording_language_classification as Rec





def main():
    ans = Rec.get_string_of_ans('best_model.pth', '../data/sample.wav')
    print(ans)


if __name__ == '__main__':
    main()