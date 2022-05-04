import pickle
# from recording_language_classification import Recording_language_classification as Rec
from language_inference import Recording_language_classification as Rec

path_to_example = 'ar_100.pkl'
def get_packs():
    with open(path_to_example, 'rb') as f:
        pack = pickle.load(f)
        return pack[2]



def main():
    # sample = get_packs()
    ans = Rec.get_string_of_ans('best_model.pth','audio_files/f_m2.wav')
    print("Expected result:   French")
    print(ans)
    ans = Rec.get_string_of_ans('best_model.pth', 'audio_files/eng_f1.wav')
    print("Expected result:   English")
    print(ans)
    ans = Rec.get_string_of_ans('best_model.pth', 'audio_files/female.wav')
    print("Expected result:   English")
    print(ans)
    ans = Rec.get_string_of_ans('best_model.pth', 'audio_files/male.wav')
    print("Expected result:   English")
    print(ans)
    ans = Rec.get_string_of_ans('best_model.pth', 'audio_files/male (1).wav')
    print("Expected result:   English")
    print(ans)
    ans = Rec.get_string_of_ans('best_model.pth', 'audio_files/h_orig.wav')
    print("Expected result:   English")
    print(ans)


if __name__ == '__main__':
    main()


# bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
# model = bundle.get_model()
# def get_sample(lang):
#     dataset = load_dataset("common_voice", lang, split="train", streaming=True)
#     dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16_000))
#     dataset_iter = iter(dataset)
#     data = next(dataset_iter)
#     array_of_wav = data["audio"]["array"]
#     sub_array = array_of_wav[0:48000]
#     tensor_data = torch.tensor([sub_array])
#     sample, _ = model(tensor_data)
#     return sample