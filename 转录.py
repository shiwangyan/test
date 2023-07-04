import whisper
from collections import defaultdict
import random


def detect_language(audio_file_path: str, whisper_model_name: str, samples_number=5):
    # load audio
    audio = whisper.load_audio(audio_file_path)

    # load the Whisper model
    model = whisper.load_model(whisper_model_name, download_root="models")

    # optimization: if the audio length is <= Whisper chunk length, then we will only take 1 sample
    # 优化：如果音频长度小于等于Whisper的分片长度，则我们只取1个样本。
    if len(audio) <= whisper.audio.CHUNK_LENGTH * whisper.audio.SAMPLE_RATE:
        samples_number = 1

    probabilities_map = defaultdict(list)

    for i in range(samples_number):
        # select a random audio fragment
        random_center = random.randint(0, len(audio))
        start = random_center - (whisper.audio.CHUNK_LENGTH // 2) * whisper.audio.SAMPLE_RATE
        end = random_center + (whisper.audio.CHUNK_LENGTH // 2) * whisper.audio.SAMPLE_RATE
        start = max(0, start)
        start = min(start, len(audio) - 1)
        end = max(0, end)
        end = min(end, len(audio) - 1)
        audio_fragment = audio[start:end]

        # pad or trim the audio fragment to match Whisper chunk length
        # 将音频片段填充或截断以匹配Whisper的分片长度。
        audio_fragment = whisper.pad_or_trim(audio_fragment)

        # extract the Mel spectrogram and detect the language of the fragment
        mel = whisper.log_mel_spectrogram(audio_fragment)
        _, _probs = model.detect_language(mel)
        for lang_key in _probs:
            probabilities_map[lang_key].append(_probs[lang_key])

    # calculate the average probability for each language
    for lang_key in probabilities_map:
        probabilities_map[lang_key] = sum(probabilities_map[lang_key]) / len(probabilities_map[lang_key])

    # return the language with the highest probability
    detected_lang = max(probabilities_map, key=probabilities_map.get)
    return detected_lang


if __name__ == '__main__':
    path = ""
    model = ""
    detect_language(path, model)


