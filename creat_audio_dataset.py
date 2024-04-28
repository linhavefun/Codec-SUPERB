from datasets import *

audio_dataset = load_dataset("json", data_files='/raid/codeclm/music/encodec/egs/codecllm_encodec/valid/data.jsonl')
audio_dataset = audio_dataset.cast_column("path", Audio())
audio_dataset = audio_dataset.rename_column("path", "audio")
audio_dataset = audio_dataset['train'].add_column(name="id", column=[i for i in range(len(audio_dataset['train']))])
audio_dataset.save_to_disk("audio-2m-valid")
# audio_dataset.to_json('encodec_valid_set.jsonl')


# dataset = load_dataset("audiofolder", data_dir="/aifs4su/mmdata/rawdata/codeclm/music/audio-2m-mp3-32khz/")
# print(dataset['train'][0])