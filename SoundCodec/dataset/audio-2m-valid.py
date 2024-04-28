from datasets import load_dataset, Audio, load_from_disk


def load_data():
    # audio_dataset = load_dataset("json", data_files='/raid/codeclm/music/encodec/egs/codecllm_encodec/valid/data.jsonl')
    # def map_file_to_id(data):
    #     data['id'] = "".join(data['path'].split("/")[-1:])
    #     return data
    # audio_dataset = audio_dataset.map(map_file_to_id)
    # audio_dataset = audio_dataset.cast_column("path", Audio())
    # audio_dataset = audio_dataset.rename_column("path", "audio")
    audio_dataset = load_from_disk('audio-2m-valid')
    return audio_dataset
