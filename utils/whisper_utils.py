import whisper


def generate_subtitles(video_file, output_srt, translate=False, whisper_model="large"):
    model = whisper.load_model(whisper_model)
    task = "translate" if translate else "transcribe"
    result = model.transcribe(video_file, task=task)

    with open(output_srt, "w", encoding="utf-8") as srt_file:
        for i, segment in enumerate(result["segments"], start=1):
            start = format_time(segment["start"])
            end = format_time(segment["end"])
            text = segment["text"].strip()
            srt_file.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
