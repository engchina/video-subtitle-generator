import subprocess


def trim_video(input_video, output_video, duration=30):
    command = [
        "ffmpeg", "-i", input_video,
        "-t", str(duration),
        "-c", "copy",
        output_video,
        "-y"
    ]
    subprocess.run(command, check=True)


def add_subtitles_to_video(input_video, subtitle_file, output_video, font_size=18, margin_v=2):
    # Alignment=2：底部居中
    # MarginV=2：离底部的距离
    command = [
        "ffmpeg", "-i", input_video,
        "-vf", f"subtitles={subtitle_file}:force_style='FontSize={font_size},Alignment=2,MarginV={margin_v}'",
        "-max_muxing_queue_size", "1024",
        "-c:a", "copy",
        output_video,
        "-y"
    ]
    subprocess.run(command, check=True)
