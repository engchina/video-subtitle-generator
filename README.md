# Video Subtitle Generator

This project automatically generates and adds subtitles to a video file using Whisper for speech recognition and
Qwen1.5_110B for translation.

## Setup

1. Create a conda environment:
   ```
   conda activate video-subtitle-generator
   ```

2. Activate the environment:
   ```
   conda activate video-subtitle-generator
   ```

3. Install required packages:
   ```
   pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
   # pip install -i https://pypi.tuna.tsinghua.edu.cn/simple torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2
   ```

4. Test Cuda
   ```
   python -c "import torch;print(torch.cuda.is_available());"
   ```

## Usage

Run the script with:

```
python main.py input_video.mp4 ./output/output_video.mp4 final_video.mp4 
```

Use the `--use-trim` flag to use 30 seconds instead of full video:
Use the `--use-translation` flag to use translation instead of original language(default: from Japanese to Chinese):

```
python main.py input_video.mp4 ./output/output_video.mp4 --use-trim --use-translation
```

## Other knowledge

刷新字体缓存

```
fc-cache -f -v
```

验证字体安装

```
fc-list :lang=zh
```

使用 ffmpeg 指定字体

```
ffmpeg -i input_video.mp4 -vf subtitles=subtitle.srt:force_style='FontName=SimSun' -c:v libx264 -c:a copy output_video.mp4
```


