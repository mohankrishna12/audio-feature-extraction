from pytube import YouTube
import pandas as pd

link = ''
yt = YouTube(link)
name = ''
start = '0:05'
end = '5:05'
genre = ''

import os
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# convert youtube time format into seconds
def get_sec(time_str):
    time_format = time_str.count(':')
    if (time_format == 1): # MM:SS
        m, s = time_str.split(':')
        return int(m) * 60 + int(s)
    else: # HH:MM:SS
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)

# get filename without extension
def path_leaf(path):
    head, tail = os.path.split(path)
    return os.path.splitext(tail)[0] or os.path.basename(head)

# read csv to generate dataset
current_video = ''
dst_folder = 'youtube/'
item = 0

index = 0



# download video

current_video = link
print('parsing video', current_video)

# download audio in original format
src = YouTube(link
    ).streams.filter(file_extension='mp4', only_audio=True).first( # get mp4 audio stream
    ).download(output_path=dst_folder, filename=name) # write stream to file
print('video downloaded')

# lossless conversion of mp4 to wav
dst = dst_folder + path_leaf(src) + '.wav'
sound = AudioSegment.from_file(src)
sound.export(dst, format="wav")

# determine time region for cuts
splice_len = 5
start_time = get_sec(start)
end_time = get_sec(end)

# make cuts by split length
for i in range(start_time, end_time - splice_len, splice_len):
    print('clip #:', item, 'clip start:', i, 'clip end:', i + splice_len, 'video start:', start_time, 'video end:', end_time)
    ffmpeg_extract_subclip(dst, i, end_time if (i + splice_len > end_time) else (i + splice_len), targetname='youtube/' + genre + '/' + name +'-'+str(i)+'.wav')
    item += 1
