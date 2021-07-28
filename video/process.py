import os
fns=os.listdir('./')
for fn in fns:
    if fn.endswith('llff_final.mp4'):
        os.system(fr'C:\Users\liuyuan\Desktop\eval\bin\ffmpeg.exe -i .\{fn} -vcodec libx264 -crf 24 compress\{fn}')