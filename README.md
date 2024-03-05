# Music Structure Visualization
![Results Sample]()


### Goal: 

Split an audio recording into segments, group them into musically meaningful categories, and produce a pretty visualization.

### Pipeline:

1. Input: audio path.
2. Compute a [self-similarity matrix](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S2_SSM.html) with path enchancement and thresholding.
3. Compute the [scape plot](https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4S3_ScapePlot.html#Scape-Plot).
4. Repeat 2. and 3. till the returned scape plot is empty.
5. Find the missing segments and gorup them in the last row.
5. From the resulting list of segment partitions, build the main image for the video. Done pixel by pixel.
6. Create the list of frames for the video.
7. Output: video with audio.

Libraries: librosa, libfmp, OpenCV, MoviePy.
Skill up: Music Processing Analysis, Image & Video Processing.
