import numpy as np
import librosa
from matplotlib import pyplot as plt
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip, AudioFileClip
import cv2
import libfmp.b
import libfmp.c3
import libfmp.c4
from scipy.ndimage import gaussian_filter1d


def create_video_with_audio(frames, audio_path, output_path, fps):
    """
    Creates and saves a video with the given list of frames and the audio file.

    Args:
        frames (List[np.ndarray]): List of frames to write to the video.
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the output video file.
        fps (int): Frames per second of the output video.
    """
    video_clip = ImageSequenceClip(frames, fps=fps)
    video_clip = video_clip.set_audio(AudioFileClip(audio_path))
    video_clip.write_videofile(output_path, codec='libx264', audio_codec='libmp3lame')

# -----------------------------------------------------------------------------------------------
def create_frames(pixelMatrix, audio_dur_seconds, fps_video, width, height):
    """
	Create frames for the video 

	Args:
	    pixelMatrix (numpy.ndarray): The pixel matrix for the frames.
	    audio_dur_seconds (float): The duration of the audio in seconds.
	    fps_video (int): The frames per second for the video.
	    width (int): The width of the frames.
	    height (int): The height of the frames.

	Returns:
	    list: A list of frames for the video.
	"""

    image = pixelMatrix.copy()

    # Resize while maintaining the aspect ratio
    image = cv2.resize(image, (pixelMatrix.shape[1], height), interpolation=cv2.INTER_NEAREST_EXACT)
    print(image.shape)
    background = np.zeros((height, width, 3), dtype=np.uint8)

    number_frames = audio_dur_seconds*fps_video
    number_frames = int(np.rint(number_frames))
    print(number_frames)
    print(type(number_frames))
    
    frames = []
    half = width // 2

    start_img, end_img = 0, half
    start_frame, end_frame = half, width

    # for short audios
    if pixelMatrix.shape[1] < half:
        end_img = pixelMatrix.shape[1]
        end_frame = half + pixelMatrix.shape[1]

    # here the frames are composed
    for i in tqdm(range(number_frames)):

        start_frame -= 1
        end_img += 1

        # once the image reaches the left side
        if start_frame == -1:
            start_frame = 0
            start_img += 1

        # once image starts getting exhausted on the right
        if end_img > pixelMatrix.shape[1]:
            end_frame -= 1

        frame = background.copy()

        frame[:,start_frame:end_frame,:] = image[:,start_img:end_img,:]
        
        frame[:,half-2:half,:].fill(255)    #white line
        frame[:,half-7:half+3,:] = gaussian_filter1d(frame[:,half-5:half+5,:], sigma=20.0, axis=0, mode='reflect')

        frames.append(frame)

    return frames

# --------------------------------------------------------------------------------

def color_row(rows):
    """
    Modify each row of the input array by multiplying it with the corresponding color.
    
    Parameters:
    rows (list): The input list of rows
    
    Returns:
    list: The colored list of rows
    """
    colors = [[23,190,207],[214,39,40], [255,127,14], [44,160,44],  [188,189,34], [148,103,189], [127,127,127], [107,69,61], [23,190,207]]
         
    for i in range(len(rows)):
        for c in range(3):
            rows[i][:,:,c] *= colors[i][c]
    return rows


def draw_structure_row_detailed(segment_family, size):
    """
    Renders a detailed structure row based on the segment family and size

    Parameters:
    - segment_family: a list of tuples representing the start and end indices of the segment
    - size: int, the size of the structure row

    Returns:
    - np.ndarray, the computed structure row with rounded corners
    """
    scale = 10
    extend = scale // 2
    size *= scale
    structure_row = np.zeros((100, size, 3), dtype=np.uint8)
    for seg in segment_family:
        start = seg[0] * scale - extend + 5
        end = seg[1] * scale + extend - 1
        # round corners:
        structure_row[:, start+12:end-12] = 1
        structure_row[1:99, start+9:end-9] = 1
        structure_row[2:98, start+7:end-7] = 1
        structure_row[3:97, start+6:end-6] = 1
        structure_row[4:96, start+5:end-5] = 1
        structure_row[5:95, start+4:end-4] = 1
        structure_row[6:94, start+3:end-3] = 1
        structure_row[7:93, start+2:end-2] = 1
        structure_row[9:91, start+1:end-1] = 1
        structure_row[12:88, start:end] = 1

    return structure_row



def visualize_segment_partitions(segment_partitions, SSM_len, show = True):
    """
    Outputs an image of the segment partitions

    Args:
    - segment_partitions (List): List of segment partitions
    - SSM_len (int): Length of the SSM matrix
    - show (bool): Flag to show the visualization

    Returns:
    - pixelArt (np.ndarray): the image
    """
    print("Creating pixel art...")
    rows = []
    for partition in segment_partitions:
        rows.append(draw_structure_row_detailed(partition, SSM_len))
    
    rows = color_row(rows)
    pixelArt = np.concatenate(rows)
    pixelArt = cv2.GaussianBlur(pixelArt, (5, 5), 0)
    if show:
        plt.figure(figsize=(20, 3))
        plt.imshow(pixelArt)
        
    return pixelArt

# --------------------------------------------------------------------------------
def find_missing_segments(total_range, segment_partitions, min_segment_len):
    """Find segments that haven't been included in segment_partitions.

    Args:
        total_range (int): Total range from 0 to N.
        segment_partitions (list): List containing lists of ranges [start, end].
        min_segment_len (int): Minimum length of uncovered segments to include.

    Returns:
        uncovered_segments (list): List of uncovered segments.
    """
    all_segments = set(range(total_range + 1))
    
    # Flatten the segment_partitions and create a set of covered segments
    for partition in segment_partitions:
        for segment in partition:
            all_segments -= set(range(segment[0], segment[1] + 1))

    # Convert the remaining segments into a list of ranges
    uncovered_segments = []
    start, end = None, None
    for val in sorted(all_segments):
        if start is None:
            start, end = val, val
        elif val == end + 1:
            end = val
        else:
            if end - start + 1 >= min_segment_len:
                uncovered_segments.append([start, end])
            start, end = val, val

    # Handle the last segment
    if start is not None and end - start + 1 >= min_segment_len:
        uncovered_segments.append([start, end])

    return uncovered_segments

# -------------------------------------------------------------------------------

def get_segment_family(seg, S):
    """
    Computes the segment family for a given segment on the input spectrogram.

    Parameters:
    - seg: tuple, representing the start and end indices of the segment
    - S: numpy array, SSM

    Returns:
    - list, the computed segment family as a list
    """
    
    
    S_seg = S[:, seg[0]:seg[1]+1]
    D, score = libfmp.c4.compute_accumulated_score_matrix(S_seg)
    path_family = libfmp.c4.compute_optimal_path_family(D)
    segment_family, coverage2 = libfmp.c4.compute_induced_segment_family_coverage(path_family)

    return segment_family.tolist()

# ----------------------------------------------------------------------------------------------

def partition_audio_into_similarity_segments(file_path, min_segment_len_seconds = 7, penalty = -2, thresh_chroma=0.3, thresh_ssm=0.7):
    """
    Generates a partition of an audio file into similarity segments.

    Parameters:
    - file_path (str): The path to the audio file.
    - min_segment_len_seconds (int, optional): The minimum length of each segment in seconds.
    - penalty (int, optional): The penalty value for the SSM matrix. Defaults to -2.
    - thresh_chroma (float, optional): Remove chroma features values below this threshold.
    - thresh_ssm (float, optional): Remove SSM values below this threshold.

    Returns:
    - segments (list): A list of segment families, where each segment family is a list of tuples or ranges.
    - SSM_shape[0] (int): The size of the SSM matrix.
    - x_duration (float): The duration of the audio file.
    """
    
    segments = []
    downsampling = 7

    # Create SSM
    print('Computing SSM Matrix...')
    
    x, x_duration, X, Fs_feature, SSM = compute_sm_from_filename(file_path, L=5, H=downsampling, penalty=penalty, thresh_chroma=thresh_chroma, thresh_ssm=thresh_ssm)
    

    fps_video = Fs_feature * downsampling
    min_segment_len = round(Fs_feature * min_segment_len_seconds)


    fig = plot_matrix(SSM, figsize=(4, 4))

    print('Computing fitness...')
    # First Scape Plot
    SP_all = libfmp.c4.compute_fitness_scape_plot(SSM)
    SP = SP_all[0]
    SP_threshed = SP.copy()
    SP_threshed[0:min_segment_len, :] = 0

    print('Creating segments...')
    # Segment and it's family with highest fitness
    seg = libfmp.c4.seg_max_sp(SP_threshed)
    #seg = libfmp.c4.seg_max_sp(SP)
    segment_family = get_segment_family(seg, SSM)
    segments.append(segment_family)

    #fig, ax, im = libfmp.c4.visualize_scape_plot(SP_threshed, Fs_feature, figsize=(6, 4))
    SSM_masked = SSM.copy()
    while True:
    # make the original S matrix zeros for thumbnailed segments
        for i in segment_family:
            SSM_masked[i[0]:i[1]+1, :]=-20
            SSM_masked[:,i[0]:i[1]+1]=-20
        np.fill_diagonal(SSM_masked, np.diagonal(SSM))

        SP_all = libfmp.c4.compute_fitness_scape_plot(SSM_masked)
        SP = SP_all[0] #Takes the SP_fitness


        SP_threshed = SP.copy()
        SP_threshed[0:min_segment_len, :] = 0

        seg = libfmp.c4.seg_max_sp(SP_threshed)

        if seg == [0, 0]:
            break

        segment_family = get_segment_family(seg, SSM_masked)
        segments.append(segment_family)
        #fig, ax, im = libfmp.c4.visualize_scape_plot(SP_threshed, Fs_feature, figsize=(6, 4))

        #rows.append(create_structure_row(seg, S))
    segments.append(find_missing_segments(SSM.shape[0], segments, min_segment_len))
    return segments, SSM.shape[0], x_duration

# -----------------------------------------------------------------------------------------------
def compute_sm_from_filename(fn_wav, L=21, H=5, penalty=-2, thresh_chroma=0.3, thresh_ssm=0.7):
    
    # Waveform
    audio, Fs = librosa.load(fn_wav)
    audio_duration = audio.shape[0] / Fs

    # Chroma Feature Sequence and SSM (10 Hz)
    C = librosa.feature.chroma_stft(y=audio, sr=Fs, tuning=0, norm=2, hop_length=2205, n_fft=4410)
    Fs_C = Fs / 2205
    #plot_matrix(C, figsize=(20,2), cmap_compression = -1, title='Chromagram')


    chromagram_masked_indices = C < thresh_chroma
    C[chromagram_masked_indices] = 0
    #plot_matrix(C, figsize=(20,2), cmap_compression = -1, title='Chromagram')

    # Chroma Feature Sequence and SSM
    X, Fs_feature = libfmp.c3.smooth_downsample_feature_sequence(C, Fs_C, filt_len=L, down_sampling=H)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)

    SSM = compute_sm_dot(X,X)

    #plot_matrix(S1, figsize=(6,6), cmap_compression = -1, title='Chromagram')

    SSM_smooth = librosa.segment.path_enhance(SSM, 10, max_ratio=19.5)
    #plot_matrix(S, figsize=(6,6), cmap_compression = -1, title='Chromagram')

    SSM_thresh  = np.where(SSM_smooth < thresh_ssm, penalty, SSM_smooth)

    return audio, audio_duration, X, Fs_feature, SSM_thresh


# --------------------------------------------------------------------------------------------------
def compute_sm_dot(X, Y):
    """Computes similarty matrix from feature sequences using dot (inner) product

    Notebook: C4/C4S2_SSM.ipynb

    Args:
        X (np.ndarray): First sequence
        Y (np.ndarray): Second Sequence

    Returns:
        S (float): Dot product
    """
    S = np.dot(np.transpose(X), Y)
    return S

# --------------------------------------------------------------------------------------------------
def plot_matrix(similarity_matrix, figsize = (6, 6), cmap_compression = -1, title='Self-Similarity Matrix'):
    
    cmap = libfmp.b.compressed_gray_cmap(cmap_compression)
    #cmap = 'gray_r'
    plt.figure(figsize=figsize)
    librosa.display.specshow(similarity_matrix, cmap=cmap, x_axis='frames', y_axis='frames')
    plt.title(title)
    plt.colorbar()
    plt.show()


# -----------------------------------------------------------------------------------------------

