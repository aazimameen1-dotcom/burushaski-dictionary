import os
import wave
import numpy as np
from moviepy.editor import AudioFileClip

def analyze():
    audio_dir = r"c:\Users\aazim\Downloads\dictionary\burushaski-dictionary\burushaski_dictionary_dataset_part2\mp3_audio"
    out_path = r"c:\Users\aazim\Downloads\dictionary\burushaski-dictionary\analysis_output_part2.txt"
    if not os.path.exists(audio_dir):
        print("Audio directory not found!")
        return

    files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".mp3")])
    
    print("Analyzing sound segments in all part 2 files...\n")
    out_f = open(out_path, "w", encoding="utf-8")
    
    for filename in files:
        file_path = os.path.join(audio_dir, filename)
        temp_wav = "temp_segments.wav"
        
        try:
            # Export to WAV
            clip = AudioFileClip(file_path)
            clip.write_audiofile(temp_wav, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
            duration = clip.duration
            clip.close()
            
            # Read WAV
            with wave.open(temp_wav, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_data = wf.readframes(n_frames)
                
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
                
            data = np.frombuffer(raw_data, dtype=np.int16)
            if n_channels > 1:
                data = data.reshape(-1, n_channels).mean(axis=1)
                
            data = data.astype(np.float32) / 32768.0
            
            # window size 50ms (800 samples), hop size 10ms (160 samples)
            window_size = 800
            hop_size = 160
            
            rms_profile = []
            times = []
            
            for i in range(0, len(data) - window_size, hop_size):
                window = data[i : i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_profile.append(rms)
                times.append((i + window_size / 2) / 16000.0)
                
            rms_profile = np.array(rms_profile)
            times = np.array(times)
            
            if rms_profile.max() > 0:
                rms_profile /= rms_profile.max()
                
            # Find active segments (threshold = 0.05)
            active = rms_profile > 0.05
            segments = []
            in_segment = False
            start_t = 0
            
            # Simple segmentation
            for t, act in zip(times, active):
                if act and not in_segment:
                    start_t = t
                    in_segment = True
                elif not act and in_segment:
                    segments.append((start_t, t))
                    in_segment = False
            if in_segment:
                segments.append((start_t, times[-1]))
                
            # Merge close segments (less than 150ms gap)
            merged = []
            for seg in segments:
                if not merged:
                    merged.append(seg)
                else:
                    last = merged[-1]
                    if seg[0] - last[1] < 0.15:
                        merged[-1] = (last[0], seg[1])
                    else:
                        merged.append(seg)
                        
            # Filter out very short noise (< 150ms)
            clean_segments = [seg for seg in merged if seg[1] - seg[0] >= 0.15]
            
            seg_strs = ", ".join([f"{s:.2f}s-{e:.2f}s" for s, e in clean_segments])
            line = f"{filename} (duration: {duration:.2f}s): [{seg_strs}]"
            print(line)
            out_f.write(line + "\n")
            out_f.flush()
            
        except Exception as e:
            print(f"Error on {filename}: {e}")
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    out_f.close()

if __name__ == "__main__":
    analyze()
