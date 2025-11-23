from pathlib import Path
import numpy as np

class CreateSheetMusic:
    """Loads the saved pitch contour and converts it to frequency-duration pairs."""

    def __init__(self, pitch_file: Path, sr: int = 44100, hop_length: int = 512) -> None:
        self.pitch_file = pitch_file
        self.pitch_hz: np.ndarray | None = None
        self.sr = sr
        self.hop_length = hop_length

    def load_pitch(self) -> np.ndarray:
        if not self.pitch_file.exists():
            raise FileNotFoundError(
                f"Pitch file not found: {self.pitch_file}. Run "
                "`python backend/tests/test_visualize_pitch.py` first."
            )

        self.pitch_hz = np.load(self.pitch_file)
        return self.pitch_hz

    def summarize(self) -> str:
        if self.pitch_hz is None:
            raise RuntimeError("Pitch data not loaded yet.")

        valid = self.pitch_hz[~np.isnan(self.pitch_hz)]
        if valid.size == 0:
            return "Pitch file loaded, but it only contains NaNs."

        summary = (
            f"Loaded {self.pitch_hz.size} pitch frames\n"
            f"Valid frames : {valid.size}\n"
            f"Min freq (Hz): {valid.min():.2f}\n"
            f"Max freq (Hz): {valid.max():.2f}\n"
            f"Mean freq(Hz): {valid.mean():.2f}"
        )
        return summary

    def frequency_duration_pairs(self, precision: int = 2) -> list[list[float]]:
        """
        Returns [[frequency_hz, duration_seconds], ...] for consecutive segments.
        NaN values break segments and are skipped.
        """
        if self.pitch_hz is None:
            raise RuntimeError("Pitch data not loaded yet.")

        frame_duration = self.hop_length / self.sr
        pairs: list[list[float]] = []

        current_freq: float | None = None
        frames_in_segment = 0

        for value in self.pitch_hz:
            if np.isnan(value):
                if current_freq is not None and frames_in_segment > 0:
                    pairs.append(
                        [current_freq, round(frames_in_segment * frame_duration, 5)]
                    )
                    current_freq = None
                    frames_in_segment = 0
                continue

            freq = round(float(value), precision)
            if current_freq is None:
                current_freq = freq
                frames_in_segment = 1
            elif freq == current_freq:
                frames_in_segment += 1
            else:
                pairs.append(
                    [current_freq, round(frames_in_segment * frame_duration, 5)]
                )
                current_freq = freq
                frames_in_segment = 1

        if current_freq is not None and frames_in_segment > 0:
            pairs.append([current_freq, round(frames_in_segment * frame_duration, 5)])

        #need to chop many pairs off and add their durations to the previous pair
        #get index of first pair we need, find index of the next pair with a delta_frequecy >= 9
        #add the duration of the pairs between two indexes, remove them all, then the new index should just be i+1
        
        '''MUST  use background colors to distinguish between the same note played twice in a row'''

        #remove pairs with a delta_frequecy < 9
        i = 0
        while i < len(pairs) - 1:
            if abs(pairs[i][0] - pairs[i + 1][0]) < 9:
                pairs[i][1] += pairs[i + 1][1]
                pairs.pop(i + 1)
            else:
                i += 1

        return pairs


        
    def get_background_color(self, freq: float) -> str:
        if freq < 100:
            return "red"
        elif freq < 200:
            return "orange"
        elif freq < 300:
            return "yellow"
        elif freq < 400:
            return "green"


if __name__ == "__main__":
    pitch_path = Path("backend/images/parselmouth_pitch_hz.npy")
    analyzer = CreateSheetMusic(pitch_path)
    analyzer.load_pitch()
    print(analyzer.summarize())
    freq_duration = analyzer.frequency_duration_pairs()
    print("Frequency-duration pairs:")
    print(freq_duration)