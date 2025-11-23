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

    def frequency_duration_pairs(self, precision: int = 2, merge_threshold_hz: float | None = None, rest_token: str = "REST",) -> list[list[float | str]]:
        """
        Returns [[frequency_hz_or_rest_token, duration_seconds], ...] for consecutive segments.
        Consecutive NaNs become REST segments
        Consecutive frames of approximatel same frequency become note segments
        """
        if self.pitch_hz is None:
            raise RuntimeError("Pitch data not loaded yet.")

        frame_duration = self.hop_length / self.sr
        pairs: list[list[float | str]] = []

        current_kind: str | None = None  # "note" or "rest"
        current_freq: float | None = None
        frames_in_segment = 0

        for value in self.pitch_hz:
            if np.isnan(value):
                # in rest frame
                if current_kind == "note" and frames_in_segment > 0:
                    # close note
                    pairs.append([round(current_freq, precision), round(frames_in_segment * frame_duration, 5)])
                    frames_in_segment = 0
                    current_freq = None
                current_kind = "rest"
                frames_in_segment += 1
            else:
                freq = round(float(value), precision)

                if current_kind == "rest" and frames_in_segment > 0:
                    # close rest
                    pairs.append([rest_token, round(frames_in_segment * frame_duration, 5)])
                    frames_in_segment = 0
                current_kind = "note"

                if current_freq is None:
                    # start new note, begin list
                    current_freq = freq
                    frames_in_segment = 1
                elif freq == current_freq:
                    # same freq as previous frame, extend segment
                    frames_in_segment += 1
                else:
                    # note changed, close old note, start new one
                    pairs.append([current_freq, round(frames_in_segment * frame_duration, 5)])
                    current_freq = freq
                    frames_in_segment = 1

        if frames_in_segment > 0:
            if current_kind == "note":
                pairs.append([round(current_freq, precision), round(frames_in_segment * frame_duration, 5)])
            elif current_kind == "rest":
                pairs.append([rest_token, round(frames_in_segment * frame_duration, 5)])

        if merge_threshold_hz is not None:
            i = 0
            while i < len(pairs) - 1:
                f1, d1 = pairs[i]
                f2, d2 = pairs[i + 1]

                if (isinstance(f1, (int, float)) and isinstance(f2, (int, float)) and abs(f1 - f2) < merge_threshold_hz):
                    pairs [i][1] += d2
                    pairs.pop(i+1)
                    continue
                if f1 == rest_token and f2 == rest_token:
                    pairs[i][1] += d2
                    pairs.pop(i + 1)
                    continue
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
        else:
            return ""


if __name__ == "__main__":
    pitch_path = Path("backend/images/parselmouth_pitch_hz.npy")
    analyzer = CreateSheetMusic(pitch_path)
    analyzer.load_pitch()
    print(analyzer.summarize())
    freq_duration = analyzer.frequency_duration_pairs()
    print("Frequency-duration pairs:")
    print(freq_duration)