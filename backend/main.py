from pathlib import Path
import numpy as np
import cv2
from scipy.signal import find_peaks

class CreateSheetMusic:
    """Loads the saved pitch contour and converts it to frequency-duration pairs."""

    def __init__(self, pitch_file: Path, sr: int = 44100, hop_length: int = 512) -> None:
        self.pitch_file = pitch_file
        self.pitch_hz: np.ndarray | None = None
        self.sr = sr
        self.hop_length = hop_length
        self.grey_image_width: int | None = None

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

    def frequency_duration_pairs(
        self, precision: int = 2, frame_breaks: list[int] | None = None
    ) -> list[list[float]]:
        """
        Returns [[frequency_hz, duration_seconds], ...] for consecutive segments.
        NaN/0 Hz are treated as rests and encoded with 0 hz
        Forces splits at frame_breaks (boundaries) to distinguish repeated notes.
        Merges adjacent pairs with similar frequencies (< 9 Hz difference) only if not separated by boundaries.
        """
        if self.pitch_hz is None:
            raise RuntimeError("Pitch data not loaded yet.")

        frame_duration = self.hop_length / self.sr
        pairs: list[list[float]] = []
        # Track which pairs end at boundaries (should not be merged with next pair)
        ends_at_boundary: list[bool] = []

        current_freq: float | None = None
        frames_in_segment = 0

        # Filter out 0 and n_frames from frame_breaks (they're not real boundaries)
        frame_breaks = sorted(set(frame_breaks or []))
        frame_breaks = [b for b in frame_breaks if 0 < b < len(self.pitch_hz)]
        
        next_break_idx = 0
        next_break = (
            frame_breaks[next_break_idx] if next_break_idx < len(frame_breaks) else None
        )

        for frame_idx, value in enumerate(self.pitch_hz):
            is_rest = np.isnan(value) or value == 0
            freq = 0.0 if is_rest else round (float(value), precision)

            # Force split at boundary frames
            if next_break is not None and frame_idx >= next_break:
                if current_freq is not None and frames_in_segment > 0:
                    pairs.append(
                        [current_freq, round(frames_in_segment * frame_duration, 5)]
                    )
                    ends_at_boundary.append(True)  # This pair ends at a boundary
                current_freq = None
                frames_in_segment = 0
                # Move to next boundary (skip any boundaries we've already passed)
                while next_break_idx < len(frame_breaks) and frame_idx >= frame_breaks[next_break_idx]:
                    next_break_idx += 1
                next_break = (
                    frame_breaks[next_break_idx]
                    if next_break_idx < len(frame_breaks)
                    else None
                )

            if current_freq is None:
                current_freq = freq
                frames_in_segment = 1
            elif freq == current_freq:
                frames_in_segment += 1
            else:
                pairs.append(
                    [current_freq, round(frames_in_segment * frame_duration, 5)]
                )
                ends_at_boundary.append(False)  # Frequency change, not a boundary
                current_freq = freq
                frames_in_segment = 1

        if current_freq is not None and frames_in_segment > 0:
            pairs.append([current_freq, round(frames_in_segment * frame_duration, 5)])
            ends_at_boundary.append(False)  # Last pair doesn't end at boundary

        # Merge pairs with similar frequencies, but not across boundaries
        # A boundary exists between pair i and pair i+1 if pair i ends at a boundary
        i = 0
        while i < len(pairs) - 1:
            f1 = pairs[i]
            f2, d2 = pairs[i + 1]

            # keep rests explicit, only merge rest-rest
            if f1 == 0 and f2 == 0:
                if ends_at_boundary[i]:
                    i += 1
                    continue
                pairs[i][1] += d2
                if ends_at_boundary[i + 1]:
                    ends_at_boundary[i] = True
                pairs.pop(i + 1)
                ends_at_boundary.pop(i + 1)
                continue
            if f1 == 0 or f2 == 0:
                i += 1
                continue

            # Check if frequencies are similar
            if abs(f1 - f2) < 9:
                # Don't merge if pair i ends at a boundary (this means pair i+1 starts after a boundary)
                if ends_at_boundary[i]:
                    # Boundary exists between them - don't merge
                    i += 1
                else:
                    # Safe to merge - no boundary between them
                    pairs[i][1] += d2
                    # If pair i+1 ended at a boundary, pair i now ends at that boundary
                    if ends_at_boundary[i + 1]:
                        ends_at_boundary[i] = True
                    pairs.pop(i + 1)
                    ends_at_boundary.pop(i + 1)
            else:
                i += 1
        
        return pairs


    def sobel_edge_detector(self, image_path: Path = Path("backend/images/pitch.png"),
    output_name: str = "grey_pitch.png",) -> np.ndarray:
        '''Runs Sobel edge detection on the image and save a grayscale PNG.'''

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        #Apply Sobel filter
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

        #Calculate gradient magnitude and normalize to 0-255
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        max_val = magnitude.max()
        if max_val == 0:
            edges = np.zeros_like(magnitude, dtype=np.uint8)
        else:
            #percentile-based threshold as we need to keep weaker edges visible
            threshold = np.percentile(magnitude, 91)
            mask = magnitude >= threshold
            normalized = (magnitude / max_val) * 255
            edges = np.uint8(np.clip(normalized, 0, 255))
            edges[mask] = np.maximum(edges[mask], 180)  #boost strongest edges

        #Save grayscale edge image
        output_path = image_path.parent / output_name
        cv2.imwrite(str(output_path), edges)

        return edges


    def find_plot_bounds(self, image_path: Path = Path("backend/images/pitch.png")) -> tuple[int, int]:
        '''Detects the left and right bounds of the actual plot area (excluding margins).'''
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        
        # Hardcode left margin (y-axis labels take up ~60 pixels)
        left_edge = 60
        
        # Find right edge: look for last column with significant variation
        col_stds = image.std(axis=0)  # Standard deviation per column
        col_means = image.mean(axis=0)
        margin_sample_size = min(50, len(col_stds) // 10)
        margin_mean_value = col_means[:margin_sample_size].mean()
        
        right_edge = len(col_stds) - 1
        margin_sample_right = col_stds[-margin_sample_size:].mean()
        for i in range(len(col_stds) - 1, -1, -1):
            if col_stds[i] > margin_sample_right * 1.5 or col_means[i] < margin_mean_value * 0.9:
                right_edge = i
                break
        
        return left_edge, right_edge


    def find_grey_boundaries(self, image_path: Path = Path("backend/images/grey_pitch.png")) -> list[int]:
        '''Finds the boundaries of the grey image. By looking for vertical lines in the greyscale image.'''
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")

        self.grey_image_width = image.shape[1]
        col_str = image.mean(axis=0)

        peaks, _ = find_peaks(col_str, height=90, distance=3)
        if peaks.size == 0:
            peaks, _ = find_peaks(col_str, height=60, distance=2)
        if peaks.size == 0:
            raise RuntimeError(
                "No vertical boundaries detected; adjust thresholds or regenerate grey_pitch.png."
            )

        boundaries: list[int] = []
        tolerance = 3

        for peak in peaks:
            x = int(peak)
            if not (100 <= x <= self.grey_image_width - 100):
                continue

            if not boundaries:
                boundaries.append(x)
            elif abs(x - boundaries[-1]) <= tolerance:
                boundaries[-1] = (boundaries[-1] + x) // 2
            else:
                boundaries.append(x)

        return boundaries
        

if __name__ == "__main__":
    pitch_path = Path("backend/images/parselmouth_pitch_hz.npy")
    analyzer = CreateSheetMusic(pitch_path)
    analyzer.load_pitch()
    print(analyzer.summarize())
    analyzer.sobel_edge_detector()
    boundaries = analyzer.find_grey_boundaries()
    print("Detected boundaries:", boundaries)
    
    # Find the actual plot area bounds (excluding margins)
    plot_left, plot_right = analyzer.find_plot_bounds()
    plot_width = plot_right - plot_left
    print(f"Plot bounds: left={plot_left}, right={plot_right}, width={plot_width}")

    n_frames = analyzer.pitch_hz.size if analyzer.pitch_hz is not None else 0
    
    # Convert pixel boundaries to frame indices
    # Adjust boundaries relative to plot area: (boundary - left_margin) / plot_width
    frame_breaks_from_boundaries = [
        round(((boundary - plot_left) / plot_width) * n_frames)
        for boundary in boundaries
        if plot_left <= boundary <= plot_right
    ]
    
    # Filter out 0 and n_frames (they're not real split points)
    frame_breaks = sorted([
        fb for fb in frame_breaks_from_boundaries
        if 0 < fb < n_frames
    ])

    freq_duration = analyzer.frequency_duration_pairs(frame_breaks=frame_breaks)
    print("Frequency-duration pairs:")
    print(freq_duration)
