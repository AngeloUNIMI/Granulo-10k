import re
import numpy as np
from pathlib import Path

# Regex patterns
FILENAME_RE = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.txt")
ITER_RE = re.compile(r"Iteration n\. (\d+)")
MAE_RE = re.compile(
    r"Error \(MAE\)\. Height: ([\d.]+); Width: ([\d.]+); Thickness: ([\d.]+)"
)
PCT_RE = re.compile(
    r"Error \(%\)\. Height: ([\d.]+)%; Width: ([\d.]+)%; Thickness: ([\d.]+)%"
)


def process_log_file(path: Path):
    rows = []
    current_iter = None
    in_testing = False

    with path.open("r") as f:
        for line in f:
            line = line.strip()

            m_iter = ITER_RE.match(line)
            if m_iter:
                current_iter = int(m_iter.group(1))
                in_testing = False
                continue

            if line == "Testing":
                in_testing = True
                continue

            if in_testing:
                m_mae = MAE_RE.search(line)
                if m_mae:
                    h_mae, w_mae, t_mae = map(float, m_mae.groups())

                m_pct = PCT_RE.search(line)
                if m_pct:
                    h_pct, w_pct, t_pct = map(float, m_pct.groups())
                    rows.append([
                        current_iter,
                        h_mae, h_pct,
                        w_mae, w_pct,
                        t_mae, t_pct
                    ])
                    in_testing = False

    data = np.array(rows, dtype=float)
    mean = data[:, 1:].mean(axis=0)
    std = data[:, 1:].std(axis=0)

    return rows, mean, std


def write_results(path: Path, rows, mean, std):
    out_file = path.parent / "results_formatted.txt"

    with out_file.open("w") as f:
        f.write(
            "Iteration\tHeight (MAE)\tHeight (%)\t"
            "Width (MAE)\tWidth (%)\t"
            "Thickness (MAE)\tThickness (%)\n"
        )

        for r in rows:
            f.write(
                f"{int(r[0])}\t"
                f"{r[1]:.2f}\t{r[2]:.2f}\t"
                f"{r[3]:.2f}\t{r[4]:.2f}\t"
                f"{r[5]:.2f}\t{r[6]:.2f}\n"
            )

        # Final row: AVG (mean ± std)
        f.write(
            "AVG\t"
            f"{mean[0]:.2f} ± {std[0]:.2f}\t{mean[1]:.2f} ± {std[1]:.2f}\t"
            f"{mean[2]:.2f} ± {std[2]:.2f}\t{mean[3]:.2f} ± {std[3]:.2f}\t"
            f"{mean[4]:.2f} ± {std[4]:.2f}\t{mean[5]:.2f} ± {std[5]:.2f}\n"
        )


def main():
    for txt_file in Path(".").rglob("*.txt"):
        if FILENAME_RE.fullmatch(txt_file.name):
            rows, mean, std = process_log_file(txt_file)
            write_results(txt_file, rows, mean, std)


if __name__ == "__main__":
    main()
