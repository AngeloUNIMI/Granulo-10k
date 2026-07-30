import csv
import numpy as np

class ResultsLogger:
    def __init__(self):
        self.results = []
        self.current_iteration = 0
    
    def add(self, height_mae, height_perc,
                  width_mae, width_perc,
                  thickness_mae, thickness_perc):
        
        self.current_iteration += 1
        
        self.results.append({
            "iteration": self.current_iteration,
            "height_mae": height_mae,
            "height_perc": height_perc,
            "width_mae": width_mae,
            "width_perc": width_perc,
            "thickness_mae": thickness_mae,
            "thickness_perc": thickness_perc
        })
    
    def compute_avg(self):

        return {
            "iteration": "AVG",
            "height_mae": np.mean([r["height_mae"] for r in self.results]),
            "height_perc": np.mean([r["height_perc"] for r in self.results]),
            "width_mae": np.mean([r["width_mae"] for r in self.results]),
            "width_perc": np.mean([r["width_perc"] for r in self.results]),
            "thickness_mae": np.mean([r["thickness_mae"] for r in self.results]),
            "thickness_perc": np.mean([r["thickness_perc"] for r in self.results]),
        }
    
    def save_csv(self, filename):
        
        new_headers_names = {
            "iteration": "Iteration",
            "height_mae": "Height (MAE)",
            "height_perc": "Height (%)",
            "width_mae": "Width (MAE)",
            "width_perc": "Width (%)",
            "thickness_mae": "Thickness (MAE)",
            "thickness_perc": "Thickness (%)"
        }

        avg = self.compute_avg()

        def format_value(key, value):
            if key == "iteration":  
                # Do NOT format iteration
                return value
            # Format all numeric values to 2 decimals
            if isinstance(value, (float, int)):
                return f"{value:.2f}"
            return value

        def format_row(row):
            return {
                new_headers_names[k]: format_value(k, v)
                for k, v in row.items()
            }

        formatted_rows = [format_row(r) for r in self.results]
        formatted_avg = format_row(avg)

        fieldnames = list(new_headers_names.values())

        # Save
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
            writer.writeheader()
            writer.writerows(formatted_rows)
            writer.writerow(formatted_avg)
        
        #print(f"Saved results + averages to {filename}")