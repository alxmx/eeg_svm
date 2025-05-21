"""
Parameter Adjustment Tool for EEG Mindfulness Index Calculator

This script provides an interface to adjust the parameters of the
mindfulness index calculator and see how they affect the results.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from eeg_mindfulness_index import (
    process_eeg_file,
    calculate_mi,
    classify_behavioral_state,
    WINDOW_SEC,
    OVERLAP,
    MI_WEIGHTS,
    THRESHOLDS
)

class MindfulnessParameterAdjuster:
    def __init__(self, root):
        self.root = root
        self.root.title("Mindfulness Index Parameter Adjuster")
        self.root.geometry("1200x800")
        
        # Variables
        self.eeg_file = tk.StringVar()
        self.results = None
        self.features = None
        self.current_weights = MI_WEIGHTS.copy()
        self.current_thresholds = THRESHOLDS.copy()
        
        # Weight sliders
        self.weight_vars = {}
        for feature in MI_WEIGHTS:
            self.weight_vars[feature] = tk.DoubleVar(value=MI_WEIGHTS[feature])
        
        # Threshold sliders
        self.threshold_vars = {}
        for state in THRESHOLDS:
            self.threshold_vars[state] = tk.DoubleVar(value=THRESHOLDS[state])
        
        # Processing parameters
        self.window_sec_var = tk.DoubleVar(value=WINDOW_SEC)
        self.overlap_var = tk.DoubleVar(value=OVERLAP)
        
        # Build UI
        self._create_ui()
        
    def _create_ui(self):
        """Create the user interface"""
        # Main frame with two columns
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left column - Parameters
        param_frame = ttk.LabelFrame(main_frame, text="Parameters", padding=10)
        param_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        # File selection
        file_frame = ttk.Frame(param_frame)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(file_frame, text="EEG Data File:").pack(side=tk.LEFT)
        ttk.Entry(file_frame, textvariable=self.eeg_file, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self._browse_file).pack(side=tk.LEFT)
        
        # Processing parameters
        proc_frame = ttk.LabelFrame(param_frame, text="Processing Parameters", padding=5)
        proc_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(proc_frame, text="Window Size (seconds):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Scale(proc_frame, from_=1, to=10, variable=self.window_sec_var, length=200,
                 command=lambda _: self._update_window_label()).grid(row=0, column=1, padx=5, pady=2)
        self.window_label = ttk.Label(proc_frame, text=str(self.window_sec_var.get()))
        self.window_label.grid(row=0, column=2, padx=5, pady=2)
        
        ttk.Label(proc_frame, text="Window Overlap:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Scale(proc_frame, from_=0, to=0.9, variable=self.overlap_var, length=200,
                 command=lambda _: self._update_overlap_label()).grid(row=1, column=1, padx=5, pady=2)
        self.overlap_label = ttk.Label(proc_frame, text=f"{int(self.overlap_var.get()*100)}%")
        self.overlap_label.grid(row=1, column=2, padx=5, pady=2)
        
        # MI Weight sliders
        weights_frame = ttk.LabelFrame(param_frame, text="MI Weights", padding=5)
        weights_frame.pack(fill=tk.X, padx=5, pady=10)
        
        for i, (feature, weight) in enumerate(self.current_weights.items()):
            ttk.Label(weights_frame, text=f"{feature}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Scale(weights_frame, from_=-1.0, to=1.0, variable=self.weight_vars[feature], length=200,
                     command=lambda _, f=feature: self._update_weight_label(f)).grid(row=i, column=1, padx=5, pady=2)
            lbl = ttk.Label(weights_frame, text=str(weight))
            lbl.grid(row=i, column=2, padx=5, pady=2)
            self.weight_vars[feature].label = lbl
        
        # Threshold sliders
        threshold_frame = ttk.LabelFrame(param_frame, text="State Thresholds", padding=5)
        threshold_frame.pack(fill=tk.X, padx=5, pady=10)
        
        for i, (state, threshold) in enumerate(self.current_thresholds.items()):
            ttk.Label(threshold_frame, text=f"{state}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
            ttk.Scale(threshold_frame, from_=0.0, to=1.0, variable=self.threshold_vars[state], length=200,
                     command=lambda _, s=state: self._update_threshold_label(s)).grid(row=i, column=1, padx=5, pady=2)
            lbl = ttk.Label(threshold_frame, text=str(threshold))
            lbl.grid(row=i, column=2, padx=5, pady=2)
            self.threshold_vars[state].label = lbl
        
        # Action buttons
        button_frame = ttk.Frame(param_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Process File", command=self._process_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Apply Changes", command=self._apply_changes).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reset to Defaults", command=self._reset_defaults).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Configuration", command=self._save_config).pack(side=tk.LEFT, padx=5)
        
        # Right column - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup matplotlib figure for visualization
        self.fig = plt.Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Results summary
        self.results_text = tk.Text(viz_frame, height=10, width=80)
        self.results_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Select an EEG file to begin.")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)
        
    def _update_window_label(self):
        """Update the window size label"""
        value = self.window_sec_var.get()
        self.window_label.config(text=f"{value:.1f} s")
        
    def _update_overlap_label(self):
        """Update the overlap label"""
        value = self.overlap_var.get()
        self.overlap_label.config(text=f"{int(value*100)}%")
        
    def _update_weight_label(self, feature):
        """Update the weight label for a specific feature"""
        value = self.weight_vars[feature].get()
        self.weight_vars[feature].label.config(text=f"{value:.2f}")
        
    def _update_threshold_label(self, state):
        """Update the threshold label for a specific state"""
        value = self.threshold_vars[state].get()
        self.threshold_vars[state].label.config(text=f"{value:.2f}")
        
    def _browse_file(self):
        """Open file browser to select EEG file"""
        filename = filedialog.askopenfilename(
            title="Select EEG Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir="data/toClasify"
        )
        if filename:
            self.eeg_file.set(filename)
            self.status_var.set(f"File selected: {os.path.basename(filename)}")
        
    def _process_file(self):
        """Process the selected EEG file"""
        if not self.eeg_file.get():
            messagebox.showerror("Error", "Please select an EEG file first.")
            return
        
        if not os.path.exists(self.eeg_file.get()):
            messagebox.showerror("Error", "Selected file does not exist.")
            return
        
        self.status_var.set("Processing file... This may take a moment.")
        self.root.update()
        
        # Process in a separate thread to avoid UI freezing
        threading.Thread(target=self._process_file_thread).start()
        
    def _process_file_thread(self):
        """Background thread for file processing"""
        try:
            # Process the file
            self.results = process_eeg_file(self.eeg_file.get())
            self.features = [r['features'] for r in self.results]
            
            # Update UI
            self.root.after(0, self._update_ui_after_processing)
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error: {str(e)}"))
            
    def _update_ui_after_processing(self):
        """Update UI after file processing completes"""
        if not self.results:
            self.status_var.set("Error: No results generated.")
            return
        
        self.status_var.set(f"Processed {len(self.results)} windows. Ready for analysis.")
        self._apply_changes()
        
    def _apply_changes(self):
        """Apply parameter changes and update visualization"""
        if not self.results:
            messagebox.showinfo("Info", "Process a file first.")
            return
        
        # Get current parameter values
        new_weights = {feature: var.get() for feature, var in self.weight_vars.items()}
        new_thresholds = {state: var.get() for state, var in self.threshold_vars.items()}
        
        # Recalculate MI with new weights
        for i, result in enumerate(self.results):
            mi = calculate_mi(result['features'], result['eda_value'], weights=new_weights)
            state = classify_behavioral_state(mi, thresholds=new_thresholds)
            
            # Update result
            result['mi_score'] = mi
            result['behavioral_state'] = state
        
        # Update visualization
        self._update_visualization()
        
    def _update_visualization(self):
        """Update the visualization with current results"""
        if not self.results:
            return
        
        # Clear figure
        self.fig.clear()
        
        # Create subplots
        ax1 = self.fig.add_subplot(211)  # MI time series
        ax2 = self.fig.add_subplot(212)  # State distribution
        
        # Get data
        timestamps = [r['timestamp'] for r in self.results]
        mi_values = [r['mi_score'] for r in self.results]
        states = [r['behavioral_state'] for r in self.results]
        
        # Create state color map
        state_colors = {
            'Focused': 'green',
            'Neutral': 'blue',
            'Unfocused': 'red'
        }
        colors = [state_colors.get(state, 'gray') for state in states]
        
        # Plot MI time series
        ax1.scatter(timestamps, mi_values, c=colors, alpha=0.7)
        ax1.plot(timestamps, mi_values, 'k-', alpha=0.3)
        
        # Add threshold lines
        for state, threshold in self.current_thresholds.items():
            color = state_colors.get(state.lower().capitalize(), 'gray')
            ax1.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=f"{state} Threshold")
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Mindfulness Index (MI)')
        ax1.set_title('Mindfulness Index Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot state distribution
        state_counts = {}
        for state in set(states):
            state_counts[state] = states.count(state)
        
        state_labels = list(state_counts.keys())
        state_values = list(state_counts.values())
        state_colors_list = [state_colors.get(state, 'gray') for state in state_labels]
        
        ax2.bar(state_labels, state_values, color=state_colors_list)
        ax2.set_xlabel('Behavioral State')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Behavioral States')
        
        # Update canvas
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update summary text
        total = len(self.results)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Summary of {total} windows:\n\n")
        
        self.results_text.insert(tk.END, "Behavioral States:\n")
        for state in sorted(state_counts.keys()):
            count = state_counts[state]
            pct = (count / total) * 100
            self.results_text.insert(tk.END, f"- {state}: {count} windows ({pct:.1f}%)\n")
        
        self.results_text.insert(tk.END, f"\nMI Range: {min(mi_values):.4f} to {max(mi_values):.4f}\n")
        self.results_text.insert(tk.END, f"MI Average: {sum(mi_values)/len(mi_values):.4f}\n")
        
    def _reset_defaults(self):
        """Reset parameters to default values"""
        # Reset weights
        for feature, weight in MI_WEIGHTS.items():
            self.weight_vars[feature].set(weight)
            self._update_weight_label(feature)
        
        # Reset thresholds
        for state, threshold in THRESHOLDS.items():
            self.threshold_vars[state].set(threshold)
            self._update_threshold_label(state)
        
        # Reset processing parameters
        self.window_sec_var.set(WINDOW_SEC)
        self._update_window_label()
        self.overlap_var.set(OVERLAP)
        self._update_overlap_label()
        
        self.status_var.set("Parameters reset to defaults.")
        
        # Update visualization if results exist
        if self.results:
            self._apply_changes()
        
    def _save_config(self):
        """Save current configuration to JSON file"""
        config = {
            'window_sec': self.window_sec_var.get(),
            'overlap': self.overlap_var.get(),
            'mi_weights': {feature: var.get() for feature, var in self.weight_vars.items()},
            'thresholds': {state: var.get() for state, var in self.threshold_vars.items()}
        }
        
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            defaultextension=".json",
            initialfile="mindfulness_config.json"
        )
        
        if filename:
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            self.status_var.set(f"Configuration saved to {os.path.basename(filename)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MindfulnessParameterAdjuster(root)
    root.mainloop()
