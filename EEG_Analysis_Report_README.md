# EEG Analysis Report: Instructions for Exporting to PDF

This file is a template for your EEG analysis report. To create a PDF with all required tables, statistics, and plots:

1. **Run your Python pipeline** to generate:
   - Filtering statistics (mean, std, etc. before/after filtering)
   - Feature extraction statistics (mean, variance, std, kurtosis, skewness per channel/band)
   - LOSO cross-validation reports
   - Plots: time-domain, frequency-domain, feature distributions, emotion over time, correlation matrix

2. **Export tables and plots**:
   - Use `matplotlib.pyplot.savefig('filename.png')` for plots.
   - Use `pandas.DataFrame.to_csv('filename.csv')` or `to_latex()` for tables.

3. **Compile the report**:
   - Insert the exported tables and plots into this markdown file.
   - Use a markdown-to-pdf tool (e.g., VSCode extension, `pandoc`, or Jupyter nbconvert) to export as PDF.

4. **Include the following sections**:
   - Data and parameters summary (already included)
   - Filtering statistics table
   - Feature extraction statistics table
   - LOSO cross-validation table
   - All relevant plots/graphics
   - Explanation paragraph (already included)

---

**Tip:**
- You can automate the export of tables and plots in your Python script for reproducibility.
- For a professional look, use Jupyter Notebook for combining code, tables, plots, and text, then export as PDF.

---

**This file is ready for you to fill in with your actual results and export as PDF.**
