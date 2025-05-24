#Developed by ODAT project
#please see https://odat.info
#please see https://github.com/ODAT-Project
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
#enable experimental features FIRST
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor
import csv

class CSVImputationApp:
    def __init__(self, master):
        self.master = master
        master.title("CSV Data Imputation Tool")
        master.geometry("1000x700") #adjusted default size slightly

        self.df = None
        self.original_df = None
        self.file_path = None
        self.missing_values_recognized = ['NA', '?', 'null', 'NaN', '', np.nan] #common missing value strings -- expand for more weird cases

        #about
        self.app_version = "1.0.0"
        self.app_author = "ODAT project"
        self.app_description = "A tool to load CSV files, identify missing values, and apply various imputation techniques."


        #applying some cool styles for the gui -- interactive
        self.style = ttk.Style()
        self.style.theme_use('clam') # Using 'clam' for a slightly more modern look
        self.style.configure("TButton", padding=6, relief="flat", font=('Helvetica', 10))
        self.style.map("TButton",
                       foreground=[('!active', 'black'), ('active', 'black')],
                       background=[('!active', '#007bff'), ('active', '#0056b3')])
        self.style.configure("Accent.TButton", foreground="white", background="#28a745") # Example for a different button style
        self.style.map("Accent.TButton", background=[('active', '#218838')])

        self.style.configure("TLabel", padding=5, font=('Helvetica', 10))
        self.style.configure("TEntry", padding=5, font=('Helvetica', 10))
        self.style.configure("TCombobox", padding=5, font=('Helvetica', 10))
        self.style.configure("Treeview.Heading", font=('Helvetica', 10, 'bold'), background="#e9ecef", foreground="#343a40")
        self.style.configure("Treeview", font=('Helvetica', 9), rowheight=25) # Font for treeview items
        self.style.map("Treeview.Heading", relief=[('active','groove'),('pressed','sunken')])


        menubar = tk.Menu(master)
        
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open CSV", command=self.load_csv, accelerator="Ctrl+O")
        filemenu.add_command(label="Save Imputed CSV", command=self.save_csv, accelerator="Ctrl+S", state=tk.DISABLED)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=master.quit, accelerator="Ctrl+Q")
        menubar.add_cascade(label="File", menu=filemenu)
        self.filemenu = filemenu

        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about_dialog)
        menubar.add_cascade(label="Help", menu=helpmenu)

        master.config(menu=menubar)

        #keyboard shortcuts for menu items
        master.bind_all("<Control-o>", lambda event: self.load_csv())
        master.bind_all("<Control-s>", lambda event: self.save_csv() if self.filemenu.entrycget("Save Imputed CSV", "state") == tk.NORMAL else None)
        master.bind_all("<Control-q>", lambda event: master.quit())


        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Label(controls_frame, text="File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.file_path_label = ttk.Label(controls_frame, text="No file loaded.", width=50, anchor="w", relief="sunken", padding=3)
        self.file_path_label.grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")
        
        self.load_button = ttk.Button(controls_frame, text="Load CSV", command=self.load_csv)
        self.load_button.grid(row=0, column=4, padx=5, pady=5)
        
        self.reset_button = ttk.Button(controls_frame, text="Reset Data", command=self.reset_data, state=tk.DISABLED)
        self.reset_button.grid(row=0, column=5, padx=5, pady=5)


        ttk.Label(controls_frame, text="Select Column:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.column_var = tk.StringVar()
        self.column_dropdown = ttk.Combobox(controls_frame, textvariable=self.column_var, state="readonly", width=25)
        self.column_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        #here we do imputation method selection
        ttk.Label(controls_frame, text="Imputation Method:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
        self.method_var = tk.StringVar()
        self.imputation_methods = [
            "Mean", "Median", "Mode", "Constant Value",
            "Forward Fill (ffill)", "Backward Fill (bfill)",
            "Linear Interpolation", "Random Sample",
            "KNN Imputer", "Iterative Imputer (MICE)",
            "Iterative Imputer (HistGradientBoostingRegressor)"
        ]
        self.method_dropdown = ttk.Combobox(controls_frame, textvariable=self.method_var, values=self.imputation_methods, state="readonly", width=30)
        self.method_dropdown.grid(row=1, column=3, padx=5, pady=5, sticky="ew")
        self.method_dropdown.bind("<<ComboboxSelected>>", self.toggle_constant_input)

        self.constant_value_label = ttk.Label(controls_frame, text="Constant Value:")
        self.constant_value_entry = ttk.Entry(controls_frame, width=15)

        self.apply_button = ttk.Button(controls_frame, text="Apply Imputation", command=self.apply_imputation, state=tk.DISABLED, style="Accent.TButton")
        self.apply_button.grid(row=1, column=4, padx=5, pady=5, columnspan=2, sticky="ew")
        
        controls_frame.grid_columnconfigure(1, weight=1) #allow column dropdown to expand
        controls_frame.grid_columnconfigure(3, weight=1) #allow method dropdown to expand


        data_frame = ttk.LabelFrame(main_frame, text="Data Preview", padding="10")
        data_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.tree = ttk.Treeview(data_frame, show="headings")
        vsb = ttk.Scrollbar(data_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(data_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        data_frame.grid_rowconfigure(0, weight=1)
        data_frame.grid_columnconfigure(0, weight=1)
        
        self.tree.tag_configure('missing', background='#FFFACD', foreground='#A52A2A') #LemonChiffon, Brown
        self.tree.tag_configure('imputed', background='#90EE90', foreground='black')   #LightGreen


        #status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Load a CSV file to begin.")
        status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w", padding=5)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_about_dialog(self):
        about_message = (
            f"CSV Data Imputation Tool\n\n"
            f"Version: {self.app_version}\n"
            f"Author: {self.app_author}\n\n"
            f"{self.app_description}\n\n"
        )
        messagebox.showinfo("About CSV Imputation Tool", about_message)


    def toggle_constant_input(self, event=None):
        if self.method_var.get() == "Constant Value":
            self.constant_value_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
            self.constant_value_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
            self.constant_value_entry.focus() #set focus to the entry field
        else:
            self.constant_value_label.grid_forget()
            self.constant_value_entry.grid_forget()

    def load_csv(self):
        path = filedialog.askopenfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Select CSV file"
        )
        if not path: #user cancelled dialog
            return

        try:
            #read CSV, explicitly recognizing common missing value placeholders
            self.df = pd.read_csv(path, na_values=self.missing_values_recognized, keep_default_na=True)
            self.original_df = self.df.copy() #keep a copy of the original for reset
            self.file_path = path
            self.file_path_label.config(text=self.file_path.split('/')[-1]) #display only filename
            
            self.display_dataframe()
            self.update_column_dropdown()
            
            self.filemenu.entryconfig("Save Imputed CSV", state=tk.NORMAL)
            self.apply_button.config(state=tk.NORMAL)
            self.reset_button.config(state=tk.NORMAL)
            self.status_var.set(f"Loaded: {self.file_path.split('/')[-1]} ({self.df.shape[0]} rows, {self.df.shape[1]} columns)")
        except FileNotFoundError:
            messagebox.showerror("Error", f"File not found: {path}")
            self.status_var.set("Error: File not found.")
        except pd.errors.EmptyDataError:
            messagebox.showerror("Error", f"The CSV file is empty: {path}")
            self.status_var.set("Error: CSV file is empty.")
        except Exception as e:
            messagebox.showerror("Error Loading CSV", f"Failed to load or parse CSV file: {e}")
            self.status_var.set("Error loading CSV.")
            #reset UI elements if loading fails
            self.df = None
            self.original_df = None
            self.file_path = None
            self.file_path_label.config(text="No file loaded.")
            self.clear_treeview()
            self.column_dropdown['values'] = []
            self.column_var.set('')
            self.filemenu.entryconfig("Save Imputed CSV", state=tk.DISABLED)
            self.apply_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)


    def reset_data(self):
        if self.original_df is not None:
            self.df = self.original_df.copy()
            self.display_dataframe()
            self.status_var.set("Data reset to its original loaded state.")
            messagebox.showinfo("Data Reset", "The data has been reset to its original state.")
        else:
            messagebox.showwarning("No Data", "No original data to reset. Please load a CSV file first.")


    def display_dataframe(self, imputed_col_name=None, original_missing_indices=None):
        self.clear_treeview()
        if self.df is None:
            return

        self.tree["columns"] = list(self.df.columns)
        for col in self.df.columns:
            self.tree.heading(col, text=col, anchor="w")
            header_width = len(col) * 8 
            #sample a few rows
            sample_data_width = 0
            if not self.df.empty:
                sample_data_width = self.df[col].astype(str).str.len().max() * 7 if len(self.df[col]) > 0 else 0
                sample_data_width = min(sample_data_width, 200) #cap max width from sample
            
            col_width = max(header_width, sample_data_width, 80) #min width 80
            self.tree.column(col, anchor="w", width=col_width, stretch=tk.NO)


        for index, row_series in self.df.iterrows():
            display_values = [str(v) if pd.isna(v) else str(v) for v in row_series.tolist()]
            
            row_tags = [] #row tag
            
            for i, col_name_iter in enumerate(self.df.columns):
                original_value_is_na = pd.isna(self.original_df.iloc[index, i]) if self.original_df is not None and index < len(self.original_df) else pd.isna(row_series.iloc[i])
                current_value_is_na = pd.isna(row_series.iloc[i])

                #if the cell in the currently displayed DataFrame (self.df) is NaN
                if current_value_is_na:
                    row_tags.append('missing') # General tag for missing in current view
                #if a specific column was just imputed
                elif imputed_col_name == col_name_iter and index in (original_missing_indices or []):
                    if not current_value_is_na:
                         row_tags.append('imputed')


            final_tags_for_row = []
            if imputed_col_name and index in (original_missing_indices or []):
                 col_idx_imputed = self.df.columns.get_loc(imputed_col_name)
                 if not pd.isna(self.df.iloc[index, col_idx_imputed]):
                     final_tags_for_row.append('imputed')

            if not final_tags_for_row and any(pd.isna(row_series.iloc[j]) for j in range(len(row_series))):
                final_tags_for_row.append('missing')
                
            self.tree.insert("", "end", values=display_values, tags=tuple(final_tags_for_row))


    def clear_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["columns"] = [] 


    def update_column_dropdown(self):
        if self.df is not None:
            column_list = list(self.df.columns)
            self.column_dropdown['values'] = column_list
            if column_list: 
                self.column_var.set(column_list[0]) 
            else:
                self.column_var.set('') 
        else:
            self.column_dropdown['values'] = []
            self.column_var.set('')

    def apply_imputation(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Please load a CSV file first.")
            return

        col_name = self.column_var.get()
        method = self.method_var.get()

        if not col_name:
            messagebox.showwarning("Selection Missing", "Please select a column to impute.")
            return
        if not method:
            messagebox.showwarning("Selection Missing", "Please select an imputation method.")
            return

        try:
            original_missing_indices_for_col = self.df[self.df[col_name].isna()].index.tolist()
            
            target_column_series = self.df[col_name].copy() 
            is_numeric_col = pd.api.types.is_numeric_dtype(target_column_series)

            #imputation here
            if method == "Mean":
                if not is_numeric_col:
                    raise ValueError("Mean imputation is only applicable to numerical columns.")
                fill_value = target_column_series.mean()
                if pd.isna(fill_value): raise ValueError(f"Cannot compute mean for column '{col_name}'. It might be all NaN.")
                target_column_series.fillna(fill_value, inplace=True)
            elif method == "Median":
                if not is_numeric_col:
                    raise ValueError("Median imputation is only applicable to numerical columns.")
                fill_value = target_column_series.median()
                if pd.isna(fill_value): raise ValueError(f"Cannot compute median for column '{col_name}'. It might be all NaN.")
                target_column_series.fillna(fill_value, inplace=True)
            elif method == "Mode":
                modes = target_column_series.mode()
                if modes.empty: 
                    raise ValueError(f"Could not determine a mode for column '{col_name}'. It might be all NaN or have no repeating values.")
                fill_value = modes[0] #use the first mode if multiple exist
                target_column_series.fillna(fill_value, inplace=True)
            elif method == "Constant Value":
                const_val_str = self.constant_value_entry.get()
                if not const_val_str: #check if entry is empty
                    raise ValueError("Please enter a constant value for imputation.")
                try:
                    if is_numeric_col: #attempt to convert to numeric if original column is numeric
                        const_val = pd.to_numeric(const_val_str)
                    else: #otherwise, use as string
                        const_val = const_val_str
                except ValueError: #if conversion fails for a numeric column
                     messagebox.showwarning("Type Conversion Warning", f"Could not convert '{const_val_str}' to a number for column '{col_name}'. Using it as a string.")
                     const_val = const_val_str 
                target_column_series.fillna(const_val, inplace=True)

            elif method == "Forward Fill (ffill)":
                target_column_series.ffill(inplace=True)
            elif method == "Backward Fill (bfill)":
                target_column_series.bfill(inplace=True)
            elif method == "Linear Interpolation":
                if not is_numeric_col:
                     raise ValueError("Linear interpolation is only applicable to numerical columns.")
                if target_column_series.isna().all():
                    raise ValueError(f"Column '{col_name}' contains only missing values. Cannot interpolate.")
                if target_column_series.count() < 2 :
                    raise ValueError(f"Column '{col_name}' has fewer than two non-missing values. Cannot interpolate.")
                target_column_series = target_column_series.interpolate(method='linear', limit_direction='both') 
            
            elif method == "Random Sample":
                non_missing_values = target_column_series.dropna()
                if non_missing_values.empty:
                    raise ValueError(f"Column '{col_name}' has no non-missing values to sample from.")
                
                missing_indices_in_series = target_column_series[target_column_series.isna()].index
                if not missing_indices_in_series.empty:
                    random_samples = np.random.choice(non_missing_values.values, size=len(missing_indices_in_series), replace=True)
                    target_column_series.loc[missing_indices_in_series] = random_samples

            elif method == "KNN Imputer":
                df_numeric = self.df.select_dtypes(include=np.number)
                if df_numeric.empty:
                    raise ValueError("KNN Imputer requires at least one numerical column in the dataset.")
                if col_name not in df_numeric.columns:
                    raise ValueError(f"Column '{col_name}' is not numeric. KNN Imputer operates on numeric columns.")

                imputer_knn = KNNImputer(n_neighbors=5) 
                imputed_numeric_array = imputer_knn.fit_transform(df_numeric)
                df_imputed_numeric = pd.DataFrame(imputed_numeric_array, columns=df_numeric.columns, index=df_numeric.index)
                target_column_series = df_imputed_numeric[col_name]

            elif method.startswith("Iterative Imputer"):
                df_numeric_iter = self.df.select_dtypes(include=np.number)
                if df_numeric_iter.empty:
                    raise ValueError("Iterative Imputer requires at least one numerical column.")
                if col_name not in df_numeric_iter.columns:
                     raise ValueError(f"Column '{col_name}' is not numeric. IterativeImputer (with current estimators) primarily works on numeric columns.")

                estimator_choice = HistGradientBoostingRegressor() if "HistGradientBoostingRegressor" in method else None
                imputer_iter = IterativeImputer(estimator=estimator_choice, max_iter=10, random_state=0, verbose=0)
                
                imputed_numeric_array_iter = imputer_iter.fit_transform(df_numeric_iter)
                df_imputed_numeric_iter = pd.DataFrame(imputed_numeric_array_iter, columns=df_numeric_iter.columns, index=df_numeric_iter.index)
                target_column_series = df_imputed_numeric_iter[col_name]
            
            self.df[col_name] = target_column_series

            #for highlighting
            self.display_dataframe(imputed_col_name=col_name, original_missing_indices=original_missing_indices_for_col)
            self.status_var.set(f"Successfully imputed column '{col_name}' using {method}.")
            messagebox.showinfo("Imputation Successful", f"Column '{col_name}' has been imputed using {method}.")

        except ValueError as ve:
            messagebox.showerror("Imputation Error", f"Could not apply imputation: {ve}")
            self.status_var.set(f"Imputation error on '{col_name}'.")
        except Exception as e:
            messagebox.showerror("Imputation Error", f"An unexpected error occurred: {e}\n(Check console for more details if running from terminal)")
            print(f"DEBUG: Detailed error during imputation: {type(e).__name__}: {e}") 
            self.status_var.set(f"Unexpected error during imputation on '{col_name}'.")


    def save_csv(self):
        if self.df is None:
            messagebox.showwarning("No Data", "No data available to save. Please load and optionally impute a CSV file.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="imputed_data.csv", 
            title="Save Imputed CSV As"
        )

        if not save_path:
            return

        try:
            self.df.to_csv(save_path, index=False, na_rep='NaN')
            self.status_var.set(f"Imputed data successfully saved to: {save_path}")
            messagebox.showinfo("Save Successful", f"Data saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save CSV file: {e}")
            self.status_var.set("Error: Failed to save imputed data.")


if __name__ == '__main__':
    root = tk.Tk()
    app = CSVImputationApp(root)
    root.mainloop()
