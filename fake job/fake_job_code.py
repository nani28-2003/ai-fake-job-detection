import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, messagebox

def safe_insert_text(widget, msg):
    #Safely insert text into widget
    widget.insert(tk.END, msg)
    widget.see(tk.END)

def is_dataframe_loaded(df):
    #Check if dataframe is loaded and valid
    return df is not None and isinstance(df, pd.DataFrame) and not df.empty

class FakeJobPostingDetectorApp:
    def __init__(self, master):
        self.master = master
        master.title("Fake Job Posting Detection Intelligence System")
        master.geometry("1100x720")
        
        btn_cfg = {'width': 22, 'bg': 'lightgreen', 'activebackground': '#bfeecf'}
        
        # Title
        self.title_label = tk.Label(master, text="Fake Job Posting Detection\nAnalytics Dashboard", 
                                   font=('Helvetica', 16, 'bold'))
        self.title_label.pack(pady=(10, 6))
        
        # Main frames
        self.top_frame = tk.Frame(master)
        self.top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)
        
        self.left_frame = tk.Frame(self.top_frame)
        self.left_frame.pack(side=tk.LEFT, anchor='n', padx=(0, 10))
        
        self.right_frame = tk.Frame(self.top_frame)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Buttons
        self.load_train_btn = tk.Button(self.left_frame, text="Load Train CSV", 
                                       command=self.load_train, **btn_cfg)
        self.load_train_btn.grid(row=0, column=0, padx=5, pady=4, sticky='w')
        
        self.load_test_btn = tk.Button(self.left_frame, text="Load Test CSV", 
                                      command=self.load_test, **btn_cfg)
        self.load_test_btn.grid(row=1, column=0, padx=5, pady=4, sticky='w')
        
        self.preprocess_btn = tk.Button(self.left_frame, text="Preprocess Data", 
                                       command=self.preprocess, **btn_cfg)
        self.preprocess_btn.grid(row=2, column=0, padx=5, pady=4, sticky='w')
        
        self.train_btn = tk.Button(self.left_frame, text="Train Model", 
                                  command=self.train_model, **btn_cfg)
        self.train_btn.grid(row=3, column=0, padx=5, pady=4, sticky='w')
        
        self.eda1_btn = tk.Button(self.left_frame, text="Bar Plot", 
                                 command=self.show_eda_fraud, **btn_cfg)
        self.eda1_btn.grid(row=4, column=0, padx=5, pady=4, sticky='w')
        
        self.eda2_btn = tk.Button(self.left_frame, text="Box Plot", 
                                 command=self.show_eda_textlen, **btn_cfg)
        self.eda2_btn.grid(row=5, column=0, padx=5, pady=4, sticky='w')
        
        self.eda3_btn = tk.Button(self.left_frame, text="Correlation Heatmap", 
                                 command=self.show_eda_heatmap, **btn_cfg)
        self.eda3_btn.grid(row=6, column=0, padx=5, pady=4, sticky='w')
        
        # Prediction widgets
        self.predict_label = tk.Label(self.left_frame, text="Predict Row ID:")
        self.predict_label.grid(row=7, column=0, padx=5, pady=(10, 0), sticky='w')
        
        self.row_entry = tk.Entry(self.left_frame, width=20)
        self.row_entry.grid(row=8, column=0, padx=5, pady=(0, 6), sticky='w')
        self.row_entry.insert(0, "0")
        
        self.predict_btn = tk.Button(self.left_frame, text="Predict Fraud", 
                                    command=self.predict_row, **btn_cfg)
        self.predict_btn.grid(row=9, column=0, padx=5, pady=4, sticky='w')
        
        self.clear_output_btn = tk.Button(self.left_frame, text="Clear Output", 
                                         command=self.clear_output, **btn_cfg)
        self.clear_output_btn.grid(row=10, column=0, padx=5, pady=12, sticky='w')
        
        # Output box
        self.output_box = tk.Text(self.right_frame, height=35, width=70, 
                                 bg='white', fg='black', font=('Courier', 10))
        self.output_box.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = tk.Label(master, textvariable=self.status_var, 
                                  bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Data model holders
        self.train_data = None
        self.test_data = None
        self.model = None
        self.label_encoders = {}
        self.original_data = {}  # Store original values for display
    
    def update_status(self, msg):
        self.status_var.set(msg)
    
    def load_train(self):
        try:
            filepath = filedialog.askopenfilename(title="Select Train CSV File", 
                                                 filetypes=[("CSV files", "*.csv")])
            if not filepath:
                return
            self.train_data = pd.read_csv(filepath)
            self.original_data['train'] = self.train_data.copy()  # Store original
            safe_insert_text(self.output_box, f"Training data loaded. Rows: {self.train_data.shape[0]}, "
                                            f"Columns: {self.train_data.shape[1]}\n")
            self.update_status("Training data loaded")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load training file: {e}")
    
    def load_test(self):
        try:
            filepath = filedialog.askopenfilename(title="Select Test CSV File", 
                                                 filetypes=[("CSV files", "*.csv")])
            if not filepath:
                return
            self.test_data = pd.read_csv(filepath)
            self.original_data['test'] = self.test_data.copy()  # Store original
            safe_insert_text(self.output_box, f"Testing data loaded. Rows: {self.test_data.shape[0]}, "
                                           f"Columns: {self.test_data.shape[1]}\n")
            self.update_status("Testing data loaded")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load testing file: {e}")
    
    def preprocess(self):
        try:
            if not is_dataframe_loaded(self.train_data) or not is_dataframe_loaded(self.test_data):
                messagebox.showerror("Error", "Please load both train and test CSV files before preprocessing.")
                return
            
            # Store original data before preprocessing
            self.original_data['train_pre'] = self.train_data.copy()
            self.original_data['test_pre'] = self.test_data.copy()
            
            # Clean numeric columns
            for df in [self.train_data, self.test_data]:
                numeric_cols = ['salary_range']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        df[col] = df[col].fillna(df[col].median())
                
                # Create text length features from original text
                text_cols = ['title', 'description', 'requirements', 'benefits', 'company_profile']
                for col in text_cols:
                    if col in df.columns:
                        df[f'{col}_len'] = df[col].fillna('').astype(str).str.len()
            
            target_col = 'fraudulent'
            cat_cols = ['employment_type', 'required_experience', 'required_education', 
                       'industry', 'function', 'telecommuting', 'has_company_logo', 'has_questions']
            
            # Encode categorical columns but keep mapping
            for col in cat_cols:
                if col in self.train_data.columns:
                    le = LabelEncoder()
                    original_values = self.train_data[col].astype(str).values
                    self.train_data[col] = le.fit_transform(original_values)
                    self.label_encoders[col] = le
                    
                    if col in self.test_data.columns:
                        try:
                            self.test_data[col] = le.transform(self.test_data[col].astype(str))
                        except:
                            mapping = {v: i for i, v in enumerate(le.classes_)}
                            self.test_data[col] = self.test_data[col].astype(str).map(mapping).fillna(-1)
            
            # Encode target (0=Real, 1=Fake)
            if target_col in self.train_data.columns:
                le_target = LabelEncoder()
                original_target = self.train_data[target_col].astype(str).values
                self.train_data[target_col] = le_target.fit_transform(original_target)
                self.label_encoders[target_col] = le_target
                
                if target_col in self.test_data.columns:
                    try:
                        self.test_data[target_col] = le_target.transform(self.test_data[target_col].astype(str))
                    except:
                        mapping = {v: i for i, v in enumerate(le_target.classes_)}
                        self.test_data[target_col] = self.test_data[target_col].astype(str).map(mapping).fillna(-1)
            
            safe_insert_text(self.output_box, "Preprocessing complete. Label encoding & numeric cleaning applied.\n")
            self.update_status("Preprocessing finished")
        except Exception as e:
            messagebox.showerror("Preprocess Error", str(e))
    
    def get_original_feature_value(self, col, encoded_val, row_id=0, data_type='test'):
        #Get actual content from original columns
        if col.endswith('_len'):
            # Get actual text content instead of length
            text_col = col.replace('_len', '')
            if text_col in self.original_data[data_type].columns:
                content = str(self.original_data[data_type][text_col].iloc[row_id])
                if pd.isna(self.original_data[data_type][text_col].iloc[row_id]):
                    return "No content"
                return content[:100] + "..." if len(content) > 100 else content
        elif col in self.label_encoders:
            try:
                return self.label_encoders[col].inverse_transform([int(encoded_val)])[0]
            except:
                if col == 'telecommuting':
                    return "Yes" if int(encoded_val) == 1 else "No"
                elif col == 'has_company_logo':
                    return "Yes" if int(encoded_val) == 1 else "No"
                elif col == 'has_questions':
                    return "Yes" if int(encoded_val) == 1 else "No"
                return str(encoded_val)
        elif col == 'salary_range':
            return f"${int(encoded_val):,}" if pd.notna(encoded_val) else "No salary data"
        return str(encoded_val)
    
    def train_model(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Training data not loaded.")
                return
            
            if 'fraudulent' not in self.train_data.columns:
                messagebox.showerror("Error", "Target column 'fraudulent' not found.")
                return
            
            feature_cols = ['title_len', 'description_len', 'requirements_len', 'benefits_len', 
                           'company_profile_len', 'telecommuting', 'has_company_logo', 'has_questions',
                           'employment_type', 'required_experience', 'required_education', 'industry', 'function']
            
            available_features = [c for c in feature_cols if c in self.train_data.columns]
            X_train = self.train_data[available_features]
            y_train = self.train_data['fraudulent']
            
            clf = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
            clf.fit(X_train, y_train)
            self.model = clf
            
            preds = clf.predict(X_train)
            acc = accuracy_score(y_train, preds)
            cm = confusion_matrix(y_train, preds)
            
            safe_insert_text(self.output_box, "Model training complete using Random Forest.\n")
            safe_insert_text(self.output_box, f"Training Accuracy: {acc:.4f}\n")
            safe_insert_text(self.output_box, "Confusion Matrix (0=Real, 1=Fake):\n")
            safe_insert_text(self.output_box, str(cm) + "\n")
            self.update_status("Model trained")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
    
    def predict_row(self):
        try:
            if self.model is None:
                messagebox.showerror("Error", "No model available. Train the model first.")
                return
            if not is_dataframe_loaded(self.test_data):
                messagebox.showerror("Error", "Test data not loaded.")
                return
            
            row_id = int(self.row_entry.get())
            if row_id < 0 or row_id >= self.test_data.shape[0]:
                messagebox.showerror("Error", f"Row ID out of range. Enter 0 to {self.test_data.shape[0] - 1}")
                return
            
            feature_cols = ['title_len', 'description_len', 'requirements_len', 'benefits_len', 
                           'company_profile_len', 'telecommuting', 'has_company_logo', 'has_questions',
                           'employment_type', 'required_experience', 'required_education', 'industry', 'function']
            
            available_features = [c for c in feature_cols if c in self.test_data.columns]
            input_row = self.test_data[available_features].iloc[row_id:row_id+1]
            
            pred = self.model.predict(input_row)[0]
            pred_proba = self.model.predict_proba(input_row)[0]
            
            le_target = self.label_encoders.get('fraudulent', None)
            fraud_status = "FAKE" if pred == 1 else "REAL"
            
            safe_insert_text(self.output_box, f"\n{'='*60}\n")
            safe_insert_text(self.output_box, f"PREDICTION FOR ROW ID {row_id}:\n")
            safe_insert_text(self.output_box, f"{'='*60}\n")
            
            # Show ACTUAL content from original columns
            for col in available_features:
                encoded_val = self.test_data[col].iloc[row_id]
                original_content = self.get_original_feature_value(col, encoded_val, row_id, 'test')
                safe_insert_text(self.output_box, f"{col:25}: {original_content}\n")
            
            safe_insert_text(self.output_box, f"{'='*60}\n")
            safe_insert_text(self.output_box, f"PREDICTED STATUS: {fraud_status}\n")
            safe_insert_text(self.output_box, f"Real Probability: {pred_proba[0]:.4f}\n")
            safe_insert_text(self.output_box, f"Fake Probability: {pred_proba[1]:.4f}\n")
            safe_insert_text(self.output_box, f"{'='*60}\n\n")
            
            self.update_status(f"Prediction done - Row {row_id}: {fraud_status}")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
    
    def show_eda_fraud(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Load training data first.")
                return
            
            win = tk.Toplevel(self.master)
            win.title("EDA - Fraudulent Job Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data = self.train_data.copy()
            data['fraud_label'] = ['REAL' if x == 0 else 'FAKE' for x in data['fraudulent']]
            
            sns.countplot(data=data, x='fraud_label', palette='viridis', ax=ax)
            ax.set_title('FAKE vs REAL Job Postings Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Job Status', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            plt.xticks(rotation=0)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("EDA Error", str(e))
    
    def show_eda_textlen(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Load training data first.")
                return
            
            win = tk.Toplevel(self.master)
            win.title("EDA - Job Posting Text Length Distribution")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            text_len_cols = [col for col in self.train_data.columns if col.endswith('_len')]
            if text_len_cols:
                data_subset = self.train_data[text_len_cols].melt(var_name='Text_Feature', value_name='Length')
                sns.boxplot(data=data_subset, x='Text_Feature', y='Length', ax=ax)
                ax.set_title('Text Length Distribution by Feature', fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("EDA Error", str(e))
    
    def show_eda_heatmap(self):
        try:
            if not is_dataframe_loaded(self.train_data):
                messagebox.showerror("Error", "Load training data first.")
                return
            
            numeric_cols = self.train_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                messagebox.showwarning("Warning", "Need at least 2 numeric columns for heatmap.")
                return
            
            corr = self.train_data[numeric_cols].corr()
            win = tk.Toplevel(self.master)
            win.title("EDA - Feature Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(12, 9))
            
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, 
                       square=True, cbar_kws={'shrink': 0.8}, ax=ax)
            ax.set_title('Feature Correlation Heatmap (Fake Job Detection)', fontsize=16, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', rotation=0, labelsize=9)
            plt.tight_layout()
            
            canvas = FigureCanvasTkAgg(fig, master=win)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("EDA Error", str(e))
    
    def clear_output(self):
        self.output_box.delete(1.0, tk.END)
        self.update_status("Output cleared")

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeJobPostingDetectorApp(root)
    root.mainloop()
