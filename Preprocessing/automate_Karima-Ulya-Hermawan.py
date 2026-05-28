import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

def run_preprocessing(input_path, output_path):
    print("Membaca data raw...")
    df = pd.read_csv(input_path)
    
    print("Memulai proses cleaning dan preprocessing...")
    # 1. Menghapus Kolom id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # 2. Menangani Missing Values
    df = df.dropna()

    # 3. Encoding Data Kategorikal
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df = df.astype(float)

    # 4. Scaling Fitur Numerik
    num_cols = ['Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 
                'Study Satisfaction', 'Job Satisfaction', 'Work/Study Hours', 'Financial Stress']
    
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # 5. Menyimpan Data Bersih
    print("Menyimpan dataset hasil preprocessing...")
    # Membuat folder output jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessing SELESAI! Data disimpan di: {output_path}")
    print(f"Ukuran data akhir: {df.shape}")

if __name__ == "__main__":
    # Menentukan jalur file 
    INPUT_FILE = "../Student_Depression_Dataset_raw/Student Depression Dataset.csv"
    OUTPUT_FILE = "Student_Depression_Dataset_preprocessing/dataset_clean.csv"
    
    run_preprocessing(INPUT_FILE, OUTPUT_FILE)