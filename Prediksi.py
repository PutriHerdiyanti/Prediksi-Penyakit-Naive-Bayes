{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e3001308-f0f4-4104-a1df-8d95396a6891",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "da11a677-b4fc-41c5-8a0a-f200f83960fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Load Data & Training\n",
    "\n",
    "data = pd.read_csv(\"Data Prediksi.csv\")\n",
    "\n",
    "# Ganti 'PENYAKIT' dengan nama kolom target di dataset kamu\n",
    "X = data[['BATUK', 'NYERI_OTOT', 'KELELAHAN', 'SAKIT_TENGGOROKAN', 'HIDUNG_MELER',\n",
    "          'HIDUNG_TERSUMBAT', 'DEMAM', 'MUAL', 'MUNTAH', 'DIARE', 'SESAK_NAPAS',\n",
    "          'SULIT_BERNAPAS', 'KEHILANGAN_INDERA_PERASA', 'KEHILANGAN_INDERA_PENCIUMAN',\n",
    "          'MATA_GATAL', 'HIDUNG_GATAL', 'MULUT_GATAL', 'TELINGA_BAGIAN_DALAM_GATAL',\n",
    "          'BERSIN', 'MATA_MERAH']]\n",
    "y = data['TARGET']  # ubah sesuai nama kolom target kamu\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42, stratify=y\n",
    ")\n",
    "\n",
    "model = GaussianNB()\n",
    "model.fit(X_train, y_train)\n",
    "akurasi = accuracy_score(y_test, model.predict(X_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "e605eba0-f807-4cfb-834d-ac2b4a588bc5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Streamlit UI\n",
    "st.set_page_config(page_title=\"Prediksi Penyakit - Naive Bayes\", page_icon=\"🩺\", layout=\"wide\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "0ff1cab4-5ae7-49cb-aea1-0508d20d7986",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-11-11 14:55:18.670 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\User\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# --- Custom CSS untuk tampilan profesional ---\n",
    "st.markdown(\"\"\"\n",
    "    <style>\n",
    "    body {\n",
    "        background-color: #f7f9fb;\n",
    "    }\n",
    "    .stButton>button {\n",
    "        color: white;\n",
    "        background-color: #007BFF;\n",
    "        border-radius: 5px;\n",
    "        padding: 0.6em 1.2em;\n",
    "    }\n",
    "    .stButton>button:hover {\n",
    "        background-color: #0056b3;\n",
    "    }\n",
    "    .result-box {\n",
    "        background-color: #e8f4ff;\n",
    "        border-left: 6px solid #007BFF;\n",
    "        padding: 15px;\n",
    "        border-radius: 5px;\n",
    "    }\n",
    "    </style>\n",
    "\"\"\", unsafe_allow_html=True)\n",
    "\n",
    "# --- Header ---\n",
    "st.title(\"🩺 Aplikasi Prediksi Penyakit Berdasarkan Gejala\")\n",
    "st.write(\"Gunakan aplikasi ini untuk memprediksi apakah pasien mengalami **Covid**, **Flu**, **Alergi**, atau **Cold** berdasarkan gejala yang dialami.\")\n",
    "\n",
    "# --- Sidebar ---\n",
    "st.sidebar.header(\"Tentang Aplikasi\")\n",
    "st.sidebar.info(\"\"\"\n",
    "Aplikasi ini menggunakan algoritma **Naive Bayes Classifier**  \n",
    "untuk mengidentifikasi kemungkinan penyakit berdasarkan gejala.\n",
    "\"\"\")\n",
    "st.sidebar.write(f\"📊 Akurasi model: **{akurasi*100:.2f}%**\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d264d24a-8f8b-4695-b3bc-10faa08ad222",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Input Gejala\n",
    "st.subheader(\"Pilih Gejala yang Dialami Pasien:\")\n",
    "\n",
    "gejala = [\n",
    "    'BATUK', 'NYERI_OTOT', 'KELELAHAN', 'SAKIT_TENGGOROKAN', 'HIDUNG_MELER',\n",
    "    'HIDUNG_TERSUMBAT', 'DEMAM', 'MUAL', 'MUNTAH', 'DIARE', 'SESak_NAPAS',\n",
    "    'SULIT_BERNAPAS', 'KEHILANGAN_INDERA_PERASA', 'KEHILANGAN_INDERA_PENCIUMAN',\n",
    "    'MATA_GATAL', 'HIDUNG_GATAL', 'MULUT_GATAL', 'TELINGA_BAGIAN_DALAM_GATAL',\n",
    "    'BERSIN', 'MATA_MERAH'\n",
    "]\n",
    "\n",
    "cols = st.columns(2)\n",
    "input_data = []\n",
    "for i, g in enumerate(gejala):\n",
    "    with cols[i % 2]:\n",
    "        val = st.checkbox(g)\n",
    "        input_data.append(1 if val else 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a5bc704c-958a-4f00-98a0-b2306c9ac9e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Prediksi\n",
    "col1, col2 = st.columns([1, 1])\n",
    "\n",
    "with col1:\n",
    "    prediksi_btn = st.button(\"🔍 Prediksi Sekarang\")\n",
    "with col2:\n",
    "    reset_btn = st.button(\"🔄 Reset Pilihan\")\n",
    "\n",
    "if prediksi_btn:\n",
    "    df_input = pd.DataFrame([input_data], columns=gejala)\n",
    "    hasil = model.predict(df_input)[0]\n",
    "\n",
    "    # Warna & deskripsi hasil\n",
    "    if hasil.lower() == \"covid\":\n",
    "        warna = \"#FF4B4B\"\n",
    "        deskripsi = \"Pasien menunjukkan gejala khas **COVID-19**, seperti demam, batuk, kehilangan indera penciuman atau perasa.\"\n",
    "    elif hasil.lower() == \"flu\":\n",
    "        warna = \"#FFC107\"\n",
    "        deskripsi = \"Pasien mungkin mengalami **flu biasa** — biasanya gejala berupa demam, sakit tenggorokan, dan kelelahan.\"\n",
    "    elif hasil.lower() == \"alergi\":\n",
    "        warna = \"#4CAF50\"\n",
    "        deskripsi = \"Pasien kemungkinan besar mengalami **alergi** — gejala umum meliputi bersin, mata gatal, dan hidung meler.\"\n",
    "    elif hasil.lower() == \"cold\":\n",
    "        warna = \"#2196F3\"\n",
    "        deskripsi = \"Pasien menunjukkan gejala **common cold** — gejala ringan seperti batuk, hidung tersumbat, dan sakit tenggorokan.\"\n",
    "    else:\n",
    "        warna = \"#6c757d\"\n",
    "        deskripsi = \"Hasil tidak dapat ditentukan secara pasti.\"\n",
    "\n",
    "    st.markdown(f\"\"\"\n",
    "        <div class=\"result-box\" style=\"border-left-color:{warna}\">\n",
    "            <h4 style=\"color:{warna};\">🧠 Hasil Prediksi: {hasil}</h4>\n",
    "            <p>{deskripsi}</p>\n",
    "        </div>\n",
    "    \"\"\", unsafe_allow_html=True)\n",
    "\n",
    "elif reset_btn:\n",
    "    st.experimental_rerun()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
