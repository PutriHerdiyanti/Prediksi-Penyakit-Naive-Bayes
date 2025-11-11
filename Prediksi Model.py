{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "97293f1a-5b2a-4813-adbb-7f22e7f8f1a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pickle"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "c7504dbc-a6d9-498c-870c-d4369a1fc4dc",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-11-11 15:34:09.196 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\User\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "# Judul Aplikasi\n",
    "st.set_page_config(page_title=\"Prediksi Penyakit\", layout=\"centered\")\n",
    "st.title(\"🤖 Prediksi Penyakit Berdasarkan Gejala\")\n",
    "st.write(\"Pilih gejala-gejala yang Anda alami untuk memprediksi kemungkinan penyakit.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "80850c14-1721-4f38-98f2-64598addaacd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# --- Fungsi untuk Memuat Model dan Metadata ---\n",
    "@st.cache_resource\n",
    "def load_resources():\n",
    "    \"\"\"Memuat model, nama kelas, dan nama fitur dari file pickle.\"\"\"\n",
    "    try:\n",
    "        model = pickle.load(open('prediksi_penyakit_model.sav', 'rb'))\n",
    "        class_names = pickle.load(open('class_names.sav', 'rb'))\n",
    "        feature_names = pickle.load(open('feature_names.sav', 'rb'))\n",
    "        return model, class_names, feature_names\n",
    "    except FileNotFoundError:\n",
    "        st.error(\"Error: File model (.sav) tidak ditemukan. Jalankan skrip `buat_model.py` terlebih dahulu.\")\n",
    "        return None, None, None\n",
    "    except Exception as e:\n",
    "        st.error(f\"Terjadi kesalahan saat memuat model: {e}\")\n",
    "        return None, None, None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "c0b19a39-cab6-42c0-a33e-2f4f9f03d267",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Muat model dan metadata\n",
    "model, class_names, feature_names = load_resources()\n",
    "\n",
    "if model is not None:\n",
    "    # --- Buat Form Input Gejala ---\n",
    "    with st.form(\"gejala_form\"):\n",
    "        st.header(\"Pilih Gejala Anda (1 = Ya, 0 = Tidak)\")\n",
    "        \n",
    "        col1, col2 = st.columns(2)\n",
    "        user_input_list = []\n",
    "        half_len = len(feature_names) // 2\n",
    "        \n",
    "        with col1:\n",
    "            for i, feature in enumerate(feature_names[:half_len]):\n",
    "                label = feature.replace(\"_\", \" \").title()\n",
    "                user_input = st.selectbox(\n",
    "                    label,\n",
    "                    options=[0, 1],\n",
    "                    format_func=lambda x: \"Ya\" if x == 1 else \"Tidak\",\n",
    "                    key=f\"feat_{i}\"\n",
    "                )\n",
    "                user_input_list.append(user_input)\n",
    "\n",
    "        with col2:\n",
    "            for i, feature in enumerate(feature_names[half_len:], start=half_len):\n",
    "                label = feature.replace(\"_\", \" \").title()\n",
    "                user_input = st.selectbox(\n",
    "                    label,\n",
    "                    options=[0, 1],\n",
    "                    format_func=lambda x: \"Ya\" if x == 1 else \"Tidak\",\n",
    "                    key=f\"feat_{i}\"\n",
    "                )\n",
    "                user_input_list.append(user_input)\n",
    "\n",
    "        submitted = st.form_submit_button(\"Prediksi Penyakit\")\n",
    "\n",
    "    # --- Logika Prediksi ---\n",
    "    if submitted:\n",
    "        if sum(user_input_list) == 0:\n",
    "            st.warning(\"Anda tidak memilih gejala apapun. Silakan pilih minimal satu gejala.\")\n",
    "        else:\n",
    "            input_array = np.array(user_input_list)\n",
    "            input_reshaped = input_array.reshape(1, -1)\n",
    "            \n",
    "            try:\n",
    "                prediction_index = model.predict(input_reshaped)\n",
    "                disease_name = class_names[prediction_index[0]]\n",
    "                \n",
    "                st.success(f\"Hasil Prediksi: **{disease_name}**\")\n",
    "                st.info(\"Catatan: Ini adalah prediksi berdasarkan model Naive Bayes dan tidak menggantikan diagnosis medis profesional.\")\n",
    "                \n",
    "            except Exception as e:\n",
    "                st.error(f\"Terjadi kesalahan saat melakukan prediksi: {e}\")"
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
