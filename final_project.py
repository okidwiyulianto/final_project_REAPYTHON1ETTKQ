import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import json
import os 
import re # Untuk ekstraksi JSON yang lebih robas

# --- Konfigurasi Awal & Konstanta ---
DEFAULT_IMAGE_URL = "https://placehold.co/200x300/CCCCCC/FFFFFF?text=No+Image&font=sans"
ITEMS_PER_PAGE = 5 

# GANTI DENGAN API KEY ANDA YANG VALID
HARDCODED_API_KEY_GOOGLE = "AIzaSyCnFDwxIHDoWFQ8tKBvdm4Yywkrgl1Oy_0" 
OPENROUTER_API_KEY = "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" 
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-4-maverick:free"  
TINA_PROMPT_FILE = "tina_prompt.txt" 

# --- Fungsi untuk Memuat Prompt Tina ---
def load_tina_parsing_prompt():
    try:
        with open(TINA_PROMPT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"File prompt '{TINA_PROMPT_FILE}' tidak ditemukan. Harap buat file tersebut dengan prompt parsing parameter yang detail.")
        return """{"error": "Template prompt parsing tidak ditemukan"} Permintaan: "{user_prompt_tina}" """ # Fallback sangat dasar
    except Exception as e:
        st.error(f"Error saat memuat prompt Tina: {e}")
        return """{"error": "Gagal memuat template prompt parsing"} Permintaan: "{user_prompt_tina}" """

TINA_PARSING_PROMPT_TEMPLATE = load_tina_parsing_prompt()

# --- Inisialisasi Session State ---
def init_session_state():
    df_buku_cols = [
        'id_buku', 'judul', 'penulis', 'genre', 'deskripsi', 'rating_rata2', 
        'jumlah_pembaca', 'url_gambar_sampul', 'isbn_13', 'source_api', 
        'tanggal_publikasi', 'penerbit', 'jumlah_halaman', 'bahasa', 
        'id_buku_api', 'item_type'
    ]
    df_buku_dtypes = {
        'id_buku': 'Int64', 'rating_rata2': 'float64', 'jumlah_pembaca': 'float64',
        'jumlah_halaman': 'float64', 'judul': 'object', 'penulis': 'object',
        'genre': 'object', 'deskripsi': 'object', 'url_gambar_sampul': 'object',
        'isbn_13': 'object', 'source_api': 'object', 'tanggal_publikasi': 'object',
        'penerbit': 'object', 'bahasa': 'object', 'id_buku_api': 'object',
        'item_type': 'object'
    }
    defaults = {
        'df_buku': pd.DataFrame(columns=df_buku_cols).astype(df_buku_dtypes),
        'next_id_buku': 1,
        'google_books_api_key': HARDCODED_API_KEY_GOOGLE,
        'all_search_results': [], 'current_search_page': 1, 
        'search_triggered': False, 'show_detail_buku': None,
        'source_filter_radio': "Semua Sumber", 'api_next_page_params': {}, 
        'api_has_more_results': {}, 'last_searched_query': "",
        'last_selected_source_option': "Semua Sumber", 
        'sources_exhausted_for_load_more': set(), 'tfidf_matrix': None, 
        'cosine_sim_matrix': None, 'indices': pd.Series(dtype='int64'),
        'user_lists': {}, 'tina_chat_history': []
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if st.session_state.google_books_api_key != HARDCODED_API_KEY_GOOGLE: 
        st.session_state.google_books_api_key = HARDCODED_API_KEY_GOOGLE
init_session_state()

# --- Fungsi Helper OpenRouter LLM ---
def call_openrouter_llm(prompt, model_name=OPENROUTER_DEFAULT_MODEL, max_tokens=300, temperature=0.5): # Tambahkan parameter temperature
    if OPENROUTER_API_KEY == "SKOR-YOUR_OPENROUTER_API_KEY_HERE" or not OPENROUTER_API_KEY:
        st.error("Harap masukkan API Key OpenRouter Anda pada variabel OPENROUTER_API_KEY.")
        return None

    app_url_referer = "http://localhost:8501" 
    try:
        app_url_referer = st.secrets.get("APP_URL", "http://localhost:8501")
    except (AttributeError, FileNotFoundError): 
        pass 

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": app_url_referer, 
                "X-Title": "Streamlit Book App" 
            },
            data=json.dumps({
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, 
                "temperature": temperature # Gunakan nilai temperature dari parameter
            }),
            timeout=45 
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        # Pembersihan tambahan jika LLM masih menyertakan markdown
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
    except requests.exceptions.Timeout: 
        st.error("Timeout saat menghubungi OpenRouter API."); 
        return None
    except requests.exceptions.RequestException as e:
        error_detail = "Tidak ada detail respons."
        if e.response is not None:
            try: error_detail = e.response.json()
            except ValueError: error_detail = e.response.text
        st.error(f"Error OpenRouter API: {e} - Detail: {error_detail}"); 
        return None
    except (KeyError, IndexError) as e:
        error_detail_resp = "Tidak ada objek respons."
        response_obj = locals().get('response')
        if response_obj is not None:
             try: error_detail_resp = response_obj.json() 
             except ValueError: error_detail_resp = response_obj.text
        st.error(f"Error parsing OpenRouter: {e} - Respons: {error_detail_resp}"); 
        return None

# --- Fungsi Klasifikasi Intent, Augmentasi, dan Penjelasan LLM ---
def classify_user_intent_with_llm(user_input_intent):
    intent_prompt_text = f"""
    Analisis input pengguna berikut. Klasifikasikan maksudnya ke SALAH SATU dari nilai berikut: "greeting_or_chitchat", "recommendation_request", "general_query", atau "unknown".

    Input Pengguna: "{user_input_intent}"

    CONTOH RESPON YANG DIHARAPKAN (WAJIB HANYA JSON):
    Input: "halo" -> Output JSON: {{"intent": "greeting_or_chitchat"}}
    Input: "rekomendasi anime action" -> Output JSON: {{"intent": "recommendation_request"}}
    Input: "kamu siapa" -> Output JSON: {{"intent": "general_query"}}
    Input: "makanan enak" -> Output JSON: {{"intent": "unknown"}}

    PENTING SEKALI: Respons ANDA HARUS dan HANYA berupa objek JSON yang valid seperti pada contoh di atas.
    JANGAN sertakan teks penjelasan, kalimat pembuka, kalimat penutup, atau teks APAPUN selain objek JSON tersebut.
    Respons harus dimulai dengan '{{' dan diakhiri dengan '}}'.

    Output JSON:
    """
    response_str_intent = call_openrouter_llm(intent_prompt_text, max_tokens=80, temperature=0.1)
    
    # st.write(f"DEBUG Intent LLM Raw Output: '{response_str_intent}'") # Aktifkan untuk debugging
    
    if response_str_intent:
        try:
            json_match = None; match = re.search(r'\{.*?\}', response_str_intent, re.DOTALL)
            if match: json_match = match.group(0)
            if json_match:
                intent_data = json.loads(json_match)
                return intent_data.get("intent", "unknown")
            else: st.warning(f"Tidak ada JSON valid di raw output intent: '{response_str_intent}'"); return "unknown"
        except json.JSONDecodeError: 
            st.warning(f"Gagal parse JSON intent dari raw output: '{json_match if json_match else response_str_intent}'")
            return "unknown"
    return "unknown"

def augment_book_metadata_with_llm(book_description, book_title, book_genre):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    if not book_description or len(book_description) < 20:
        return {"tags": [], "sub_genre": book_genre, "target_audience": "Umum", "tone_mood": "Tidak diketahui"}
    prompt = f"""Analisis deskripsi buku berikut: Judul: "{book_title}", Genre Utama: "{book_genre}", Deskripsi: "{book_description}"
    Output JSON: {{"tags": ["kata kunci detail"], "sub_genre": "Sub Genre Lebih Spesifik", "target_audience": "Deskripsi Target Audiens", "tone_mood": "Deskripsi Nada dan Mood"}}
    Pastikan output HANYA berupa JSON yang valid."""
    augmented_data_str = call_openrouter_llm(prompt, max_tokens=300)
    if augmented_data_str:
        try:
            json_start = augmented_data_str.find('{'); json_end = augmented_data_str.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                data = json.loads(augmented_data_str[json_start:json_end])
                if all(k in data for k in ["tags", "sub_genre", "target_audience", "tone_mood"]): return data
                else: st.warning(f"Output JSON LLM augmentasi tidak lengkap. Output: {augmented_data_str[json_start:json_end]}")
            else: st.warning(f"JSON augmentasi tidak ditemukan di output LLM. Output: {augmented_data_str}")
        except json.JSONDecodeError as e: st.warning(f"Gagal parse JSON augmentasi: {e}. Output LLM: {augmented_data_str}")
    return {"tags": [], "sub_genre": book_genre, "target_audience": "Umum", "tone_mood": "Tidak diketahui"}


def explain_recommendation_with_llm(source_book_title, source_book_desc, rec_book_title, rec_book_desc):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    if not source_book_desc or not rec_book_desc: return "Tidak cukup informasi untuk penjelasan."
    prompt = f"""Buku "{rec_book_title}" direkomendasikan karena mirip dengan buku "{source_book_title}".
    Deskripsi "{source_book_title}": "{source_book_desc}"
    Deskripsi "{rec_book_title}": "{rec_book_desc}"
    Jelaskan secara singkat (1-2 kalimat) mengapa pembaca "{source_book_title}" mungkin juga akan tertarik dengan "{rec_book_title}", fokus pada kemungkinan kesamaan tema, gaya, premis, atau elemen cerita. Hindari frasa seperti "Berdasarkan perbandingan". Langsung saja ke penjelasannya."""
    explanation = call_openrouter_llm(prompt, max_tokens=100, temperature=0.7) # Boleh lebih kreatif di sini
    return explanation if explanation else "Tidak dapat menghasilkan penjelasan saat ini."

# --- Fungsi Helper UI & Data Lokal ---
# (Implementasi lengkap display_book_card, render_detail_buku_modal, add_item_to_default_list dari kode Anda sebelumnya)
def display_book_card(book_data, key_prefix, column_context, is_from_df=False):
    with column_context: 
        book = book_data.to_dict() if isinstance(book_data, pd.Series) else book_data
        item_id_val = book.get('id_buku') if is_from_df else book.get('id_buku_api', str(hash(book.get('judul', 'untitled_fb'))))
        key_suffix = f"{key_prefix}_{item_id_val}"
        st.image(book.get('url_gambar_sampul', DEFAULT_IMAGE_URL), use_container_width=True, width=100)
        st.markdown(f"**{book.get('judul', 'N/A')}**")
        st.caption(f"Oleh: {book.get('penulis', 'N/A')}")
        item_type_disp = book.get('item_type', 'Buku')
        if item_type_disp and item_type_disp.lower() not in ['book', 'buku']:
             st.caption(f"Tipe: {item_type_disp}")
        rating_val = book.get('rating_rata2', np.nan)
        if pd.notna(rating_val) and rating_val > 0: st.caption(f"Rating: {rating_val:.1f}/5.0")
        if st.button("Lihat Detail", key=f"detail_{key_suffix}", use_container_width=True):
            st.session_state.show_detail_buku = book; st.rerun()

def render_detail_buku_modal(buku):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    with st.container(): 
        st.subheader(f"Profil Lengkap") # ... (Sama seperti implementasi render_detail_buku_modal sebelumnya)
        st.markdown("---")
        col_img, col_info_utama = st.columns([1, 2])
        with col_img: st.image(buku.get('url_gambar_sampul', DEFAULT_IMAGE_URL), use_container_width=True)
        with col_info_utama:
            st.markdown(f"### {buku.get('judul', 'N/A')}")
            st.caption(f"Oleh: {buku.get('penulis', 'N/A')}")
            st.markdown(f"**Genre:** {buku.get('genre', 'N/A')}")
            item_type_modal = buku.get('item_type', 'Buku')
            st.markdown(f"**Tipe Item:** {item_type_modal}")
            rating_modal = buku.get('rating_rata2', np.nan)
            jumlah_pembaca_modal = buku.get('jumlah_pembaca', np.nan)
            if pd.notna(rating_modal) and rating_modal > 0:
                st.markdown(f"**Rating:** {rating_modal:.1f}/5.0 (dari {int(jumlah_pembaca_modal) if pd.notna(jumlah_pembaca_modal) else 'N/A'} pembaca)")
            else: st.markdown("**Rating:** Belum ada rating")
            source_api_modal = buku.get('source_api', 'Lokal')
            source_link_modal = None
            if source_api_modal == "MyAnimeList" and buku.get('mal_url'): source_link_modal = buku.get('mal_url')
            elif source_api_modal == "Google Books" and buku.get('gb_info_link'): source_link_modal = buku.get('gb_info_link')
            if source_link_modal: st.markdown(f"🔗 **[Lihat di {source_api_modal}]({source_link_modal})**")
        st.markdown("---"); st.subheader("Tentang Edisi Ini")
        col_detail1, col_detail2 = st.columns(2)
        with col_detail1:
            st.markdown(f"**Tgl Publikasi:** {buku.get('tanggal_publikasi', 'N/A')}")
            st.markdown(f"**Penerbit:** {buku.get('penerbit', 'N/A')}")
            isbn_val_modal = buku.get('isbn_13')
            if isbn_val_modal or item_type_modal.lower() == 'book':
                 st.markdown(f"**ISBN-13:** {isbn_val_modal if isbn_val_modal else 'N/A'}")
        with col_detail2:
            jml_hal_modal = buku.get('jumlah_halaman')
            st.markdown(f"**Jml Halaman/Vol:** {str(int(jml_hal_modal)) if pd.notna(jml_hal_modal) else 'N/A'}")
            st.markdown(f"**Bahasa:** {buku.get('bahasa', 'N/A')}")
            st.markdown(f"**Sumber Data Internal:** {source_api_modal}")
        if source_api_modal == "MyAnimeList":
            st.markdown("---"); st.subheader("Detail MyAnimeList")
            mal_col1, mal_col2 = st.columns(2)
            with mal_col1: # ... (Tampilkan detail MAL)
                if buku.get('mal_title_english') and buku.get('mal_title_english') != buku.get('judul'): st.markdown(f"**Judul Inggris:** {buku.get('mal_title_english')}")
                st.markdown(f"**Status:** {buku.get('mal_status', 'N/A')}")
                if buku.get('mal_chapters') is not None: st.markdown(f"**Total Chapter:** {buku.get('mal_chapters')}")
                if buku.get('mal_volumes') is not None: st.markdown(f"**Total Volume:** {buku.get('mal_volumes')}")
            with mal_col2:
                if buku.get('mal_rank') is not None: st.markdown(f"**Peringkat MAL:** #{buku.get('mal_rank')}")
                if buku.get('mal_popularity') is not None: st.markdown(f"**Popularitas MAL:** #{buku.get('mal_popularity')}")
                if buku.get('mal_members') is not None: st.markdown(f"**Anggota MAL:** {buku.get('mal_members'):,}")
                if buku.get('mal_favorites') is not None: st.markdown(f"**Favorit MAL:** {buku.get('mal_favorites'):,}")
            if buku.get('mal_background'):
                with st.expander("Latar Belakang Cerita (Background)"): st.markdown(buku.get('mal_background'))

        if source_api_modal == "Google Books":
            st.markdown("---"); st.subheader("Detail Google Books") # ... (Tampilkan detail GB)
            gb_col1, gb_col2 = st.columns(2)
            with gb_col1:
                if buku.get('gb_subtitle'): st.markdown(f"**Subjudul:** {buku.get('gb_subtitle')}")
                st.markdown(f"**Tipe Cetak:** {buku.get('gb_print_type', 'N/A')}")
            with gb_col2:
                st.markdown(f"**Rating Konten:** {buku.get('gb_maturity_rating', 'N/A')}")
                if buku.get('gb_preview_link'): st.markdown(f"🔗 [Pratinjau di Google Books]({buku.get('gb_preview_link')})")
        
        st.markdown("---"); st.subheader("Analisis Tambahan (LLM)")
        desc_aug = buku.get('deskripsi', ''); title_aug = buku.get('judul', 'N/A'); genre_aug = buku.get('genre', 'Umum')
        aug_key = f"augmented_data_{buku.get('id_buku_api', buku.get('id_buku', 'temp_aug'))}"
        if aug_key not in st.session_state: st.session_state[aug_key] = None
        if st.button("Dapatkan Analisis LLM", key=f"btn_aug_{aug_key}"):
            with st.spinner("Menganalisis dengan LLM..."): st.session_state[aug_key] = augment_book_metadata_with_llm(desc_aug, title_aug, genre_aug)
            st.rerun()
        if st.session_state[aug_key]:
            aug_info = st.session_state[aug_key]
            st.markdown(f"**Sub-Genre (LLM):** {aug_info.get('sub_genre', 'N/A')}")
            st.markdown(f"**Target Audiens (LLM):** {aug_info.get('target_audience', 'N/A')}")
            st.markdown(f"**Nada & Mood (LLM):** {aug_info.get('tone_mood', 'N/A')}")
            if aug_info.get('tags'): st.markdown(f"**Tag Detail (LLM):** {', '.join(aug_info['tags'])}")
        else: st.caption("Klik tombol untuk analisis LLM.")
        st.markdown("---"); st.subheader("Deskripsi")
        with st.expander("Lihat Deskripsi Lengkap", expanded=True): st.markdown(buku.get('deskripsi', 'N/A'))
        st.markdown("---"); st.subheader("Simpan ke Daftar Bacaan")
        modal_book_id = buku.get('id_buku_api', buku.get('id_buku', 'temp_modal_id'))
        if st.button("➕ Simpan ke 'Item Tersimpan'", key=f"save_from_modal_{modal_book_id}"):
            if add_item_to_default_list(buku): st.success(f"'{buku.get('judul')}' disimpan.")
            else: st.info(f"'{buku.get('judul')}' sudah ada di 'Item Tersimpan'.")
        st.markdown("---")
        modal_id_key = buku.get('id_buku_api', buku.get('id_buku', str(hash(buku.get('judul')))))
        if st.button("Tutup Detail", key=f"close_modal_{modal_id_key}"):
            st.session_state.show_detail_buku = None; st.rerun()

def add_item_to_default_list(book_item_dict, default_list_name="Item Tersimpan"):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    if not isinstance(book_item_dict, dict): st.warning("Format item tidak valid."); return False
    if default_list_name not in st.session_state.user_lists: st.session_state.user_lists[default_list_name] = []
    item_id_to_check = book_item_dict.get('id_buku_api') or book_item_dict.get('id_buku')
    if not item_id_to_check: st.warning("Item tidak memiliki ID unik."); return False
    is_in_list = any( (entry.get('id_buku_api') or entry.get('id_buku')) == item_id_to_check
                     for entry in st.session_state.user_lists[default_list_name] 
                     if isinstance(entry, dict) and (entry.get('id_buku_api') or entry.get('id_buku')))
    if not is_in_list: st.session_state.user_lists[default_list_name].append(book_item_dict); return True
    return False

# --- Fungsi Model Konten & Rekomendasi (Implementasi Fungsional Minimal) ---
# (Implementasi lengkap placeholder dari kode Anda sebelumnya)
def update_content_based_model():
    if not st.session_state.df_buku.empty and 'judul' in st.session_state.df_buku.columns:
        text_cols = ['genre', 'deskripsi', 'penulis', 'item_type']
        for col in text_cols:
            if col not in st.session_state.df_buku.columns: st.session_state.df_buku[col] = ""
            st.session_state.df_buku[col] = st.session_state.df_buku[col].fillna('')
        st.session_state.df_buku['konten'] = st.session_state.df_buku[text_cols].agg(' '.join, axis=1).str.lower()
        try: 
            konten_non_empty = st.session_state.df_buku['konten'].str.strip()
            valid_konten_for_tfidf = konten_non_empty[konten_non_empty != '']
            if not valid_konten_for_tfidf.empty:
                tfidf = TfidfVectorizer(stop_words='english', max_features=500, min_df=1) 
                st.session_state.tfidf_matrix = tfidf.fit_transform(valid_konten_for_tfidf)
                st.session_state.cosine_sim_matrix = cosine_similarity(st.session_state.tfidf_matrix, st.session_state.tfidf_matrix)
                df_used_for_tfidf = st.session_state.df_buku.loc[valid_konten_for_tfidf.index]
                valid_titles_tfidf = df_used_for_tfidf['judul'].notna() & (df_used_for_tfidf['judul'] != '')
                if valid_titles_tfidf.any():
                    st.session_state.indices = pd.Series(range(len(df_used_for_tfidf[valid_titles_tfidf])), 
                                                         index=df_used_for_tfidf[valid_titles_tfidf]['judul']).drop_duplicates()
                else: st.session_state.indices = pd.Series(dtype='int64')
            else: raise ValueError("Tidak ada konten valid untuk TF-IDF.")
        except ValueError: 
            st.session_state.tfidf_matrix = None; st.session_state.cosine_sim_matrix = None
            st.session_state.indices = pd.Series(dtype='int64')
    else:
        st.session_state.tfidf_matrix = None; st.session_state.cosine_sim_matrix = None
        st.session_state.indices = pd.Series(dtype='int64')
if 'tfidf_matrix' not in st.session_state: update_content_based_model()

def tambah_buku_ke_df(buku_api_dict):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    if not isinstance(buku_api_dict, dict): st.error("Data buku API tidak valid."); return
    api_judul = str(buku_api_dict.get('judul', '')).lower(); api_penulis = str(buku_api_dict.get('penulis', '')).lower()
    is_duplicate = False
    if not st.session_state.df_buku.empty and api_judul:
        is_duplicate = ((st.session_state.df_buku['judul'].astype(str).str.lower() == api_judul) & \
                        (st.session_state.df_buku['penulis'].astype(str).str.lower() == api_penulis)).any()
    if not is_duplicate:
        buku_baru_data = {col: buku_api_dict.get(col) for col in st.session_state.df_buku.columns if col != 'id_buku'}
        buku_baru_data['id_buku'] = st.session_state.next_id_buku
        for num_col in ['rating_rata2', 'jumlah_pembaca', 'jumlah_halaman']:
            if num_col in buku_baru_data: buku_baru_data[num_col] = pd.to_numeric(buku_baru_data[num_col], errors='coerce')
        if 'url_gambar_sampul' not in buku_baru_data or not buku_baru_data['url_gambar_sampul']:
            buku_baru_data['url_gambar_sampul'] = DEFAULT_IMAGE_URL
        new_row_df = pd.DataFrame([buku_baru_data], columns=st.session_state.df_buku.columns)
        for col, dtype in st.session_state.df_buku.dtypes.items():
            if col in new_row_df.columns:
                try: new_row_df[col] = new_row_df[col].astype(dtype)
                except Exception: 
                    if 'Int' in str(dtype): new_row_df[col] = pd.to_numeric(new_row_df[col], errors='coerce').astype(dtype)
                    pass
        st.session_state.df_buku = pd.concat([st.session_state.df_buku, new_row_df], ignore_index=True)
        st.session_state.next_id_buku += 1; update_content_based_model()
        st.success(f"'{buku_api_dict.get('judul')}' ditambahkan ke Pustaka Lokal!")
    else: st.warning(f"'{buku_api_dict.get('judul')}' sudah ada di Pustaka Lokal.")

def rekomendasi_berdasarkan_popularitas(top_n=5):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    if st.session_state.df_buku.empty or 'rating_rata2' not in st.session_state.df_buku.columns:
        return pd.DataFrame(columns=st.session_state.df_buku.columns)
    df_temp = st.session_state.df_buku.copy()
    df_temp['rating_rata2'] = pd.to_numeric(df_temp['rating_rata2'], errors='coerce').fillna(0)
    return df_temp.sort_values('rating_rata2', ascending=False).head(top_n)

def rekomendasi_berdasarkan_konten(judul_buku_pilihan, top_n=5):
    # (Implementasi lengkap dari kode Anda sebelumnya)
    if st.session_state.df_buku.empty or 'judul' not in st.session_state.df_buku.columns or \
       st.session_state.cosine_sim_matrix is None or st.session_state.indices.empty or \
       judul_buku_pilihan not in st.session_state.indices:
        return pd.DataFrame(columns=st.session_state.df_buku.columns)
    try:
        idx_posisi = st.session_state.indices[judul_buku_pilihan]
        if isinstance(idx_posisi, (pd.Series, np.ndarray)): idx_posisi = idx_posisi.iloc[0] 
        if not isinstance(idx_posisi, (int, np.integer)) or idx_posisi < 0 or idx_posisi >= st.session_state.cosine_sim_matrix.shape[0]:
            return pd.DataFrame(columns=st.session_state.df_buku.columns)
        sim_scores = list(enumerate(st.session_state.cosine_sim_matrix[idx_posisi]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        konten_non_empty = st.session_state.df_buku['konten'].str.strip()
        df_used_for_tfidf = st.session_state.df_buku.loc[konten_non_empty[konten_non_empty != ''].index]
        top_original_indices = []
        for i_score, score_val in sim_scores[1:]: 
            if len(top_original_indices) < top_n:
                if i_score < len(df_used_for_tfidf): 
                    original_df_index = df_used_for_tfidf.index[i_score] 
                    top_original_indices.append(original_df_index)
            else: break
        if not top_original_indices: return pd.DataFrame(columns=st.session_state.df_buku.columns)
        return st.session_state.df_buku.loc[top_original_indices]
    except KeyError: return pd.DataFrame(columns=st.session_state.df_buku.columns)
    except Exception: return pd.DataFrame(columns=st.session_state.df_buku.columns)

# --- Fungsi Pencarian API ---
# (Implementasi lengkap search_google_books, search_open_library_by_query, search_myanimelist, search_top_jikan dari kode sebelumnya)
def search_google_books(query, api_key, startIndex=0, maxResults=40):
    if not api_key or api_key == "YOUR_GOOGLE_BOOKS_API_KEY": return [], False, {} 
    base_url = "https://www.googleapis.com/books/v1/volumes"; params = {"q": query, "key": api_key, "startIndex": startIndex, "maxResults": maxResults, "langRestrict": "id"}
    hasil_buku, has_more, next_page_params = [], False, {'startIndex': startIndex, 'maxResults': maxResults}
    try:
        response = requests.get(base_url, params=params, timeout=10); response.raise_for_status()
        data = response.json(); items = data.get("items", [])
        for item in items:
            volume_info = item.get("volumeInfo", {})
            hasil_buku.append({
                "id_buku_api": item.get("id"), "judul": volume_info.get("title", "N/A"),
                "penulis": ", ".join(volume_info.get("authors", ["N/A"])), "item_type": "Book",
                "deskripsi": volume_info.get("description", "N/A"), "genre": ", ".join(volume_info.get("categories", ["Umum"])),
                "rating_rata2": volume_info.get("averageRating", np.nan), "jumlah_pembaca": volume_info.get("ratingsCount", np.nan),
                "url_gambar_sampul": volume_info.get("imageLinks", {}).get("thumbnail", DEFAULT_IMAGE_URL).replace("http://", "https://"),
                "isbn_13": next((id_obj.get('identifier') for id_obj in volume_info.get('industryIdentifiers', []) if id_obj.get('type') == 'ISBN_13'), None),
                "tanggal_publikasi": volume_info.get("publishedDate", "N/A"), "penerbit": volume_info.get("publisher", "N/A"),
                "jumlah_halaman": volume_info.get("pageCount", np.nan), "bahasa": volume_info.get("language", "N/A"),
                "source_api": "Google Books", "gb_info_link": volume_info.get("infoLink"), "gb_preview_link": volume_info.get("previewLink"),
                "gb_subtitle": volume_info.get("subtitle"), "gb_maturity_rating": volume_info.get("maturityRating"), "gb_print_type": volume_info.get("printType")
            })
        items_returned = len(items); total_items_api = data.get("totalItems", 0)
        if items_returned > 0 and (startIndex + items_returned) < total_items_api: has_more = True
        next_page_params = {'startIndex': startIndex + items_returned, 'maxResults': maxResults}
        return hasil_buku, has_more, next_page_params
    except requests.exceptions.Timeout: st.warning("Google Books API Timeout."); return [], False, {'startIndex': startIndex, 'maxResults': maxResults}
    except Exception as e: st.warning(f"Google Books API Error: {e}"); return [], False, {'startIndex': startIndex, 'maxResults': maxResults}

def search_open_library_by_query(query, page=1, limit=30):
    base_url = "http://openlibrary.org/search.json"; params = {"q": query, "page": page, "limit": limit, "fields": "key,cover_edition_key,title,author_name,first_sentence,subject_key,isbn,publisher,language,publish_date,number_of_pages_median"}
    hasil_buku_ol, has_more, next_page_params = [], False, {'page': page, 'limit': limit}
    try:
        response = requests.get(base_url, params=params, timeout=10); response.raise_for_status()
        data = response.json(); docs = data.get("docs", [])
        for item in docs:
            isbn_list = item.get("isbn", []); valid_isbns = [i for i in isbn_list if i and (len(i) == 10 or len(i) == 13)]
            isbn_13_ol = valid_isbns[0] if valid_isbns else None; cover_olid = item.get('cover_edition_key')
            img_url = DEFAULT_IMAGE_URL
            if cover_olid: img_url = f"https://covers.openlibrary.org/b/olid/{cover_olid}-L.jpg"
            elif isbn_13_ol: img_url = f"https://covers.openlibrary.org/b/isbn/{isbn_13_ol}-L.jpg"
            hasil_buku_ol.append({
                "id_buku_api": f"ol_{item.get('key','').split('/')[-1]}_{cover_olid or isbn_13_ol or np.random.randint(10000)}", 
                "judul": item.get("title", "N/A"), "penulis": ", ".join(item.get("author_name", ["N/A"])), "item_type": "Book",
                "deskripsi": (item.get("first_sentence") or ["N/A"])[0], "genre": ", ".join([s.replace("_"," ").title() for s in item.get("subject_key",[])[:3]]),
                "url_gambar_sampul": img_url, "source_api": "Open Library", "isbn_13": isbn_13_ol,
                "tanggal_publikasi": (item.get("publish_date") or [item.get("first_publish_year", "N/A")])[0],
                "penerbit": (item.get("publisher") or ["N/A"])[0], "jumlah_halaman": item.get("number_of_pages_median", np.nan),
                "bahasa": (item.get("language") or ["N/A"])[0],
            })
        items_returned = len(docs); total_items_api = data.get("numFound", 0)
        if items_returned > 0 and (page * limit) < total_items_api: has_more = True
        next_page_params = {'page': page + 1, 'limit': limit}
        return hasil_buku_ol, has_more, next_page_params
    except requests.exceptions.Timeout: st.warning("Open Library API Timeout."); return [], False, {'page': page, 'limit': limit}
    except Exception as e: st.warning(f"Open Library API Error: {e}"); return [], False, {'page': page, 'limit': limit}

def search_myanimelist(query, page=1, limit=25, order_by=None, sort=None, genres=None, item_type_mal=None):
    base_url = "https://api.jikan.moe/v4/manga"; params = {"q": query if query else "", "page": page, "limit": limit, "sfw": "true"} # Pastikan query tidak None
    if order_by: params["order_by"] = order_by
    if sort: params["sort"] = sort
    if genres and isinstance(genres, list): params["genres"] = ",".join(str(g) for g in genres) 
    elif genres and isinstance(genres, str): params["genres"] = genres # Jika sudah string dipisah koma
    if item_type_mal: params["type"] = item_type_mal
    results, has_more, next_page_params = [], False, {'page': page, 'limit': limit}
    if order_by: next_page_params["order_by"] = order_by; 
    if sort: next_page_params["sort"] = sort
    if genres: next_page_params["genres"] = genres; 
    if item_type_mal: next_page_params["type"] = item_type_mal
    try:
        response = requests.get(base_url, params=params, timeout=15); response.raise_for_status()
        data = response.json(); items = data.get("data", [])
        for item in items: # (Pemetaan field lengkap)
             results.append({
                "id_buku_api": f"mal_{item.get('mal_id')}", "judul": item.get("title", "N/A"),
                "penulis": ", ".join([a.get("name") for a in item.get("authors", []) if a.get("name")]) or "N/A",
                "item_type": item.get("type", "Manga"), "deskripsi": item.get("synopsis", "N/A"),
                "genre": ", ".join([g.get("name") for g in item.get("genres", []) if g.get("name")]) or "Umum",
                "rating_rata2": item.get("score", np.nan), "jumlah_pembaca": item.get("scored_by", np.nan),
                "url_gambar_sampul": item.get("images", {}).get("jpg", {}).get("image_url", DEFAULT_IMAGE_URL),
                "source_api": "MyAnimeList", "tanggal_publikasi": item.get("published", {}).get("string", "N/A"),
                "penerbit": ", ".join([s.get("name") for s in item.get("serializations", []) if s.get("name")]) or "N/A",
                "jumlah_halaman": item.get("volumes") if item.get("volumes") is not None else item.get("chapters", np.nan),
                "bahasa": "Japanese", "mal_url": item.get("url"), "mal_title_english": item.get("title_english"),
                "mal_title_japanese": item.get("title_japanese"), "mal_chapters": item.get("chapters"),
                "mal_volumes": item.get("volumes"), "mal_status": item.get("status"), "mal_rank": item.get("rank"),
                "mal_popularity": item.get("popularity"), "mal_members": item.get("members"),
                "mal_favorites": item.get("favorites"), "mal_background": item.get("background")
            })
        pagination_info = data.get("pagination", {})
        if items and pagination_info.get("has_next_page", False): has_more = True
        current_page_api = pagination_info.get("current_page", page); next_page_params['page'] = current_page_api + 1
        return results, has_more, next_page_params
    except requests.exceptions.Timeout: st.warning("MyAnimeList API Timeout."); return [], False, next_page_params
    except requests.exceptions.HTTPError as http_err:
        error_content = "Tidak diketahui"; err_resp = http_err.response
        if err_resp is not None:
            try: error_content = err_resp.json()
            except ValueError: error_content = err_resp.text
        st.warning(f"MyAnimeList API (Jikan) HTTP Error: {http_err} - Detail: {error_content}")
        return [], False, next_page_params
    except Exception as e: st.warning(f"MyAnimeList API Error: {e}"); return [], False, next_page_params

def search_top_jikan(item_type="anime", page=1, limit=5):
    if item_type not in ["anime", "manga"]: return [], False, {}
    base_url = f"https://api.jikan.moe/v4/top/{item_type}"; params = {"page": page, "limit": limit, "sfw": "true"}
    results, has_more, next_page_params = [], False, {'page': page, 'limit': limit, 'item_type': item_type}
    try:
        response = requests.get(base_url, params=params, timeout=15); response.raise_for_status()
        data = response.json(); items = data.get("data", [])
        for item in items: # (Pemetaan field lengkap)
            results.append({
                "id_buku_api": f"mal_{item.get('mal_id')}", "judul": item.get("title", "N/A"),
                "penulis": ", ".join([a.get("name") for a in item.get("authors", []) if a.get("name")]) or "N/A",
                "item_type": item.get("type", item_type.capitalize()), "deskripsi": item.get("synopsis", "N/A"),
                "genre": ", ".join([g.get("name") for g in item.get("genres", []) if g.get("name")]) or "Umum",
                "rating_rata2": item.get("score", np.nan), "jumlah_pembaca": item.get("scored_by", np.nan),
                "url_gambar_sampul": item.get("images", {}).get("jpg", {}).get("image_url", DEFAULT_IMAGE_URL),
                "source_api": "MyAnimeList", "tanggal_publikasi": item.get("published", {}).get("string", "N/A"),
                "mal_url": item.get("url"), "mal_rank": item.get("rank"), "mal_status": item.get("status"),
            })
        pagination_info = data.get("pagination", {})
        if items and pagination_info.get("has_next_page", False): has_more = True
        current_page_api = pagination_info.get("current_page", page); next_page_params['page'] = current_page_api + 1
        return results, has_more, next_page_params
    except requests.exceptions.Timeout: st.warning(f"Jikan API /top/{item_type} Timeout."); return [], False, next_page_params
    except requests.exceptions.HTTPError as http_err:
        error_content = "Tidak diketahui"; err_resp = http_err.response
        if err_resp is not None:
            try: error_content = err_resp.json()
            except ValueError: error_content = err_resp.text
        st.warning(f"Jikan API /top/{item_type} HTTP Error: {http_err} - Detail: {error_content}")
        return [], False, next_page_params
    except Exception as e: st.warning(f"Error proses Jikan /top/{item_type}: {e}"); return [], False, next_page_params

# --- Antarmuka Streamlit Utama ---
st.set_page_config(layout="wide", page_title="Sistem Rekomendasi Buku Interaktif")
tab_cari, tab_tina, tab_daftar = st.tabs(["Cari & Temukan 🔎", "Tina (Asisten AI) 🤖", "Daftar Bacaan Saya 📚"])

with tab_cari:
    st.title("📖 Cari & Temukan Buku, Manga, dan Lainnya")

    # Create a form to wrap the input field and the search button
    with st.form(key="search_form_tab1"):
        # Text input for search query within the form
        search_query_val_form = st.text_input(
            "Cari judul, penulis, atau tema:",
            value=st.session_state.get("last_searched_query", ""),  # Initialize with the last searched query
            key="search_text_input_in_form"  # Unique key for the text input within the form
        )

        # Radio button for source selection within the form
        source_selection_val_form = st.radio(
            "Pilih sumber pencarian:",
            ('Semua Sumber', 'Google Books', 'Open Library', 'MyAnimeList'),
            # Initialize with the last selected option
            index=['Semua Sumber', 'Google Books', 'Open Library', 'MyAnimeList'].index(
                st.session_state.get("last_selected_source_option", "Semua Sumber")
            ),
            horizontal=True,
            key="source_radio_in_form"  # Unique key for the radio button within the form
        )

        # Form submit button (replaces the original st.button)
        search_form_submitted = st.form_submit_button("Cari Item Online")

    # This block now executes if the form is submitted (by button click OR pressing Enter in the text input)
    if search_form_submitted:
        # When form is submitted, update session state with values from the form
        st.session_state.last_searched_query = search_query_val_form
        st.session_state.last_selected_source_option = source_selection_val_form
        
        # This line is no longer needed if st.session_state.source_filter_radio is consistently
        # updated or replaced by st.session_state.last_selected_source_option usage
        # st.session_state.source_filter_radio = source_selection_val_form 

        # --- This is the original search logic block ---
        st.session_state.all_search_results = []
        st.session_state.current_search_page = 1
        st.session_state.search_triggered = True
        # st.session_state.last_searched_query is now set from search_query_val_form
        # st.session_state.last_selected_source_option is now set from source_selection_val_form

        st.session_state.api_next_page_params = {}
        st.session_state.api_has_more_results = {}
        st.session_state.sources_exhausted_for_load_more = set()
        
        # Use the values from session state that were just updated by the form submission
        selected_source_tab1 = st.session_state.last_selected_source_option
        query_tab1 = st.session_state.last_searched_query
        
        processed_ids_in_current_search_tab1 = set()
        api_calls_tab1 = []
        
        # The original API call logic using query_tab1 and selected_source_tab1 follows here
        # This part of the logic (from your original file, approximately line 465 onwards) 
        # remains the same. Example:
        if (selected_source_tab1 == "Google Books" or selected_source_tab1 == "Semua Sumber"):
            if st.session_state.google_books_api_key and st.session_state.google_books_api_key != "YOUR_GOOGLE_BOOKS_API_KEY":
                api_calls_tab1.append({'func': search_google_books, 'args': [query_tab1, st.session_state.google_books_api_key], 'kwargs': {'startIndex': 0}, 'name': 'google_books', 'label': 'Google Books'})
            elif selected_source_tab1 == "Google Books": st.warning("API Key Google Books tidak valid atau belum diset.")
        if selected_source_tab1 == "Open Library" or selected_source_tab1 == "Semua Sumber":
            api_calls_tab1.append({'func': search_open_library_by_query, 'args': [query_tab1], 'kwargs': {'page': 1}, 'name': 'open_library', 'label': 'Open Library'})
        if selected_source_tab1 == "MyAnimeList" or selected_source_tab1 == "Semua Sumber":
            api_calls_tab1.append({'func': search_myanimelist, 'args': [query_tab1], 'kwargs': {'page': 1}, 'name': 'myanimelist', 'label': 'MyAnimeList'})
        
        for call_info_tab1 in api_calls_tab1:
            with st.spinner(f"Mencari di {call_info_tab1['label']} untuk '{query_tab1}'..."):
                res_list_tab1, has_more_api_tab1, next_params_api_tab1 = call_info_tab1['func'](*call_info_tab1['args'], **call_info_tab1['kwargs'])
                count_added_tab1 = 0
                for item_tab1 in res_list_tab1:
                    item_api_id_tab1 = item_tab1.get('id_buku_api')
                    if item_api_id_tab1 and item_api_id_tab1 not in processed_ids_in_current_search_tab1:
                        st.session_state.all_search_results.append(item_tab1)
                        processed_ids_in_current_search_tab1.add(item_api_id_tab1); count_added_tab1 +=1
                st.session_state.api_has_more_results[call_info_tab1['name']] = has_more_api_tab1
                st.session_state.api_next_page_params[call_info_tab1['name']] = next_params_api_tab1
                if count_added_tab1 > 0: st.info(f"{call_info_tab1['label']}: {count_added_tab1} hasil awal.")
        if not st.session_state.all_search_results: st.info("Tidak ada hasil pencarian.")

    if st.session_state.get('all_search_results'):
        # (Kode untuk menampilkan hasil pencarian dan paginasi sisi klien di Tab 1 - seperti sebelumnya)
        total_items_disp_t1 = len(st.session_state.all_search_results)
        total_pages_disp_t1 = (total_items_disp_t1 + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE if total_items_disp_t1 > 0 else 1
        current_page_c_disp_t1 = st.session_state.get('current_search_page', 1)
        current_page_c_disp_t1 = min(current_page_c_disp_t1, total_pages_disp_t1) 
        st.session_state.current_search_page = current_page_c_disp_t1
        start_idx_c_t1 = (current_page_c_disp_t1 - 1) * ITEMS_PER_PAGE; end_idx_c_t1 = start_idx_c_t1 + ITEMS_PER_PAGE
        paginated_results_c_t1 = st.session_state.all_search_results[start_idx_c_t1:end_idx_c_t1]
        display_query_t1 = st.session_state.get('last_searched_query', '')
        st.subheader(f"Hasil Pencarian '{display_query_t1}': {total_items_disp_t1} item (Hal. {current_page_c_disp_t1}/{total_pages_disp_t1})")
        if not paginated_results_c_t1 and total_items_disp_t1 > 0: st.info("Tidak ada item di halaman ini.")
        for i_t1, buku_item_t1 in enumerate(paginated_results_c_t1):
            st.markdown("---"); col1_img_t1, col2_info_btns_t1 = st.columns([1,3])
            with col1_img_t1: st.image(buku_item_t1.get('url_gambar_sampul', DEFAULT_IMAGE_URL), width=100, caption=f"Sumber: {buku_item_t1.get('source_api','N/A')}")
            with col2_info_btns_t1:
                item_type_main_t1 = buku_item_t1.get('item_type','N/A')
                type_display_t1 = f" ({item_type_main_t1})" if item_type_main_t1 and item_type_main_t1.lower() not in ['book','buku'] else ""
                st.markdown(f"**{buku_item_t1.get('judul','N/A')}**{type_display_t1}")
                st.caption(f"Penulis: {buku_item_t1.get('penulis','N/A')}"); st.caption(f"Genre: {buku_item_t1.get('genre','N/A')}")
                api_book_id_val_t1 = buku_item_t1.get('id_buku_api', f"search_tab1_{start_idx_c_t1 + i_t1}_{hash(buku_item_t1.get('judul'))}") # Key lebih unik
                btn_cols_t1 = st.columns(3) 
                with btn_cols_t1[0]:
                    if st.button("Lihat Detail", key=f"detail_s_tab1_{api_book_id_val_t1}",use_container_width=True):
                        st.session_state.show_detail_buku = buku_item_t1; st.rerun()
                with btn_cols_t1[1]: 
                    if st.button("Ke Pustaka", key=f"add_df_tab1_{api_book_id_val_t1}",use_container_width=True, help="Tambah ke Pustaka Lokal (df_buku)"):
                        tambah_buku_ke_df(buku_item_t1) 
                with btn_cols_t1[2]: 
                    if st.button("➕ Simpan", key=f"save_list_tab1_{api_book_id_val_t1}",use_container_width=True, help="Simpan ke 'Item Tersimpan'"):
                        if add_item_to_default_list(buku_item_t1): st.success(f"'{buku_item_t1.get('judul')}' disimpan.")
                        else: st.info(f"'{buku_item_t1.get('judul')}' sudah ada di 'Item Tersimpan'.")
        if paginated_results_c_t1: st.markdown("---")
        if total_pages_disp_t1 > 1:
            nav_prev_c_t1, nav_info_c_t1, nav_next_c_t1 = st.columns([1,3,1])
            with nav_prev_c_t1:
                if st.button("⬅️ Seb (Lokal)", disabled=(current_page_c_disp_t1 <= 1), key="prev_page_c_t1", use_container_width=True):
                    st.session_state.current_search_page -= 1; st.rerun()
            with nav_info_c_t1: st.markdown(f"<p style='text-align: center; margin-top: 0.5em;'>Hal. {current_page_c_disp_t1}/{total_pages_disp_t1} (Lokal)</p>", unsafe_allow_html=True)
            with nav_next_c_t1:
                if st.button("Ber (Lokal) ➡️", disabled=(current_page_c_disp_t1 >= total_pages_disp_t1), key="next_page_c_t1", use_container_width=True):
                    st.session_state.current_search_page += 1; st.rerun()
        
        show_load_more_t1 = False; last_src_opt_t1 = st.session_state.get("last_selected_source_option", "Semua Sumber")
        source_map_t1 = {'google_books': 'Google Books', 'open_library': 'Open Library', 'myanimelist': 'MyAnimeList'}
        load_more_candidates_t1 = [src_label for src_key, src_label in source_map_t1.items() if (last_src_opt_t1 == src_label or last_src_opt_t1 == "Semua Sumber") and st.session_state.api_has_more_results.get(src_key) and src_key not in st.session_state.sources_exhausted_for_load_more]
        if load_more_candidates_t1: show_load_more_t1 = True
        
        if show_load_more_t1:
            # (Implementasi Tombol "Muat Lebih Banyak dari API..." seperti kode Anda sebelumnya, dengan perbaikan `nonlocal`)
            st.markdown("---"); st.markdown(f"Sumber API yang mungkin masih memiliki hasil: {', '.join(load_more_candidates_t1)}")
            if st.button("Muat Lebih Banyak dari API...", key="load_more_api_tab1_button", type="primary", use_container_width=True):
                query_lm_t1 = st.session_state.last_searched_query; source_opt_lm_t1 = st.session_state.last_selected_source_option
                total_items_loaded_in_this_action_t1 = 0
                def append_new_items_lm_t1(new_list_param_t1):
                    items_appended_this_call_t1 = 0
                    current_ids_lm_t1 = {item.get('id_buku_api') for item in st.session_state.all_search_results if item.get('id_buku_api')}
                    for item_to_add_lm_t1 in new_list_param_t1:
                        item_api_id_add_lm = item_to_add_lm_t1.get('id_buku_api')
                        if item_api_id_add_lm and item_api_id_add_lm not in current_ids_lm_t1:
                            st.session_state.all_search_results.append(item_to_add_lm_t1)
                            current_ids_lm_t1.add(item_api_id_add_lm); items_appended_this_call_t1 += 1
                    return items_appended_this_call_t1
                api_call_map_lm_t1 = {
                    'Google Books': {'func': search_google_books, 'args': [query_lm_t1, st.session_state.google_books_api_key], 'name': 'google_books'},
                    'Open Library': {'func': search_open_library_by_query, 'args': [query_lm_t1], 'name': 'open_library'},
                    'MyAnimeList': {'func': search_myanimelist, 'args': [query_lm_t1], 'name': 'myanimelist'}
                }
                sources_to_load_from_t1 = []
                if source_opt_lm_t1 == "Semua Sumber": sources_to_load_from_t1 = load_more_candidates_t1
                elif source_opt_lm_t1 in api_call_map_lm_t1 and source_opt_lm_t1 in load_more_candidates_t1: sources_to_load_from_t1 = [source_opt_lm_t1]
                for src_label_lm_t1 in sources_to_load_from_t1:
                    call_detail_t1 = api_call_map_lm_t1[src_label_lm_t1]; src_name_lm_t1 = call_detail_t1['name']
                    params_lm_t1 = st.session_state.api_next_page_params.get(src_name_lm_t1, {})
                    if not params_lm_t1: 
                        if src_name_lm_t1 == 'google_books': params_lm_t1 = {'startIndex':0, 'maxResults': 40}
                        elif src_name_lm_t1 == 'open_library': params_lm_t1 = {'page':1, 'limit': 30}
                        else: params_lm_t1 = {'page':1, 'limit': 25}
                    with st.spinner(f"Memuat dari {src_label_lm_t1}..."):
                        new_list_api_t1, has_more_api_new_t1, next_params_api_new_t1 = call_detail_t1['func'](*call_detail_t1['args'], **params_lm_t1)
                        total_items_loaded_in_this_action_t1 += append_new_items_lm_t1(new_list_api_t1)
                        st.session_state.api_has_more_results[src_name_lm_t1] = has_more_api_new_t1
                        st.session_state.api_next_page_params[src_name_lm_t1] = next_params_api_new_t1
                        if not new_list_api_t1 and not has_more_api_new_t1: st.session_state.sources_exhausted_for_load_more.add(src_name_lm_t1)
                if total_items_loaded_in_this_action_t1 > 0:
                    st.success(f"{total_items_loaded_in_this_action_t1} item baru dimuat!"); st.session_state.current_search_page = (len(st.session_state.all_search_results) + ITEMS_PER_PAGE -1) // ITEMS_PER_PAGE
                else: st.info("Tidak ada item baru yang dapat dimuat.")
                st.rerun()
    elif st.session_state.get("search_triggered") and not st.session_state.get('all_search_results'):
         st.info("Tidak ada hasil pencarian untuk kueri ini.")

    st.markdown("---"); st.header("Populer di Pustaka Lokal Anda") 
    buku_populer_tab1 = rekomendasi_berdasarkan_popularitas(top_n=5)
    if not buku_populer_tab1.empty:
        cols_pop_tab1 = st.columns(min(5, len(buku_populer_tab1))) 
        for i_pop_tab1, (_, row_pop_tab1) in enumerate(buku_populer_tab1.iterrows()): 
            display_book_card(row_pop_tab1, "pop_tab1", cols_pop_tab1[i_pop_tab1], is_from_df=True)
    else: st.info("Tambahkan buku ke Pustaka Lokal untuk melihat item populer.")
    st.markdown("---"); st.header("Rekomendasi Sesuai Pilihan Anda (dari Pustaka Lokal)")
    if not st.session_state.df_buku.empty:
        valid_titles_rec_tab1 = [t for t in st.session_state.df_buku['judul'].dropna().unique() if str(t).strip() != ""]
        if len(valid_titles_rec_tab1) > 0:
            judul_pilihan_tab1 = st.selectbox("Pilih buku dari pustaka Anda:", valid_titles_rec_tab1, key="sb_konten_tab1", index=None, placeholder="Pilih buku...")
            if judul_pilihan_tab1:
                buku_pilihan_data_s_tab1 = st.session_state.df_buku[st.session_state.df_buku['judul'] == judul_pilihan_tab1].iloc[0]
                col_info_s_tab1, col_rec_s_tab1 = st.columns([2,3])
                with col_info_s_tab1:
                    st.subheader(f"Tentang '{judul_pilihan_tab1}':")
                    # Menampilkan kartu buku untuk item yang dipilih
                    container_detail_pilihan = st.container()
                    display_book_card(buku_pilihan_data_s_tab1, "sel_tab1", container_detail_pilihan, is_from_df=True)
                    with st.expander("Deskripsi Lengkap Pilihan"): st.markdown(buku_pilihan_data_s_tab1.get('deskripsi', 'N/A'))

                with col_rec_s_tab1:
                    st.subheader(f"Jika Anda suka '{judul_pilihan_tab1}', coba juga:")
                    rec_konten_tab1 = rekomendasi_berdasarkan_konten(judul_pilihan_tab1, top_n=3)
                    if not rec_konten_tab1.empty:
                        desc_buku_pilihan_s_tab1 = buku_pilihan_data_s_tab1.get('deskripsi', '')
                        cols_rec_s_inner_tab1 = st.columns(len(rec_konten_tab1))
                        for j_idx_tab1, (_, baris_rec_s_tab1) in enumerate(rec_konten_tab1.iterrows()):
                            display_book_card(baris_rec_s_tab1, f"rec_cont_tab1_{j_idx_tab1}", cols_rec_s_inner_tab1[j_idx_tab1], is_from_df=True)
                            rec_title_s_tab1 = baris_rec_s_tab1.get('judul', 'N/A'); rec_desc_s_tab1 = baris_rec_s_tab1.get('deskripsi', '')
                            expl_key_s_tab1 = f"expl_tab1_{buku_pilihan_data_s_tab1.get('id_buku', 'src')}_{baris_rec_s_tab1.get('id_buku', f'r{j_idx_tab1}')}"
                            if expl_key_s_tab1 not in st.session_state: st.session_state[expl_key_s_tab1] = None
                            with cols_rec_s_inner_tab1[j_idx_tab1]: 
                                if st.button(f"Kenapa ini?", key=f"btn_expl_tab1_{expl_key_s_tab1}", help="Penjelasan dari LLM", use_container_width=True):
                                    with st.spinner("Meminta penjelasan..."):
                                        st.session_state[expl_key_s_tab1] = explain_recommendation_with_llm(judul_pilihan_tab1, desc_buku_pilihan_s_tab1, rec_title_s_tab1, rec_desc_s_tab1)
                                    st.rerun()
                                if st.session_state[expl_key_s_tab1]: st.caption(f"AI: {st.session_state[expl_key_s_tab1]}")
                    else: st.info("Tidak ada rekomendasi konten yang cocok atau model belum siap.")
        else: st.info("Pustaka Lokal tidak memiliki buku dengan judul valid untuk dipilih.")
    else: st.info("Pustaka Lokal kosong. Tambahkan buku untuk fitur ini.")


with tab_tina:
    # (Implementasi lengkap Tab Tina dari kode sebelumnya)
    st.header("🤖 Berbincang dengan Tina, Asisten Pustakawan AI Anda")
    # ... (Sama seperti implementasi tab_tina sebelumnya, pastikan `classify_user_intent_with_llm` didefinisikan) ...
    def classify_user_intent_with_llm(user_input_intent): # Definisikan di scope yang benar atau pass sebagai argumen
        intent_prompt_text = f"""Klasifikasikan maksud input pengguna: "greeting_or_chitchat", "recommendation_request", "general_query". Input: "{user_input_intent}". Output JSON: {{"intent": "JENIS_INTENT"}}"""
        response_str_intent = call_openrouter_llm(intent_prompt_text, max_tokens=50, model_name=OPENROUTER_DEFAULT_MODEL) # Gunakan model yang lebih cepat jika ada
        st.write(f"DEBUG Intent LLM Raw Output: '{response_str_intent}'") 
        if response_str_intent:
            try:
                json_start_intent = response_str_intent.find('{'); json_end_intent = response_str_intent.rfind('}') + 1
                if json_start_intent != -1 and json_end_intent != -1:
                    intent_data = json.loads(response_str_intent[json_start_intent:json_end_intent])
                    return intent_data.get("intent")
            except json.JSONDecodeError: return "unknown"
        return "unknown"
    
    for message_tina_hist in st.session_state.tina_chat_history:
        with st.chat_message(message_tina_hist["role"]):
            st.markdown(message_tina_hist["content"])
            if message_tina_hist["role"] == "assistant" and "recommendations_data" in message_tina_hist and message_tina_hist["recommendations_data"]:
                for i_tina_chat_hist, rec_book_tina_hist in enumerate(message_tina_hist["recommendations_data"]):
                    rec_title_tina_hist = rec_book_tina_hist.get('judul', 'N/A')
                    rec_id_tina_hist = rec_book_tina_hist.get('id_buku_api', f"hist_tina_rec_{i_tina_chat_hist}_{hash(rec_title_tina_hist)}")
                    cols_chat_rec_tina_hist_disp = st.columns([3,1,1])
                    with cols_chat_rec_tina_hist_disp[0]: st.markdown(f"➡️ _{rec_title_tina_hist}_")
                    with cols_chat_rec_tina_hist_disp[1]:
                        if st.button("Detail", key=f"detail_tina_hist_{rec_id_tina_hist}", use_container_width=True):
                            st.session_state.show_detail_buku = rec_book_tina_hist; st.rerun()
                    with cols_chat_rec_tina_hist_disp[2]:
                        if st.button("Simpan", key=f"save_tina_hist_{rec_id_tina_hist}", use_container_width=True):
                            if add_item_to_default_list(rec_book_tina_hist): st.success(f"'{rec_title_tina_hist}' disimpan.")
                            else: st.info(f"'{rec_title_tina_hist}' sudah ada di 'Item Tersimpan'.")
    
    user_prompt_tina_input = st.chat_input("Ketik permintaanmu di sini... (mis: 5 anime fantasi petualangan)")

    if user_prompt_tina_input:
        st.session_state.tina_chat_history.append({"role": "user", "content": user_prompt_tina_input})
        with st.chat_message("user"):
            st.markdown(user_prompt_tina_input)

        with st.chat_message("assistant"):
            with st.spinner("Tina sedang memproses..."):
                user_input_lower = user_prompt_tina_input.lower().strip()
                
                determined_intent = "unknown" # Default intent

                # --- Definisi Variabel untuk Deteksi Praktis ---
                common_greetings = [
                    "halo", "hai", "hi", "hei", "pagi", "siang", "sore", "malam", 
                    "apa kabar", "kabar baik", "selamat pagi", "selamat siang", 
                    "selamat sore", "selamat malam", "thanks", "terima kasih", "makasih"
                ]
                request_keywords = [ # Kata kunci yang kuat mengindikasikan permintaan rekomendasi/pencarian
                    "rekomendasi", "rekomendasikan", "kasih rekomendasi", "minta rekomendasi",
                    "cari", "carikan", "ada", "top", "terbaik", "teratas", "populer",
                    "genre", "tema", "penulis", "pengarang", "saran", "judul", "buku", "manga", 
                    "anime", "manhwa", "manhua", "donghua", "novel", "komik", "skor tertinggi",
                    "rating tertinggi", "rilis terbaru"
                ]
                # --- Akhir Definisi Variabel ---

                # --- Logika Intent yang Diperbarui (Deteksi Praktis) ---
                is_pure_greeting = False # Inisialisasi di sini
                if len(user_input_lower.split()) <= 3: 
                    for greeting in common_greetings:
                        # Cek apakah input adalah sapaan itu sendiri atau dimulai dengan sapaan tersebut
                        if greeting == user_input_lower or \
                           (user_input_lower.startswith(greeting + " ") or user_input_lower.startswith(greeting + "!")):
                            is_pure_greeting = True
                            break
                
                if is_pure_greeting:
                    determined_intent = "greeting_or_chitchat"
                else:
                    is_recommendation_keyword_present = False
                    for keyword in request_keywords: # Gunakan variabel yang sudah didefinisikan
                        if keyword in user_input_lower:
                            is_recommendation_keyword_present = True
                            break
                    
                    if is_recommendation_keyword_present:
                        determined_intent = "recommendation_request"
                    else:
                        # Jika bukan sapaan murni DAN tidak ada kata kunci permintaan eksplisit,
                        # anggap sebagai general_conversation.
                        determined_intent = "general_conversation" 
                # --- Akhir Logika Intent ---
                
                # st.write(f"DEBUG: Intent Praktis Terdeteksi -> {determined_intent}") # Untuk debugging

                assistant_response_text = ""
                recommendations_for_chat_tina_list = [] # Reset untuk setiap giliran

                if determined_intent == "recommendation_request":
                    # ... (LOGIKA LENGKAP PARSING PARAMETER DETAIL DENGAN LLM DAN PEMANGGILAN API BUKU/MANGA) ...
                    # Ini menggunakan TINA_PARSING_PROMPT_TEMPLATE
                    final_parsing_prompt_tina_chat = TINA_PARSING_PROMPT_TEMPLATE.format(user_prompt_tina=user_prompt_tina_input)
                    llm_params_str_tina_chat_p = call_openrouter_llm(final_parsing_prompt_tina_chat, max_tokens=400, temperature=0.2) # temperature rendah untuk JSON
                    
                    if llm_params_str_tina_chat_p:
                        try:
                            json_match_param = None; match_param = re.search(r'\{.*?\}', llm_params_str_tina_chat_p, re.DOTALL)
                            if match_param: json_match_param = match_param.group(0)
                            
                            if json_match_param:
                                params_tina_chat = json.loads(json_match_param)
                                
                                # Anda perlu memproses SEMUA parameter yang diekstrak LLM di sini
                                # untuk memanggil fungsi API yang sesuai (search_... atau search_top_jikan).
                                # Bagian ini adalah yang paling kompleks dan membutuhkan perhatian detail.
                                query_chat_llm = params_tina_chat.get("query") 
                                item_type_llm = params_tina_chat.get("type", "book").lower()
                                limit_llm = int(params_tina_chat.get("limit", 5))
                                sort_by_llm = params_tina_chat.get("sort_by")
                                genre_llm = params_tina_chat.get("genre") # Ini list atau null
                                author_llm = params_tina_chat.get("author")
                                # ... (Ambil parameter lain seperti year_start, year_end, exclude_genre, format, status, demographic, rating_min, dll.)
                                
                                api_results_chat_list_p = [] 
                                # CONTOH LOGIKA PEMANGGILAN API (PERLU DISESUAIKAN DAN DILENGKAPI DENGAN SEMUA PARAMETER)
                                if sort_by_llm in ["rating", "popularity"] and \
                                   (item_type_llm in ["anime", "manga", "manhwa", "manhua", "donghua"]) and \
                                   not query_chat_llm and not genre_llm and not author_llm: # Permintaan top umum
                                    api_results_chat_list_p, _, _ = search_top_jikan(item_type=item_type_llm, limit=limit_llm)
                                elif item_type_llm in ["anime", "manga", "manhwa", "manhua", "donghua"]:
                                    api_results_chat_list_p, _, _ = search_myanimelist(
                                        query=query_chat_llm if query_chat_llm else "", 
                                        page=1, limit=limit_llm, 
                                        order_by=sort_by_llm if sort_by_llm in ["score","rank","popularity","title","start_date", "favorites", "members"] else None, 
                                        sort="desc" if sort_by_llm and sort_by_llm not in ["title", "alphabetical"] else ("asc" if sort_by_llm in ["title", "alphabetical"] else None),
                                        genres=genre_llm, # search_myanimelist perlu diupdate untuk menangani list genre dari LLM
                                        item_type_mal=item_type_llm if item_type_llm not in ["anime","manga"] else None # type spesifik Jikan
                                        # Tambahkan parameter lain seperti author_llm, status, demographic ke search_myanimelist
                                    )
                                elif item_type_llm in ["book", "novel"]:
                                    effective_query_book = query_chat_llm if query_chat_llm else (genre_llm[0] if genre_llm else (author_llm if author_llm else "populer indonesia")) # Kueri default yang lebih baik
                                    # Modifikasi search_google_books dan search_open_library untuk menerima lebih banyak parameter (genre, author, sort_by, dll.)
                                    api_results_chat_list_p_gb, _, _ = search_google_books(effective_query_book, st.session_state.google_books_api_key, maxResults=limit_llm)
                                    api_results_chat_list_p.extend(api_results_chat_list_p_gb)
                                    if len(api_results_chat_list_p) < limit_llm: # Coba OL jika GB kurang
                                         api_results_chat_list_p_ol, _, _ = search_open_library_by_query(effective_query_book, limit = limit_llm - len(api_results_chat_list_p))
                                         api_results_chat_list_p.extend(api_results_chat_list_p_ol)
                                
                                if api_results_chat_list_p:
                                    assistant_response_text = f"Baik! Tina menemukan {len(api_results_chat_list_p)} item untukmu berdasarkan '{user_prompt_tina_input}':\n"
                                    for idx_c, book_c_llm in enumerate(api_results_chat_list_p[:limit_llm]): # Pastikan tidak melebihi limit asli
                                        skor_disp_llm = f" (Skor: {book_c_llm.get('rating_rata2', 'N/A')})" if pd.notna(book_c_llm.get('rating_rata2')) else ""
                                        rank_disp_llm = f" (Rank: #{book_c_llm.get('mal_rank', 'N/A')})" if pd.notna(book_c_llm.get('mal_rank')) and sort_by_llm in ["rating", "popularity"] else ""
                                        assistant_response_text += f"{idx_c+1}. **{book_c_llm.get('judul', 'N/A')}**{skor_disp_llm}{rank_disp_llm} (Tipe: {book_c_llm.get('item_type', 'N/A')})\n"
                                    recommendations_for_chat_tina_list = api_results_chat_list_p[:limit_llm]
                                else: 
                                    assistant_response_text = f"Maaf, Tina tidak menemukan hasil pencarian yang cocok untuk '{user_prompt_tina_input}' saat ini. Mungkin coba kriteria lain?"
                            else: 
                                assistant_response_text = "Tina kesulitan memahami format parameter dari AI (tidak ada JSON terdeteksi)."
                                st.write(f"DEBUG: LLM Parsing Output Mentah -> '{llm_params_str_tina_chat_p}'") 
                        except json.JSONDecodeError: 
                            assistant_response_text = "Respons parameter dari AI tidak valid (gagal memproses JSON)."
                            st.write(f"DEBUG: LLM Parsing Output Mentah Gagal JSONDecode -> '{llm_params_str_tina_chat_p}'")
                        except Exception as e_chat_p: 
                            assistant_response_text = f"Oops, terjadi kesalahan saat Tina mencari: {str(e_chat_p)[:100]}"
                            st.exception(e_chat_p) 
                    else: 
                        assistant_response_text = "Tina tidak bisa menghubungi layanan AI untuk memproses permintaan pencarian."
                
                elif determined_intent == "greeting_or_chitchat":
                    sapaan_options_tina = [
                        "Halo juga! Ada yang bisa Tina bantu carikan buku, manga, atau anime hari ini?",
                        "Hai! Kabar baik. Sedang ingin mencari bacaan atau tontonan apa?",
                        "Selamat datang! Jangan ragu untuk bertanya jika kamu butuh rekomendasi."
                    ]
                    assistant_response_text = np.random.choice(sapaan_options_tina)
                
                elif determined_intent == "general_conversation": 
                    general_conversation_prompt = f"""
                    Kamu adalah Tina, asisten pustakawan AI yang ramah dan membantu.
                    Pengguna berkata: "{user_prompt_tina_input}"
                    Tanggapi percakapan pengguna secara natural dan singkat (1-2 kalimat). Jika pengguna tampak bertanya sesuatu di luar topik buku/manga/anime, jawablah dengan sopan bahwa fokusmu adalah membantu terkait hal tersebut, namun tetap berikan respons yang relevan jika memungkinkan.
                    Hindari memberikan rekomendasi kecuali diminta secara eksplisit dalam input pengguna ini.
                    """
                    assistant_response_text = call_openrouter_llm(general_conversation_prompt, max_tokens=150, temperature=0.7)
                    if not assistant_response_text: 
                        assistant_response_text = "Hmm, Tina sedang memikirkan jawabannya. Bisa coba tanyakan hal lain?"
                
                else: # determined_intent == "unknown"
                    assistant_response_text = "Maaf, Tina belum yakin bagaimana merespons itu. Bisa coba tanyakan tentang rekomendasi buku, manga, atau anime?"

                st.markdown(assistant_response_text)
                if recommendations_for_chat_tina_list: 
                    st.session_state.tina_chat_history.append({"role": "assistant", "content": assistant_response_text, "recommendations_data": recommendations_for_chat_tina_list})
                else:
                    st.session_state.tina_chat_history.append({"role": "assistant", "content": assistant_response_text})

with tab_daftar:
    # (Implementasi lengkap Tab Daftar Bacaan dari kode sebelumnya)
    st.header("📚 Daftar Bacaan Saya")
    new_list_name_tab3 = st.text_input("Buat Daftar Bacaan Baru:", placeholder="Nama Daftar", key="new_list_name_input_tab3")
    if st.button("Buat Daftar", key="create_new_list_button_tab3"):
        if new_list_name_tab3 and new_list_name_tab3 not in st.session_state.user_lists:
            st.session_state.user_lists[new_list_name_tab3] = []; st.success(f"Daftar '{new_list_name_tab3}' dibuat!"); st.rerun()
        elif not new_list_name_tab3: st.warning("Nama daftar tidak boleh kosong.")
        else: st.warning(f"Daftar '{new_list_name_tab3}' sudah ada.")
    st.markdown("---")
    if not st.session_state.user_lists: st.info("Belum ada daftar bacaan.")
    else:
        selected_list_view_tab3 = st.selectbox("Pilih Daftar:", options=list(st.session_state.user_lists.keys()), key="view_list_select_tab3")
        if selected_list_view_tab3:
            st.subheader(f"Isi: {selected_list_view_tab3} ({len(st.session_state.user_lists[selected_list_view_tab3])} item)")
            items_in_list_tab3 = st.session_state.user_lists[selected_list_view_tab3]
            if not items_in_list_tab3: st.info(f"Daftar '{selected_list_view_tab3}' kosong.")
            for i_tab3, item_list_tab3 in enumerate(items_in_list_tab3):
                item_id_list_tab3 = item_list_tab3.get('id_buku_api', item_list_tab3.get('id_buku', f"list_tab3_{i_tab3}_{hash(item_list_tab3.get('judul'))}")) # Key lebih unik
                item_title_list_tab3 = item_list_tab3.get('judul', 'N/A')
                cols_list_item_tab3 = st.columns([1, 3, 1, 1])
                with cols_list_item_tab3[0]: st.image(item_list_tab3.get('url_gambar_sampul', DEFAULT_IMAGE_URL), width=50)
                with cols_list_item_tab3[1]:
                    st.markdown(f"**{item_title_list_tab3}**"); st.caption(f"Penulis: {item_list_tab3.get('penulis', 'N/A')} (Sumber: {item_list_tab3.get('source_api', 'N/A')})")
                with cols_list_item_tab3[2]:
                    if st.button("🔍", key=f"detail_list_tab3_{item_id_list_tab3}", help="Lihat Detail"):
                        st.session_state.show_detail_buku = item_list_tab3; st.rerun()
                with cols_list_item_tab3[3]:
                    if st.button("🗑️", key=f"remove_list_tab3_{item_id_list_tab3}", help="Hapus"):
                        item_id_to_remove = item_list_tab3.get('id_buku_api') or item_list_tab3.get('id_buku')
                        st.session_state.user_lists[selected_list_view_tab3] = [item for item in st.session_state.user_lists[selected_list_view_tab3] if (item.get('id_buku_api') or item.get('id_buku')) != item_id_to_remove]
                        st.success(f"'{item_title_list_tab3}' dihapus."); st.rerun()
                if i_tab3 < len(items_in_list_tab3) -1 : st.markdown("---", unsafe_allow_html=True) 
            if st.button(f"Hapus Daftar '{selected_list_view_tab3}' Ini", key=f"delete_list_btn_{selected_list_view_tab3}", type="secondary"):
                if selected_list_view_tab3 in st.session_state.user_lists:
                    del st.session_state.user_lists[selected_list_view_tab3]; st.success(f"Daftar '{selected_list_view_tab3}' dihapus."); st.rerun()

if st.session_state.get('show_detail_buku'):
    render_detail_buku_modal(st.session_state.show_detail_buku)

st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("Sistem rekomendasi buku/manga interaktif dengan pencarian multi-sumber, asisten AI, dan daftar bacaan personal.")
st.sidebar.header("Pustaka Lokal (df_buku)")
if not st.session_state.df_buku.empty:
    cols_sidebar = ['id_buku', 'judul', 'item_type', 'rating_rata2']
    valid_cols_sidebar = [col for col in cols_sidebar if col in st.session_state.df_buku.columns]
    st.sidebar.caption(f"Total di Pustaka Lokal: {len(st.session_state.df_buku)}")
    if valid_cols_sidebar: st.sidebar.dataframe(st.session_state.df_buku[valid_cols_sidebar].tail(), hide_index=True, use_container_width=True)
else: st.sidebar.text("Pustaka Lokal (df_buku) kosong.")