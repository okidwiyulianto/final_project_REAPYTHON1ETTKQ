import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import json
import os 
import re 

# --- Konfigurasi Awal & Konstanta ---
DEFAULT_IMAGE_URL = "https://placehold.co/200x300/CCCCCC/FFFFFF?text=No+Image&font=sans"
ITEMS_PER_PAGE = 5 

# GANTI DENGAN API KEY ANDA YANG VALID
HARDCODED_API_KEY_GOOGLE = "AIzaSyCnFDwxIHDoWFQ8tKBvdm4Yywkrgl1Oy_0" 
OPENROUTER_API_KEY = "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" 
OPENROUTER_DEFAULT_MODEL = "meta-llama/llama-4-maverick:free"  

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
        'user_specific_lists': {}, 
        'llm_cache': {},          
        'current_user_id': None    
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
    if st.session_state.google_books_api_key != HARDCODED_API_KEY_GOOGLE: 
        st.session_state.google_books_api_key = HARDCODED_API_KEY_GOOGLE
init_session_state()

# --- Fungsi Helper OpenRouter LLM ---
def call_openrouter_llm(prompt, model_name=OPENROUTER_DEFAULT_MODEL, max_tokens=300, temperature=0.5):
    if OPENROUTER_API_KEY == "SKOR-YOUR_OPENROUTER_API_KEY_HERE" or not OPENROUTER_API_KEY: # Ganti placeholder jika Anda menggunakan secrets
        # Coba ambil dari secrets jika placeholder tidak diganti manual
        api_key_secret = st.secrets.get("OPENROUTER_API_KEY")
        if not api_key_secret:
            st.error("API Key OpenRouter belum disetel. Harap konfigurasi di secrets atau di kode.")
            return None
        current_openrouter_api_key = api_key_secret
    else:
        current_openrouter_api_key = OPENROUTER_API_KEY

    app_url_referer = "http://localhost:8501" # Default aman untuk pengembangan lokal
    try:
        # Hanya coba akses st.secrets jika memang ada dan dikonfigurasi
        if hasattr(st, 'secrets') and "APP_URL" in st.secrets:
            app_url_referer = st.secrets.APP_URL
    except (AttributeError, FileNotFoundError): 
        # Biarkan app_url_referer menggunakan default jika secrets tidak ada atau APP_URL tidak ditemukan
        pass 
    except Exception: # Tangkap error lain yang mungkin muncul dari akses secrets
        pass

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {current_openrouter_api_key}", 
                     "Content-Type": "application/json",
                     "HTTP-Referer": app_url_referer, 
                     "X-Title": "Streamlit Book App"}, # Nama aplikasi Anda
            data=json.dumps({"model": model_name, 
                             "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": max_tokens, 
                             "temperature": temperature}),
            timeout=45 )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```json"): content = content[7:]
        if content.endswith("```"): content = content[:-3]
        return content.strip()
    except requests.exceptions.Timeout: st.warning("Timeout OpenRouter API."); return None
    except requests.exceptions.RequestException as e:
        error_detail = e.response.json() if e.response and hasattr(e.response, 'json') else str(e)
        st.warning(f"Error OpenRouter API: {str(e)[:100]} - Detail: {str(error_detail)[:200]}"); return None
    except (KeyError, IndexError, json.JSONDecodeError) as e: # Menambahkan json.JSONDecodeError
        response_text = locals().get('response', None)
        response_text = response_text.text if hasattr(response_text, 'text') else str(response_text)
        st.warning(f"Error parsing OpenRouter: {str(e)[:100]} - Respons: {response_text[:200]}"); return None

# --- Fungsi LLM untuk Fitur Detail (Hanya Terjemahan, Ringkasan Ulasan, "Baca Jika Suka") ---
def get_llm_cached_data(item_id, data_key):
    user_id = st.session_state.current_user_id
    if user_id and item_id and user_id in st.session_state.llm_cache and \
       item_id in st.session_state.llm_cache[user_id] and \
       data_key in st.session_state.llm_cache[user_id][item_id]:
        return st.session_state.llm_cache[user_id][item_id][data_key]
    return None

def set_llm_cached_data(item_id, data_key, value):
    user_id = st.session_state.current_user_id
    if user_id and item_id and value is not None:
        if user_id not in st.session_state.llm_cache: st.session_state.llm_cache[user_id] = {}
        if item_id not in st.session_state.llm_cache[user_id]: st.session_state.llm_cache[user_id][item_id] = {}
        st.session_state.llm_cache[user_id][item_id][data_key] = value

def translate_text_with_llm(text_to_translate, item_id_for_cache, data_key_suffix="", target_language="Bahasa Indonesia"):
    if not text_to_translate or text_to_translate in ["N/A", "Tidak ada deskripsi.", "Tidak ada latar belakang."]:
        return "Tidak ada teks untuk diterjemahkan."
    
    cache_key_name = f"translation_{target_language.replace(' ','_').lower()}{data_key_suffix}"
    cached_translation = get_llm_cached_data(item_id_for_cache, cache_key_name)
    if cached_translation: return cached_translation
    
    prompt = f"Terjemahkan teks berikut ke {target_language} yang baik, benar, dan natural. Pertahankan makna dan nuansa aslinya. Jika teks sudah dalam {target_language}, kembalikan teks asli. Teks asli:\n\n\"{text_to_translate}\""
    # Perkirakan token output bisa lebih panjang, terutama untuk bahasa Indonesia
    estimated_output_tokens = int(len(text_to_translate.split()) * 2.5 + 50) # Perkiraan kasar
    max_llm_tokens = min(estimated_output_tokens, 2000) # Batasi agar tidak terlalu besar

    translation = call_openrouter_llm(prompt, max_tokens=max_llm_tokens, temperature=0.3)
    
    if translation: set_llm_cached_data(item_id_for_cache, cache_key_name, translation)
    return translation if translation else "Gagal menerjemahkan saat ini."

def summarize_reviews_with_llm(review_texts, item_id_for_cache):
    if not review_texts or not isinstance(review_texts, list) or len(review_texts) == 0 : return None
    cached_summary = get_llm_cached_data(item_id_for_cache, "review_summary")
    if cached_summary: return cached_summary

    reviews_str = "\n\n".join([f"Ulasan {i+1}: \"{rev[:500]}\"" for i, rev in enumerate(review_texts[:3])]) 
    if not reviews_str.strip(): return None

    prompt = f"""Berikut adalah beberapa ulasan untuk sebuah karya:\n{reviews_str}\n\nIdentifikasi dan ringkas poin-poin positif utama (Pro) dan poin-poin negatif utama (Kontra) dari ulasan-ulasan di atas. Sajikan masing-masing maksimal 2-3 poin.

    PENTING SEKALI:
    1. Output HARUS dan HANYA berupa objek JSON yang valid.
    2. Respons harus dimulai dengan karakter kurung kurawal '{{' dan diakhiri dengan '}}'.
    3. JANGAN gunakan tanda kurung biasa '(' atau ')' untuk membungkus objek JSON.
    
    Contoh format yang benar: {{"pro": ["Alur cerita menarik.", "Karakter kuat."], "kontra": ["Alur terasa lambat di tengah."]}}

    Output JSON:
    """
    summary_str = call_openrouter_llm(prompt, max_tokens=500, temperature=0.2) # Temperature rendah untuk output terstruktur
    
    # st.write(f"DEBUG Ringkasan Ulasan Raw Output: '{summary_str}'") # Tetap aktifkan ini untuk debugging
    
    if summary_str:
        # Langkah Pembersihan Output LLM di Python
        clean_str = summary_str.strip()
        # Jika masih ada markdown code block, hapus
        if clean_str.startswith("```json"): clean_str = clean_str[7:]
        if clean_str.endswith("```"): clean_str = clean_str[:-3]
        # Jika ada tanda kurung pembungkus yang salah, hapus
        if clean_str.startswith('(') and clean_str.endswith(')'):
            clean_str = clean_str[1:-1]

        try:
            summary_data = json.loads(clean_str)
            if "pro" in summary_data and "kontra" in summary_data:
                set_llm_cached_data(item_id_for_cache, "review_summary", summary_data)
                return summary_data
            else:
                st.warning(f"Format JSON ringkasan ulasan tidak memiliki kunci 'pro'/'kontra': {clean_str}")
        except json.JSONDecodeError: 
            st.warning(f"Gagal parse JSON ringkasan ulasan (setelah dibersihkan): '{clean_str}'")
    return None

def get_read_if_you_like_with_llm(book_title, book_genre, book_description, item_id_for_cache):
    if not book_description or len(book_description) < 20: return None
    cached_recommendations = get_llm_cached_data(item_id_for_cache, "read_if_you_like")
    if cached_recommendations: return cached_recommendations

    prompt = f"""Analisis item berikut: Judul: "{book_title}", Genre: "{book_genre}", Deskripsi: "{book_description}"
    Berikan 2-3 saran "Baca/Tonton Jika Kamu Suka..." (bisa berupa judul karya lain yang mirip, penulis/sutradara dengan gaya serupa, atau konsep/tema umum yang relevan).
    Outputkan HANYA dalam format JSON dengan satu kunci "baca_jika_suka" yang nilainya adalah SEBUAH LIST BERISI STRING-STRING SARAN.
    Contoh JSON yang benar: {{"baca_jika_suka": ["Game of Thrones", "karya Joe Abercrombie", "cerita dengan akhir ambigu"]}}
    Contoh lain: {{"baca_jika_suka": ["Petualangan fantasi epik", "Dunia sihir yang kompleks"]}}
    Pastikan setiap elemen dalam list adalah string.
    """
    # TINGKATKAN MAX_TOKENS DI SINI
    recommendations_str = call_openrouter_llm(prompt, max_tokens=350, temperature=0.6) # Coba naikkan ke 300 atau 350
    st.write(f"DEBUG 'Baca Jika Suka' Raw Output LLM: '{recommendations_str}'")
    
    # st.write(f"DEBUG 'Baca Jika Suka' Raw Output LLM: '{recommendations_str}'") # Aktifkan ini untuk debugging

    if recommendations_str:
        try:
            json_match = None
            # Regex untuk mengekstrak blok JSON utama (termasuk yang multi-baris)
            match = re.search(r'\{[\s\S]*?\}', recommendations_str) 
            if match: 
                json_match = match.group(0)
            
            if json_match:
                rec_data = json.loads(json_match)
                if "baca_jika_suka" in rec_data and isinstance(rec_data["baca_jika_suka"], list):
                    if all(isinstance(item, str) for item in rec_data["baca_jika_suka"]):
                        set_llm_cached_data(item_id_for_cache, "read_if_you_like", rec_data["baca_jika_suka"])
                        return rec_data["baca_jika_suka"]
                    else:
                        st.warning(f"Elemen dalam list 'baca_jika_suka' bukan string: {rec_data['baca_jika_suka']}")
                        return None 
                else:
                    st.warning(f"Format 'Baca Jika Suka' LLM tidak sesuai (kunci 'baca_jika_suka' tidak ada atau bukan list): {json_match}")
            else:
                # Ini adalah pesan yang Anda lihat. Artinya regex tidak menemukan JSON yang lengkap.
                st.warning(f"Tidak ada JSON valid yang ditemukan dalam output 'Baca Jika Suka': {recommendations_str}")        
        except json.JSONDecodeError: 
            # Jika regex menemukan sesuatu yang mirip JSON tapi tidak valid
            st.warning(f"Gagal parse JSON 'Baca Jika Suka' (setelah ekstraksi regex): {json_match if json_match else recommendations_str}")
    return None 
# --- Fungsi Pengambilan Ulasan (Placeholder) ---
def get_item_reviews(item_data):
    source_api = item_data.get("source_api")
    item_api_id = item_data.get("id_buku_api") 
    # Hanya implementasi placeholder untuk MyAnimeList
    if source_api == "MyAnimeList" and item_api_id:
        mal_id_numeric = str(item_api_id).replace("mal_", "")
        # Untuk implementasi nyata, Anda akan memanggil Jikan API di sini
        # Contoh: https://api.jikan.moe/v4/manga/{mal_id_numeric}/reviews
        # atau /anime/{mal_id_numeric}/reviews
        # JANGAN LUPA BATASAN RATE JIKAN (jangan panggil terlalu sering saat testing)
        # st.caption(f"DEBUG: Akan mengambil review untuk MAL ID {mal_id_numeric}")
        return [
            f"Ini adalah ulasan placeholder pertama untuk item MAL ID {mal_id_numeric}. Ceritanya sangat menarik dengan karakter yang kuat.",
            f"Ulasan kedua untuk {mal_id_numeric}: Saya suka bagaimana penulis membangun dunia dalam karya ini. Sangat imersif!",
            f"Ulasan ketiga: Meskipun awalnya lambat, plotnya menjadi sangat intens menjelang akhir. Direkomendasikan!"
        ]
    # Anda bisa menambahkan logika untuk sumber lain jika API mereka menyediakan ulasan
    return ["Tidak ada ulasan yang tersedia untuk sumber ini atau fitur belum diimplementasikan."] 

# --- Fungsi Helper UI & Data Lokal ---
def display_book_card(book_data, key_prefix, column_context, is_from_df=False):
    with column_context: 
        book = book_data.to_dict() if isinstance(book_data, pd.Series) else book_data
        # Membuat ID item yang lebih unik untuk key widget
        item_unique_id_for_key = book.get('id_buku_api', book.get('id_buku', str(hash(book.get('judul','untitled_fb')))))
        key_suffix = f"{key_prefix}_{item_unique_id_for_key}"

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

def add_item_to_list(book_item_dict, list_name, user_id):
    if not isinstance(book_item_dict, dict): st.warning("Format item tidak valid."); return False
    if not user_id: st.warning("Pengguna belum login."); return False
    
    # Pastikan struktur user_specific_lists ada untuk user_id
    if user_id not in st.session_state.user_specific_lists: 
        st.session_state.user_specific_lists[user_id] = {}
    # Pastikan daftar spesifik ada untuk pengguna
    if list_name not in st.session_state.user_specific_lists[user_id]: 
        # Jika kita ingin membuat daftar secara otomatis saat pertama kali item disimpan ke nama daftar baru
        # st.session_state.user_specific_lists[user_id][list_name] = [] 
        # atau, jika daftar harus dibuat manual dulu di tab "Daftar Bacaan Saya":
        st.warning(f"Daftar '{list_name}' tidak ditemukan. Harap buat daftar terlebih dahulu di tab 'Daftar Bacaan Saya'.")
        return False
    
    item_id_to_check = book_item_dict.get('id_buku_api') or book_item_dict.get('id_buku')
    if not item_id_to_check: st.warning("Item tidak memiliki ID unik."); return False
    
    is_in_list = any( (entry.get('id_buku_api') or entry.get('id_buku')) == item_id_to_check
                     for entry in st.session_state.user_specific_lists[user_id][list_name] 
                     if isinstance(entry, dict) and (entry.get('id_buku_api') or entry.get('id_buku')))
    if not is_in_list: 
        st.session_state.user_specific_lists[user_id][list_name].append(book_item_dict)
        return True
    return False

# --- Fungsi Model Konten & Rekomendasi (Implementasi Fungsional Minimal) ---
# (Bagian ini dihapus total: update_content_based_model, tambah_buku_ke_df, rekomendasi_berdasarkan_popularitas, rekomendasi_berdasarkan_konten)

# --- Fungsi Pencarian API ---
# (Implementasi search_google_books, search_open_library_by_query, search_myanimelist, search_top_jikan dari kode sebelumnya yang sudah lengkap)
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
    base_url = "https://api.jikan.moe/v4/manga"; params = {"q": query if query else "", "page": page, "limit": limit, "sfw": "true"} 
    if order_by: params["order_by"] = order_by
    if sort: params["sort"] = sort
    if genres and isinstance(genres, list): params["genres"] = ",".join(str(g) for g in genres) 
    elif genres and isinstance(genres, str): params["genres"] = genres 
    if item_type_mal: params["type"] = item_type_mal
    results, has_more, next_page_params = [], False, {'page': page, 'limit': limit}
    if order_by: next_page_params["order_by"] = order_by; 
    if sort: next_page_params["sort"] = sort
    if genres: next_page_params["genres"] = genres; 
    if item_type_mal: next_page_params["type"] = item_type_mal
    try:
        response = requests.get(base_url, params=params, timeout=15); response.raise_for_status()
        data = response.json(); items = data.get("data", [])
        for item in items: 
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
        for item in items: 
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
def render_detail_buku_modal(buku): # buku adalah dictionary
    with st.container(): 
        st.subheader(f"Profil Lengkap")
        st.markdown("---")

        if isinstance(buku, pd.Series):
            buku = buku.to_dict()
        elif not isinstance(buku, dict):
            st.error("Format data buku tidak valid untuk ditampilkan.")
            if st.button("Tutup", key="btn_tutup_error_modal_detail"):
                st.session_state.show_detail_buku = None
                st.rerun()
            return

        item_id_modal_suffix = buku.get('id_buku_api') or buku.get('id_buku') or str(hash(buku.get('judul', np.random.randint(100000))))
        item_id_modal = f"modal_{item_id_modal_suffix}"

        # --- Bagian Info Utama Buku ---
        col_img, col_info_utama = st.columns([1, 2])
        with col_img: 
            st.image(buku.get('url_gambar_sampul', DEFAULT_IMAGE_URL), use_container_width=True)
        with col_info_utama:
            st.markdown(f"### {buku.get('judul', 'Judul Tidak Tersedia')}")
            st.caption(f"Oleh: {buku.get('penulis', 'Penulis Tidak Diketahui')}")
            st.markdown(f"**Genre:** {buku.get('genre', 'N/A')}")
            item_type_modal = buku.get('item_type', 'Buku')
            st.markdown(f"**Tipe Item:** {item_type_modal}")
            rating_modal = buku.get('rating_rata2', np.nan)
            jumlah_pembaca_modal = buku.get('jumlah_pembaca', np.nan)
            if pd.notna(rating_modal) and rating_modal > 0:
                st.markdown(f"**Rating:** {rating_modal:.1f}/5.0 (dari {int(jumlah_pembaca_modal) if pd.notna(jumlah_pembaca_modal) else 'N/A'} pembaca)")
            else: 
                st.markdown("**Rating:** Belum ada rating")
            source_api_modal = buku.get('source_api', 'Lokal')
            source_link_modal = None
            if source_api_modal == "MyAnimeList" and buku.get('mal_url'): source_link_modal = buku.get('mal_url')
            elif source_api_modal == "Google Books" and buku.get('gb_info_link'): source_link_modal = buku.get('gb_info_link')
            if source_link_modal: st.markdown(f"🔗 **[Lihat di {source_api_modal}]({source_link_modal})**")

        # --- Bagian Tentang Edisi Ini ---
        st.markdown("---"); st.subheader("Tentang Edisi Ini")
        col_detail1, col_detail2 = st.columns(2)
        # ... (Implementasi detail edisi seperti ISBN, penerbit, dll. dari kode Anda sebelumnya)
        with col_detail1:
            st.markdown(f"**Tanggal Publikasi:** {buku.get('tanggal_publikasi', 'N/A')}")
            st.markdown(f"**Penerbit:** {buku.get('penerbit', 'N/A')}")
            isbn_val_modal = buku.get('isbn_13')
            if isbn_val_modal or item_type_modal.lower() == 'book':
                 st.markdown(f"**ISBN-13:** {isbn_val_modal if isbn_val_modal else 'N/A'}")
        with col_detail2:
            jml_hal_modal = buku.get('jumlah_halaman')
            st.markdown(f"**Jml Halaman/Vol:** {str(int(jml_hal_modal)) if pd.notna(jml_hal_modal) and jml_hal_modal != 0 else 'N/A'}")
            st.markdown(f"**Bahasa:** {buku.get('bahasa', 'N/A')}")
            st.markdown(f"**Sumber Data Internal:** {source_api_modal}")


        # --- Detail Spesifik API (MyAnimeList & Google Books) ---
        if source_api_modal == "MyAnimeList":
            # ... (Implementasi detail spesifik MyAnimeList dari kode Anda sebelumnya)
            latar_belakang_mal = buku.get('mal_background') # Contoh, Anda punya banyak field MAL
            if latar_belakang_mal and latar_belakang_mal != "N/A":
                st.markdown("---"); st.subheader("Latar Belakang Cerita (MAL)")
                with st.expander("Lihat Latar Belakang", expanded=False):
                    st.markdown(latar_belakang_mal)
                    if st.session_state.current_user_id:
                        key_terjemah_bg_modal = f"btn_translate_bg_{item_id_modal}"
                        if st.button("Terjemahkan Latar ke ID 🇮🇩", key=key_terjemah_bg_modal):
                            with st.spinner("Menerjemahkan latar belakang..."):
                                translate_text_with_llm(latar_belakang_mal, item_id_modal, data_key_suffix="_bg") 
                            st.rerun()
                        cached_translation_bg = get_llm_cached_data(item_id_modal, "translation_bahasa_indonesia_bg")
                        if cached_translation_bg:
                            st.markdown("---"); st.markdown(f"**Terjemahan Latar (ID):**\n{cached_translation_bg}")
        
        if source_api_modal == "Google Books":
            # ... (Implementasi detail spesifik Google Books dari kode Anda sebelumnya)
            pass


        # --- Deskripsi dan Terjemahan (Dengan Tombol) ---
        st.markdown("---"); st.subheader("Deskripsi")
        desc_asli_modal = buku.get('deskripsi', 'Tidak ada deskripsi.')
        with st.expander("Deskripsi Lengkap", expanded=True):
            st.markdown(desc_asli_modal)
            if st.session_state.current_user_id and desc_asli_modal not in ["N/A", "Tidak ada deskripsi."]:
                key_terjemah_desc_modal = f"btn_translate_desc_{item_id_modal}"
                # Hanya tampilkan tombol jika belum ada terjemahan di cache
                cached_translation_desc = get_llm_cached_data(item_id_modal, "translation_bahasa_indonesia_desc")
                if not cached_translation_desc:
                    if st.button("Terjemahkan Deskripsi ke ID 🇮🇩", key=key_terjemah_desc_modal):
                        if desc_asli_modal and desc_asli_modal.strip() and desc_asli_modal != 'Tidak ada deskripsi.':
                            with st.spinner("Menerjemahkan deskripsi..."):
                                translate_text_with_llm(desc_asli_modal, item_id_modal, data_key_suffix="_desc")
                            st.rerun()
                        else:
                             st.caption("Tidak ada deskripsi untuk diterjemahkan.")
                
                if cached_translation_desc: # Tampilkan jika ada di cache (setelah diklik atau dari sesi sebelumnya)
                    st.markdown("---"); st.markdown(f"**Terjemahan Deskripsi (ID):**\n{cached_translation_desc}")
        
        # --- Ulasan dan Ringkasan Ulasan LLM (Dengan Tombol Pemicu) ---
        st.markdown("---"); st.subheader("Ulasan Pengguna")
        ulasan_item_modal = get_item_reviews(buku) 
        if ulasan_item_modal and isinstance(ulasan_item_modal, list) and len(ulasan_item_modal) > 0 and \
           not (len(ulasan_item_modal) == 1 and "Tidak ada ulasan" in ulasan_item_modal[0]):
            with st.expander("Lihat Ulasan (Maks. 3 Ditampilkan)", expanded=False):
                for i_rev, rev_text in enumerate(ulasan_item_modal[:3]): 
                    st.markdown(f"**Ulasan {i_rev+1}:**\n{rev_text}")
                    if i_rev < len(ulasan_item_modal[:3]) - 1: st.markdown("---")
            
            if st.session_state.current_user_id:
                cached_ringkasan = get_llm_cached_data(item_id_modal, "review_summary")
                
                if cached_ringkasan:
                    st.markdown("**Ringkasan Ulasan (AI):**")
                    if cached_ringkasan.get("pro") and cached_ringkasan["pro"]:
                        st.markdown("**👍 Pro:**")
                        # Ganti list comprehension dengan loop for biasa
                        for pro_point in cached_ringkasan["pro"][:3]:
                            st.markdown(f"- {pro_point}")
                    
                    if cached_ringkasan.get("kontra") and cached_ringkasan["kontra"]:
                        st.markdown("**👎 Kontra:**")
                        # Ganti list comprehension dengan loop for biasa
                        for con_point in cached_ringkasan["kontra"][:3]:
                            st.markdown(f"- {con_point}")
                else: # Jika belum ada di cache, tampilkan tombol
                    if st.button("Dapatkan Ringkasan Ulasan AI", key=f"btn_summarize_reviews_{item_id_modal}"):
                        with st.spinner("Membuat ringkasan ulasan dengan AI..."):
                            summarize_reviews_with_llm(ulasan_item_modal, item_id_modal)
                        st.rerun() # Rerun untuk menampilkan hasil dari cache
                    else:
                        st.caption("Klik tombol untuk mendapatkan ringkasan ulasan dari AI.")
        else:
            st.caption("Belum ada ulasan untuk item ini atau tidak dapat diambil.")

        # --- "Baca/Tonton Jika Kamu Suka..." oleh LLM (Dengan Tombol Pemicu) ---
        if st.session_state.current_user_id:
            st.markdown("---"); st.subheader("AI Menyarankan: Baca/Tonton Jika Suka...")
            cached_saran_baca = get_llm_cached_data(item_id_modal, "read_if_you_like")
            
            if cached_saran_baca and isinstance(cached_saran_baca, list):
                for saran_item in cached_saran_baca[:3]: 
                    st.markdown(f"- {saran_item}")
            else:
                if st.button("Dapatkan Saran AI (Baca Jika Suka...)", key=f"btn_read_if_like_{item_id_modal}"):
                    with st.spinner("Mencari saran terkait dari AI..."):
                        get_read_if_you_like_with_llm(
                            buku.get('judul',''), buku.get('genre',''), 
                            buku.get('deskripsi',''), item_id_modal
                        )
                    st.rerun()
                else:
                    st.caption("Klik tombol untuk mendapatkan saran dari AI.")
        
        # --- Tombol Simpan ke Daftar Bacaan ---
        st.markdown("---"); st.subheader("Simpan ke Daftar Bacaan")
        # ... (Implementasi Tombol Simpan ke Daftar Bacaan seperti kode Anda sebelumnya) ...
        if st.session_state.current_user_id:
            user_lists_modal = st.session_state.user_specific_lists.get(st.session_state.current_user_id, {})
            list_names_modal = list(user_lists_modal.keys())
            default_list_name_modal = "Item Tersimpan"

            # Opsi Expander untuk memilih daftar
            with st.expander("Pilih Daftar untuk Menyimpan", expanded=True):
                if not list_names_modal:
                     st.caption(f"Anda belum memiliki daftar kustom. Akan disimpan ke '{default_list_name_modal}'.")
                
                options_for_select = [default_list_name_modal] + list_names_modal
                chosen_list_to_save_modal = st.selectbox(
                    "Simpan ke:", 
                    options_for_select, 
                    key=f"sbox_modal_savelist_{item_id_modal}",
                    index=0 # Default ke "Item Tersimpan" atau daftar pertama jika ada
                )

                if st.button(f"Simpan ke '{chosen_list_to_save_modal}'", key=f"btn_modal_savelist_action_{item_id_modal}"):
                    # Buat daftar jika belum ada (terutama untuk "Item Tersimpan" atau daftar baru)
                    if st.session_state.current_user_id not in st.session_state.user_specific_lists:
                        st.session_state.user_specific_lists[st.session_state.current_user_id] = {}
                    if chosen_list_to_save_modal not in st.session_state.user_specific_lists[st.session_state.current_user_id]:
                        st.session_state.user_specific_lists[st.session_state.current_user_id][chosen_list_to_save_modal] = []
                    
                    if add_item_to_list(buku, chosen_list_to_save_modal, st.session_state.current_user_id):
                        st.success(f"'{buku.get('judul')}' disimpan ke '{chosen_list_to_save_modal}'.")
                    else: 
                        st.info(f"'{buku.get('judul')}' sudah ada di '{chosen_list_to_save_modal}'.")
        else:
            st.caption("Login untuk menyimpan item ke daftar bacaan Anda.")


        st.markdown("---")
        if st.button("Tutup Detail", key=f"close_modal_main_btn_unique_{item_id_modal}"): # Key unik
            st.session_state.show_detail_buku = None
            st.rerun()
# --- Antarmuka Streamlit Utama ---
st.set_page_config(layout="wide", page_title="Sistem Rekomendasi Media Interaktif")

# --- Sistem Login Sederhana di Sidebar ---
st.sidebar.header("Login Pengguna")
st.sidebar.caption("Masukkan ID unik untuk menyimpan preferensi dan daftar bacaan Anda dalam sesi ini.")

# Bagian ini (ketika pengguna SUDAH login) tidak perlu diubah.
if st.session_state.get("current_user_id"):
    st.sidebar.success(f"Masuk sebagai: {st.session_state.current_user_id}")
    if st.sidebar.button("Keluar", key="logout_button_main_sidebar"):
        # Reset semua state terkait pengguna
        st.session_state.current_user_id = None
        st.session_state.all_search_results = []
        # Anda bisa menambahkan reset state lain di sini jika perlu
        st.rerun()
else:
    # --- PERUBAHAN DIMULAI DI SINI ---

    # 1. Buat fungsi callback untuk memproses login
    def proses_login():
        # Ambil nilai input dari st.session_state menggunakan kuncinya
        user_id = st.session_state.user_id_input_main_sidebar
        
        if user_id and user_id.strip():
            # Jika input valid, set ID pengguna dan siapkan state
            st.session_state.current_user_id = user_id.strip()
            
            # Inisialisasi cache & daftar jika pengguna baru
            if st.session_state.current_user_id not in st.session_state.llm_cache:
                st.session_state.llm_cache[st.session_state.current_user_id] = {}
            if st.session_state.current_user_id not in st.session_state.user_specific_lists:
                st.session_state.user_specific_lists[st.session_state.current_user_id] = {}
            
            # Tidak perlu menampilkan pesan sukses di sini karena st.rerun()
            # akan menjalankan ulang skrip dan menampilkan pesan di blok 'if' di atas.
            st.rerun()
        else:
            # Jika input kosong, kita bisa menampilkan peringatan sementara,
            # meskipun idealnya validasi terjadi saat submit.
            # Untuk on_change, lebih baik tidak menampilkan apa-apa atau biarkan kosong.
            # Pengguna akan sadar inputnya tidak diterima karena tidak terjadi apa-apa.
            pass

    # 2. Gunakan on_change pada text_input dan hapus tombol "Masuk"
    st.sidebar.text_input(
        "ID Pengguna Pilihan Anda:",
        key="user_id_input_main_sidebar",
        help="Gunakan ID yang sama setiap kali untuk mengakses daftar Anda.",
        on_change=proses_login  # <-- Pemicu login otomatis ada di sini
    )
    
    # Pesan peringatan bisa ditampilkan di bawah input jika diperlukan,
    # tapi seringkali tidak diperlukan dengan pola on_change.
    if st.session_state.get('login_warning'):
        st.sidebar.warning(st.session_state.login_warning)
        del st.session_state.login_warning # Hapus setelah ditampilkan
# --- Struktur Tab Utama ---
if not st.session_state.current_user_id:
    st.info("👋 Selamat datang! Harap masukkan ID Pengguna Anda di sidebar kiri untuk memulai.")
    st.image("https://images.unsplash.com/photo-1532012197267-da84d127e765?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8Ym9va3N8ZW58MHx8MHx8fDA%3D&auto=format&fit=crop&w=500&q=60", 
             caption="Mulailah petualangan literasi Anda!", 
             width=400) # Atur lebar sesuai keinginan, atau gunakan use_container_width=True
else:
    # Tampilkan Tab hanya jika pengguna sudah "login"
    tab_cari, tab_daftar = st.tabs(["Cari & Temukan 🔎", "Daftar Bacaan Saya 📚"])

    # --- Kode sebelum 'with tab_cari:' tetap sama ---

    with tab_cari:
        st.title("📖 Cari & Temukan Buku, Manga, dan Lainnya")
        st.write("Ketik pencarian Anda di bawah dan tekan Enter untuk memulai.")

        # Kunci widget untuk konsistensi
        search_query_input_widget_key = "search_query_input_for_tab1"

        # --- PERUBAHAN 1: Membuat Fungsi Callback ---
        # Fungsi ini akan dijalankan setiap kali input teks berubah (saat menekan Enter).
        # Tujuannya adalah untuk MENYIAPKAN state untuk pencarian baru.
        def siapkan_pencarian():
            # Ambil query saat ini dari widget input
            query_saat_ini = st.session_state[search_query_input_widget_key]
            
            # Jika query kosong, jangan lakukan apa-apa
            if not query_saat_ini:
                return
                
            # Simpan query dan sumber pencarian terakhir
            st.session_state.last_searched_query = query_saat_ini
            st.session_state.last_selected_source_option = st.session_state.source_filter_radio
            
            # Reset semua hasil dan state pencarian sebelumnya
            st.session_state.all_search_results = []
            st.session_state.current_search_page = 1
            st.session_state.api_next_page_params = {}
            st.session_state.api_has_more_results = {}
            st.session_state.sources_exhausted_for_load_more = set()
            
            # Tandai bahwa pencarian baru telah dipicu
            st.session_state.search_triggered = True

        # --- PERUBAHAN 2: Menambahkan argumen 'on_change' pada st.text_input ---
        # Tombol "Cari" sudah tidak diperlukan lagi.
        st.text_input(
            "Cari judul, penulis, atau tema:",
            value=st.session_state.get("last_searched_query", ""),
            key=search_query_input_widget_key,
            on_change=siapkan_pencarian  # <-- Pemicu otomatis ada di sini
        )

        st.session_state.source_filter_radio = st.radio(
            "Pilih sumber pencarian:",
            ('Semua Sumber', 'Google Books', 'Open Library', 'MyAnimeList'),
            index=['Semua Sumber', 'Google Books', 'Open Library', 'MyAnimeList'].index(st.session_state.get("last_selected_source_option", "Semua Sumber")),
            horizontal=True, 
            key="source_filter_radio_tab1_key",
            on_change=siapkan_pencarian # <-- Tambahkan juga di sini agar perubahan sumber memicu pencarian ulang
        )

        # --- PERUBAHAN 3: Menjalankan pencarian berdasarkan 'search_triggered' ---
        # Logika ini sekarang terpisah dari input widget.
        if st.session_state.get("search_triggered", False):
            # Ambil query dan sumber dari session_state yang sudah disimpan oleh callback
            selected_source_tab1 = st.session_state.last_selected_source_option
            query_tab1_logic = st.session_state.last_searched_query
            
            # Jika query kosong, hentikan dan reset trigger
            if not query_tab1_logic:
                st.session_state.search_triggered = False
            else:
                processed_ids_in_current_search_tab1 = set()
                api_calls_tab1 = []
        
                if (selected_source_tab1 == "Google Books" or selected_source_tab1 == "Semua Sumber"):
                    if st.session_state.google_books_api_key and st.session_state.google_books_api_key != "YOUR_GOOGLE_BOOKS_API_KEY":
                        api_calls_tab1.append({'func': search_google_books, 'args': [query_tab1_logic, st.session_state.google_books_api_key], 'kwargs': {'startIndex': 0}, 'name': 'google_books', 'label': 'Google Books'})
                    elif selected_source_tab1 == "Google Books": 
                        st.warning("API Key Google Books tidak valid atau belum diset.")
                
                if selected_source_tab1 == "Open Library" or selected_source_tab1 == "Semua Sumber":
                    api_calls_tab1.append({'func': search_open_library_by_query, 'args': [query_tab1_logic], 'kwargs': {'page': 1}, 'name': 'open_library', 'label': 'Open Library'})
                
                if selected_source_tab1 == "MyAnimeList" or selected_source_tab1 == "Semua Sumber":
                    api_calls_tab1.append({'func': search_myanimelist, 'args': [query_tab1_logic], 'kwargs': {'page': 1}, 'name': 'myanimelist', 'label': 'MyAnimeList'})
                
                # Eksekusi panggilan API
                for call_info_tab1 in api_calls_tab1:
                    with st.spinner(f"Mencari di {call_info_tab1['label']} untuk '{query_tab1_logic}'..."):
                        res_list_tab1, has_more_api_tab1, next_params_api_tab1 = call_info_tab1['func'](*call_info_tab1['args'], **call_info_tab1['kwargs'])
                        count_added_tab1 = 0
                        for item_tab1 in res_list_tab1:
                            item_api_id_tab1 = item_tab1.get('id_buku_api')
                            if item_api_id_tab1 and item_api_id_tab1 not in processed_ids_in_current_search_tab1:
                                st.session_state.all_search_results.append(item_tab1)
                                processed_ids_in_current_search_tab1.add(item_api_id_tab1)
                                count_added_tab1 += 1
                        
                        st.session_state.api_has_more_results[call_info_tab1['name']] = has_more_api_tab1
                        st.session_state.api_next_page_params[call_info_tab1['name']] = next_params_api_tab1
                        if count_added_tab1 > 0: 
                            st.info(f"{call_info_tab1['label']}: {count_added_tab1} hasil awal ditemukan.")

                # Reset trigger agar tidak berjalan lagi pada rerun berikutnya
                st.session_state.search_triggered = False

                if not st.session_state.all_search_results: 
                    st.info("Tidak ada hasil pencarian yang ditemukan.")

        # --- Kode untuk menampilkan hasil pencarian (jika ada) tetap di sini ---
        # Misalnya:
        # if st.session_state.get("all_search_results"):
        #     st.write("Menampilkan hasil:")
        #     # ... logika display hasil ...

        if st.session_state.get('all_search_results'):
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
                st.markdown("---"); col1_img_t1, col2_info_btns_t1 = st.columns([1,3]) # Kolom untuk gambar dan info
                with col1_img_t1: st.image(buku_item_t1.get('url_gambar_sampul', DEFAULT_IMAGE_URL), width=100, caption=f"Sumber: {buku_item_t1.get('source_api','N/A')}")
                with col2_info_btns_t1:
                    item_type_main_t1 = buku_item_t1.get('item_type','N/A')
                    type_display_t1 = f" ({item_type_main_t1})" if item_type_main_t1 and item_type_main_t1.lower() not in ['book','buku'] else ""
                    st.markdown(f"**{buku_item_t1.get('judul','N/A')}**{type_display_t1}")
                    st.caption(f"Penulis: {buku_item_t1.get('penulis','N/A')}"); st.caption(f"Genre: {buku_item_t1.get('genre','N/A')}")
                    api_book_id_val_t1 = buku_item_t1.get('id_buku_api', f"search_tab1_{start_idx_c_t1 + i_t1}_{hash(buku_item_t1.get('judul'))}")
                    
                    btn_cols_t1_actions = st.columns(2) # Kolom untuk tombol aksi di bawah info
                    with btn_cols_t1_actions[0]:
                        if st.button("Lihat Detail", key=f"detail_s_tab1_{api_book_id_val_t1}",use_container_width=True):
                            st.session_state.show_detail_buku = buku_item_t1; st.session_state.detail_modal_origin_tab = "Cari & Temukan"; st.rerun()
                    with btn_cols_t1_actions[1]:
                         # Perbaikan Bug 4: Simpan ke Daftar Kustom
                        with st.expander("➕ Simpan ke Daftar", expanded=False):
                            user_lists_tab1 = st.session_state.user_specific_lists.get(st.session_state.current_user_id, {})
                            list_names_options_tab1 = list(user_lists_tab1.keys())
                            if not list_names_options_tab1:
                                st.caption("Buat daftar dulu di tab 'Daftar Bacaan Saya'.")
                                if st.button("Simpan ke 'Item Tersimpan'", key=f"save_def_exp_tab1_{api_book_id_val_t1}"):
                                    default_list_name_tab1 = "Item Tersimpan"
                                    if st.session_state.current_user_id not in st.session_state.user_specific_lists: st.session_state.user_specific_lists[st.session_state.current_user_id] = {}
                                    if default_list_name_tab1 not in st.session_state.user_specific_lists[st.session_state.current_user_id]:
                                        st.session_state.user_specific_lists[st.session_state.current_user_id][default_list_name_tab1] = []
                                    if add_item_to_list(buku_item_t1, default_list_name_tab1, st.session_state.current_user_id):
                                        st.success(f"'{buku_item_t1.get('judul')}' disimpan.")
                                    else: st.info(f"'{buku_item_t1.get('judul')}' sudah ada di daftar.")
                            else:
                                chosen_list_tab1 = st.selectbox("Pilih daftar:", options=["(Pilih...)"] + list_names_options_tab1, 
                                                                key=f"select_list_tab1_{api_book_id_val_t1}", index=0)
                                if chosen_list_tab1 and chosen_list_tab1 != "(Pilih...)":
                                    if st.button(f"Simpan ke '{chosen_list_tab1}'", key=f"btn_savelist_tab1_{api_book_id_val_t1}_{chosen_list_tab1}"):
                                        if add_item_to_list(buku_item_t1, chosen_list_tab1, st.session_state.current_user_id):
                                            st.success(f"'{buku_item_t1.get('judul')}' disimpan ke '{chosen_list_tab1}'.")
                                        else: st.info(f"'{buku_item_t1.get('judul')}' sudah ada di '{chosen_list_tab1}'.")
            if paginated_results_c_t1: st.markdown("---")

            if total_pages_disp_t1 > 1:
                # ... (Tombol Paginasi Lokal Sebelumnya & Berikutnya) ...
                nav_prev_c_t1, nav_info_c_t1, nav_next_c_t1 = st.columns([1,3,1])
                with nav_prev_c_t1:
                    if st.button("⬅️ Seb (Lokal)", disabled=(current_page_c_disp_t1 <= 1), key="prev_page_c_t1_key", use_container_width=True):
                        st.session_state.current_search_page -= 1; st.rerun()
                with nav_info_c_t1: st.markdown(f"<p style='text-align: center; margin-top: 0.5em;'>Hal. {current_page_c_disp_t1}/{total_pages_disp_t1} (Lokal)</p>", unsafe_allow_html=True)
                with nav_next_c_t1:
                    if st.button("Ber (Lokal) ➡️", disabled=(current_page_c_disp_t1 >= total_pages_disp_t1), key="next_page_c_t1_key", use_container_width=True):
                        st.session_state.current_search_page += 1; st.rerun()
            
            show_load_more_t1 = False; last_src_opt_t1 = st.session_state.get("last_selected_source_option", "Semua Sumber")
            source_map_t1 = {'google_books': 'Google Books', 'open_library': 'Open Library', 'myanimelist': 'MyAnimeList'}
            load_more_candidates_t1 = [src_label for src_key, src_label in source_map_t1.items() if (last_src_opt_t1 == src_label or last_src_opt_t1 == "Semua Sumber") and st.session_state.api_has_more_results.get(src_key) and src_key not in st.session_state.sources_exhausted_for_load_more]
            if load_more_candidates_t1: show_load_more_t1 = True
            
            if show_load_more_t1:
                # (Implementasi Tombol "Muat Lebih Banyak dari API..." seperti kode sebelumnya, dengan perbaikan `nonlocal` menjadi return value)
                st.markdown("---"); st.markdown(f"Sumber API yang mungkin masih memiliki hasil: {', '.join(load_more_candidates_t1)}")
                if st.button("Muat Lebih Banyak dari API...", key="load_more_api_tab1_button_key", type="primary", use_container_width=True):
                    query_lm_t1 = st.session_state.last_searched_query; source_opt_lm_t1 = st.session_state.last_selected_source_option
                    total_items_loaded_in_this_action_t1 = 0
                    def append_new_items_lm_t1(new_list_param_t1_func): # Nama parameter diubah
                        items_appended_this_call_t1_func = 0
                        current_ids_lm_t1_func = {item.get('id_buku_api') for item in st.session_state.all_search_results if item.get('id_buku_api')}
                        for item_to_add_lm_t1_func in new_list_param_t1_func:
                            item_api_id_add_lm_func = item_to_add_lm_t1_func.get('id_buku_api')
                            if item_api_id_add_lm_func and item_api_id_add_lm_func not in current_ids_lm_t1_func:
                                st.session_state.all_search_results.append(item_to_add_lm_t1_func)
                                current_ids_lm_t1_func.add(item_api_id_add_lm_func); items_appended_this_call_t1_func += 1
                        return items_appended_this_call_t1_func
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

    with tab_daftar:
        st.header("📚 Daftar Bacaan Saya")
        user_id_daftar = st.session_state.current_user_id
        
        # Pastikan struktur dasar ada untuk pengguna saat ini
        if user_id_daftar not in st.session_state.user_specific_lists:
            st.session_state.user_specific_lists[user_id_daftar] = {}
        
        current_user_lists_tab3 = st.session_state.user_specific_lists[user_id_daftar]

        new_list_name_tab3 = st.text_input("Buat Daftar Bacaan Baru:", placeholder="Nama Daftar (mis: Favorit)", key="new_list_name_input_tab3_user_key")
        if st.button("Buat Daftar", key="create_new_list_button_tab3_user_key"):
            if new_list_name_tab3.strip() and new_list_name_tab3.strip() not in current_user_lists_tab3:
                current_user_lists_tab3[new_list_name_tab3.strip()] = []
                st.session_state.user_specific_lists[user_id_daftar] = current_user_lists_tab3 # Simpan kembali
                st.success(f"Daftar '{new_list_name_tab3.strip()}' berhasil dibuat!"); st.rerun()
            elif not new_list_name_tab3.strip(): st.warning("Nama daftar tidak boleh kosong.")
            else: st.warning(f"Daftar '{new_list_name_tab3.strip()}' sudah ada.")
        
        st.markdown("---")
        
        if not current_user_lists_tab3:
            st.info("Anda belum memiliki daftar bacaan. Buat satu di atas atau simpan item dari hasil pencarian!")
        else:
            list_names_to_display = list(current_user_lists_tab3.keys())
            if not list_names_to_display: # Tambahan jika dictionary ada tapi kosong
                 st.info("Anda belum memiliki daftar bacaan. Buat satu di atas atau simpan item dari hasil pencarian!")
            else:
                selected_list_view_tab3 = st.selectbox(
                    "Pilih Daftar untuk Dilihat:",
                    options=list_names_to_display,
                    key="view_list_select_tab3_user_key",
                    index = 0 if list_names_to_display else None, # Pilih yg pertama jika ada
                    placeholder="Pilih daftar..." if not list_names_to_display else None
                )

                if selected_list_view_tab3 and selected_list_view_tab3 in current_user_lists_tab3:
                    st.subheader(f"Isi Daftar: {selected_list_view_tab3} ({len(current_user_lists_tab3[selected_list_view_tab3])} item)")
                    items_in_list_tab3 = current_user_lists_tab3[selected_list_view_tab3]
                    
                    if not items_in_list_tab3:
                        st.info(f"Daftar '{selected_list_view_tab3}' masih kosong.")
                    
                    for i_tab3, item_list_tab3 in enumerate(items_in_list_tab3):
                        item_id_list_tab3 = item_list_tab3.get('id_buku_api', item_list_tab3.get('id_buku', f"list_tab3_{i_tab3}_{hash(item_list_tab3.get('judul'))}"))
                        item_title_list_tab3 = item_list_tab3.get('judul', 'N/A')
                        
                        cols_list_item_tab3 = st.columns([1, 3, 1, 1]) 
                        with cols_list_item_tab3[0]: 
                            st.image(item_list_tab3.get('url_gambar_sampul', DEFAULT_IMAGE_URL), width=50)
                        with cols_list_item_tab3[1]:
                            st.markdown(f"**{item_title_list_tab3}**")
                            st.caption(f"Penulis: {item_list_tab3.get('penulis', 'N/A')} (Sumber: {item_list_tab3.get('source_api', 'N/A')})")
                        with cols_list_item_tab3[2]:
                            if st.button("🔍", key=f"detail_list_tab3_{item_id_list_tab3}", help="Lihat Detail"):
                                st.session_state.show_detail_buku = item_list_tab3
                                st.session_state.detail_modal_origin_tab = "Daftar Bacaan Saya" # Set origin
                                st.rerun()
                        with cols_list_item_tab3[3]:
                            if st.button("🗑️", key=f"remove_list_tab3_{item_id_list_tab3}", help="Hapus dari daftar"):
                                item_id_to_remove = item_list_tab3.get('id_buku_api') or item_list_tab3.get('id_buku')
                                st.session_state.user_specific_lists[user_id_daftar][selected_list_view_tab3] = [
                                    item_loop for item_loop in st.session_state.user_specific_lists[user_id_daftar][selected_list_view_tab3]
                                    if (item_loop.get('id_buku_api') or item_loop.get('id_buku')) != item_id_to_remove
                                ]
                                st.success(f"'{item_title_list_tab3}' dihapus dari '{selected_list_view_tab3}'."); st.rerun()
                        if i_tab3 < len(items_in_list_tab3) -1 : st.markdown("---", unsafe_allow_html=True) 
                    
                    if st.button(f"Hapus Daftar '{selected_list_view_tab3}' Ini", key=f"delete_list_btn_{selected_list_view_tab3}", type="secondary"):
                        if selected_list_view_tab3 in st.session_state.user_specific_lists[user_id_daftar]:
                            del st.session_state.user_specific_lists[user_id_daftar][selected_list_view_tab3]
                            st.success(f"Daftar '{selected_list_view_tab3}' telah dihapus."); st.rerun()
                elif list_names_to_display : # Jika ada daftar tapi tidak ada yang terpilih (misal setelah menghapus daftar yg aktif)
                    st.info("Pilih salah satu daftar Anda untuk dilihat.")



    # Modal Detail Buku (Luar struktur tab)
    if st.session_state.get('show_detail_buku') and st.session_state.current_user_id: 
        render_detail_buku_modal(st.session_state.show_detail_buku)



# --- Sidebar Informasi ---
st.sidebar.markdown("---")
st.sidebar.header("Tentang Aplikasi")
st.sidebar.info("Sistem rekomendasi media interaktif dengan pencarian multi-sumber, analisis LLM, dan daftar bacaan personal.")