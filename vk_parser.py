# Импорт библиотек
import math
import time
import re
import requests
import configparser
import os
import threading

import pandas as pd
import numpy as np

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from wordcloud import WordCloud

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
from gensim import corpora
from gensim.models import LdaModel

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Настройки NLTK
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass
nltk.download('stopwords', quiet=True)


# Файл конфигурации
CONFIG_FILE = "config.ini"
config = configparser.ConfigParser()
if os.path.exists(CONFIG_FILE):
    config.read(CONFIG_FILE)
else:
    config['VK'] = {'token': '', 'domain_short': 'tassagency'}


# Tkinter окно ввода
class InputWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VK Parser Input")
        self.geometry("750x310")
        self.resizable(False, False)
        self.cancel_flag = False


        frm = tk.Frame(self)
        frm.pack(padx=10, pady=10, fill='x')


        tk.Label(frm, text="VK API Token:").grid(row=0, column=0, sticky='w', pady=4)
        self.token_entry = tk.Entry(frm, width=112)
        self.token_entry.grid(row=0, column=1, sticky='ew', pady=4)
        self.token_entry.insert(0, config['VK'].get('token', ''))


        tk.Label(frm, text="Домен группы:").grid(row=1, column=0, sticky='w', pady=4)
        self.domain_entry = tk.Entry(frm, width=75)
        self.domain_entry.grid(row=1, column=1, sticky='ew', pady=4)
        self.domain_entry.insert(0, config['VK'].get('domain_short', 'tassagency'))


        tk.Label(frm, text="Количество постов:").grid(row=2, column=0, sticky='w', pady=4)
        self.posts_entry = tk.Entry(frm, width=75)
        self.posts_entry.grid(row=2, column=1, sticky='ew', pady=4)
        self.posts_entry.insert(0, "500")


        tk.Label(frm, text="Пропустить закрепленный (первый) пост").grid(row=3, column=0, sticky='w', pady=4)
        self.skip_pinned_var = tk.BooleanVar(value=False)
        self.skip_pinned_cb = tk.Checkbutton(frm, variable=self.skip_pinned_var)
        self.skip_pinned_cb.grid(row=3, column=1, sticky='w', pady=4)


        tk.Label(frm, text="Дополнительные стоп-слова (через запятую):").grid(row=4, column=0, sticky='w', pady=4)
        self.stop_entry = tk.Entry(frm, width=75)
        self.stop_entry.grid(row=4, column=1, sticky='ew', pady=4)


        tk.Label(frm, text="Сохранить результаты в Excel").grid(row=5, column=0, sticky='w', pady=4)
        self.save_excel_var = tk.BooleanVar(value=False)
        self.save_excel_cb = tk.Checkbutton(frm, variable=self.save_excel_var)
        self.save_excel_cb.grid(row=5, column=1, sticky='w', pady=4)


        frm.grid_columnconfigure(1, weight=1)


        tk.Label(frm, text="Прогресс:").grid(row=6, column=0, sticky='w', pady=4)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frm, maximum=100, variable=self.progress_var, length=550)
        self.progress_bar.grid(row=6, column=1, sticky='ew', pady=4)


        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        self.start_btn = tk.Button(btn_frame, text="Начать", command=self.start_analysis_thread)
        self.start_btn.pack(side='left', padx=5)
        self.cancel_btn = tk.Button(btn_frame, text="Отменить", command=self.cancel_analysis)
        self.cancel_btn.pack(side='left', padx=5)


        entries = [self.token_entry, self.domain_entry, self.posts_entry, self.stop_entry]
        self.setup_paste_bindings(entries)


    def setup_paste_bindings(self, entries):
        for entry in entries:
            entry.bind("<Control-v>", self.paste_handler)
            entry.bind("<Command-v>", self.paste_handler)
            entry.bind("<Shift-Insert>", self.paste_handler)
            entry.bind("<Button-3>", self.show_context_menu)


    def paste_handler(self, event):
        try:
            entry = event.widget
            clipboard_content = self.clipboard_get()
            entry.delete(0, tk.END)
            entry.insert(0, clipboard_content)
            return "break"
        except tk.TclError:
            pass
        return "break"


    def show_context_menu(self, event):
        entry = event.widget
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Вставить", command=lambda: self.paste_handler(event))
        menu.add_command(label="Вырезать", command=lambda: entry.event_generate("<<Cut>>"))
        menu.add_command(label="Копировать", command=lambda: entry.event_generate("<<Copy>>"))
        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()


    def cancel_analysis(self):
        self.cancel_flag = True


    def start_analysis_thread(self):
        self.cancel_flag = False
        thread = threading.Thread(target=self.run_analysis)
        thread.start()


    # Основная функция анализа
    def run_analysis(self):
        try:
            token = self.token_entry.get().strip()
            domain = self.domain_entry.get().strip()
            posts_count = self.posts_entry.get().strip()
            extra_stopwords = self.stop_entry.get().strip()
            skip_pinned = self.skip_pinned_var.get()
            save_excel = self.save_excel_var.get()


            if not posts_count or not posts_count.isdigit():
                self.after(0, lambda: messagebox.showerror("Ошибка", "Введите количество постов (число)!"))
                return


            POSTS_TO_FETCH = int(posts_count)
            if not token or not domain:
                self.after(0, lambda: messagebox.showerror("Ошибка", "Введите токен и домен."))
                return


            extra_stopwords_list = [w.strip().lower() for w in extra_stopwords.split(',') if w.strip()] if extra_stopwords else []


            # Сохраняем конфиг
            config['VK']['token'] = token
            config['VK']['domain_short'] = domain
            with open(CONFIG_FILE, 'w') as f:
                config.write(f)


            vk_config = {
                "token": token,
                "version": "5.126",
                "domain_short": domain,
                "api_root": "https://api.vk.com/method/"
            }


            per_request = 100
            iterations = math.ceil(POSTS_TO_FETCH / per_request)
            all_items = []


            for i in range(iterations):
                if self.cancel_flag:
                    self.after(0, lambda: messagebox.showinfo("Прервано", "Анализ прерван"))
                    return


                offset = i * per_request
                count = min(per_request, POSTS_TO_FETCH - offset)
                params = {
                    "access_token": vk_config["token"],
                    "v": vk_config["version"],
                    "domain": vk_config["domain_short"],
                    "offset": offset,
                    "count": count
                }
                resp = requests.get(vk_config["api_root"] + "wall.get", params=params)
                resp.raise_for_status()
                data = resp.json()
                if not data.get("response") or not data["response"].get("items"):
                    break
                all_items.extend(data["response"]["items"])


                progress_percent = ((i + 1) / iterations) * 100
                self.after(0, lambda p=progress_percent: self.progress_var.set(min(100, p)))
                time.sleep(0.3)


            if not all_items:
                self.after(0, lambda: messagebox.showinfo("Пусто", "Посты не найдены."))
                return


            self.after(0, lambda: ResultWindow(all_items, extra_stopwords_list, skip_pinned, save_excel))


        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Ошибка", str(e)))


# Результаты
class ResultWindow(tk.Toplevel):
    def __init__(self, all_items, extra_stopwords, skip_pinned, save_excel):
        super().__init__()
        self.title("VK Text Analysis")
        self.geometry("1200x700")

        # Применяем фильтр пропуска закрепленного поста
        if skip_pinned and len(all_items) > 0:
            all_items = all_items[1:]

        df = pd.DataFrame({
            "date": [item.get("date") for item in all_items],
            "post_text": [item.get("text", "") for item in all_items]
        })
        df["date"] = pd.to_datetime(df["date"], unit="s", errors="coerce")

        stop_words = set(stopwords.words('russian'))
        extra_tokens = ["'", "''", '""', '"', "<", ">", "?", ")", "(", ".", "!", ",", ":", "-", "%", "$", "^", "@", "[", "]", "{", "}", "/", "\\", "_", "&", "№", "~", "`", "``", '1.', "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "0.", "–", "«", "»"]
        stop_words.update(extra_tokens)
        my_stopwords = ["это", "также", "ещё", "вот", "как", "еще", "быть", "весь", "всех", "всё",
                       "который", "которого","которые", "тот", "того", "этот", "этого", "эти",
                       "этих", "для", "под","при", "над", "про", "через", "между", "самый",
                       "свой", "свои", "своих", "наш","наши", "наших", "ваш", "ваши", "ваших",
                       "мой", "мои", "моих", "твой", "твои","твоих", "любой", "любая", "любое",
                       "любые", "которая", "которой", "которым", "которыми", "однако"]
        stop_words.update(my_stopwords)
        stop_words.update([w.lower() for w in extra_stopwords])

        def preprocessing(text):
            text = str(text).lower()
            text = re.sub(r'http\S+', '', text)
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#\w+', '', text)
            text = re.sub(r'[^\w\sа-яёa-z]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def tokenize_and_filter(text, min_len=2):
            if not text:
                return []
            tokens = word_tokenize(text, language='russian')
            tokens = [t.lower() for t in tokens if t.strip() and len(t) >= min_len]
            tokens = [t for t in tokens if re.fullmatch(r'[а-яёa-z]+', t)]
            tokens = [t for t in tokens if t not in stop_words]
            return tokens

        df["clean_text"] = df["post_text"].apply(preprocessing)
        df["tokens"] = df["clean_text"].apply(tokenize_and_filter)
        df['tokens_str'] = df['tokens'].apply(lambda x: ' '.join(x))

        # ГЛОБАЛЬНЫЙ ФИЛЬТР ТОП-1000 СЛОВ
        all_tokens = [tok for sub in df["tokens"] for tok in sub]
        freq_counter = Counter(all_tokens)
        top_words = [word for word, freq in freq_counter.most_common(1000)]  # ТОП-1000

        # ФИЛЬТРУЕМ tokens только топ-1000 слов
        df["tokens_filtered"] = df["tokens"].apply(lambda tokens: [t for t in tokens if t in top_words])
        df['tokens_str_filtered'] = df['tokens_filtered'].apply(lambda x: ' '.join(x))

        #топ-100 берём после фильтрации
        all_tokens_filtered = [tok for sub in df["tokens_filtered"] for tok in sub]
        freq_counter_filtered = Counter(all_tokens_filtered)
        top100_df = pd.DataFrame(freq_counter_filtered.most_common(100), columns=["word", "frequency"])

        # TF-IDF (только топ-1000)
        tfidf_top_df = pd.DataFrame(columns=["word", "tfidf_score"])
        if len(df['tokens_str_filtered']) > 0:
            vectorizer = TfidfVectorizer(ngram_range=(1,2), vocabulary=top_words)
            X = vectorizer.fit_transform(df['tokens_str_filtered'])
            mean_tfidf = X.mean(axis=0).A1
            terms = vectorizer.get_feature_names_out()
            tfidf_scores = sorted(zip(terms, mean_tfidf), key=lambda x: x[1], reverse=True)[:30]
            tfidf_top_df = pd.DataFrame(tfidf_scores, columns=["word", "tfidf_score"])

        # LDA (только топ-1000)
        lda_top_df = pd.DataFrame(columns=["word", "weight"])
        if len(df["tokens_filtered"]) > 0 and sum(len(t) for t in df["tokens_filtered"]) > 0:
            dictionary = corpora.Dictionary(df["tokens_filtered"])
            corpus = [dictionary.doc2bow(text) for text in df["tokens_filtered"]]
            num_topics = min(5, max(1, len(df)//1))
            lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42, passes=10)
            agg = defaultdict(float)
            for t in range(lda.num_topics):
                for word, weight in lda.show_topic(t, topn=50):
                    agg[word] += weight
            lda_top_df = pd.DataFrame(sorted(agg.items(), key=lambda x: x[1], reverse=True)[:10],
                                    columns=["word", "weight"])


        # Частота слов по дате (ТОЛЬКО ТОП-1000)
        freq_data = df[["date","tokens_filtered"]].copy()
        freq_data['date_only'] = freq_data['date'].dt.date
        all_tokens_unique = sorted(set(all_tokens_filtered))


        tokens_counts_df = pd.DataFrame(
            [Counter(t) for t in freq_data['tokens_filtered']],
            columns=all_tokens_unique
        ).fillna(0).astype(int)


        freq_data = pd.concat([freq_data, tokens_counts_df], axis=1)
        freq_over_time = freq_data.groupby('date_only')[all_tokens_unique].sum().reset_index()
        freq_over_time.rename(columns={"date_only":"date"}, inplace=True)


        # Excel (если чекбокс активен)
        if save_excel:
            output_filename = f"vk_text_analysis_{int(time.time())}.xlsx"
            with pd.ExcelWriter(output_filename) as writer:
                df[["date","post_text","clean_text","tokens","tokens_filtered"]].to_excel(writer, sheet_name="posts", index=False)
                top100_df.to_excel(writer, sheet_name="top_words_freq", index=False)
                tfidf_top_df.to_excel(writer, sheet_name="tfidf_top", index=False)
                lda_top_df.to_excel(writer, sheet_name="lda_top", index=False)

                if not freq_over_time.empty:
                    ft = freq_over_time.set_index('date').T
                    ft['TOTAL'] = ft.sum(axis=1)
                    date_cols = [c for c in ft.columns if c != 'TOTAL']
                    ft_reset = ft.reset_index().rename(columns={'index': 'word'})
                    ordered_cols = ['word', 'TOTAL'] + date_cols
                    ft_reset = ft_reset[ordered_cols]
                    ft_reset.to_excel(writer, sheet_name="freq_over_time", index=False)
                else:
                    freq_over_time.to_excel(writer, sheet_name="freq_over_time", index=False)

            print(f"✅ Excel сохранён: {output_filename} (топ-1000 слов)")


        notebook = ttk.Notebook(self)
        notebook.pack(fill='both', expand=True)


        # Облако слов (ТОЛЬКО ТОП-1000)
        tab1 = ttk.Frame(notebook)
        notebook.add(tab1, text="Облако слов (топ-1000)")
        text_for_wc = " ".join(all_tokens)
        fig1, ax1 = plt.subplots(figsize=(10,8))
        wc = WordCloud(width=800, height=600, max_words=200, collocations=False, background_color="white").generate(text_for_wc)
        ax1.imshow(wc, interpolation="bilinear")
        ax1.axis('off')
        canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
        canvas1.get_tk_widget().pack(fill='both', expand=True)


        # TF-IDF
        tab2 = ttk.Frame(notebook)
        notebook.add(tab2, text="TF-IDF топ-30")
        fig2, ax2 = plt.subplots(figsize=(10,6))
        ax2.barh(tfidf_top_df['word'][::-1], tfidf_top_df['tfidf_score'][::-1])
        ax2.set_title("TF-IDF топ-30")
        canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
        canvas2.get_tk_widget().pack(fill='both', expand=True)


        # LDA
        tab3 = ttk.Frame(notebook)
        notebook.add(tab3, text="LDA топ-10")
        fig3, ax3 = plt.subplots(figsize=(10,6))
        ax3.barh(lda_top_df['word'][::-1], lda_top_df['weight'][::-1])
        ax3.set_title("LDA топ-10")
        canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
        canvas3.get_tk_widget().pack(fill='both', expand=True)


        # Частота слов по времени (ТОЛЬКО ТОП-1000)
        tab4 = ttk.Frame(notebook)
        notebook.add(tab4, text="Частота слов по времени (топ-1000)")
        fig4, ax4 = plt.subplots(figsize=(10,6))
        canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
        canvas4.get_tk_widget().pack(side="right", fill='both', expand=True)


        # Поле ввода слов для графика
        input_frame = tk.Frame(tab4)
        input_frame.pack(side="left", fill="y", padx=5, pady=5)
        tk.Label(input_frame, text="Введите слова через запятую (из топ-1000):").pack(anchor="w")
        words_entry = tk.Entry(input_frame, width=30)
        words_entry.pack(anchor="w")


        def update_plot():
            ax4.clear()
            input_words = [w.strip() for w in words_entry.get().split(',') if w.strip()]
            valid_words = [w for w in input_words if w in freq_over_time.columns]
            for w in valid_words:
                ax4.plot(freq_over_time['date'], freq_over_time[w], label=w, marker='o')
            ax4.set_title("Динамика слов по дате (топ-1000)")
            ax4.set_xlabel("Дата")
            ax4.set_ylabel("Количество в день")
            ax4.xaxis.set_major_formatter(DateFormatter("%d-%m-%Y"))
            fig4.autofmt_xdate()
            if valid_words:
                ax4.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            canvas4.draw()


        tk.Button(input_frame, text="Построить график", command=update_plot).pack(anchor="w", pady=5)


if __name__ == "__main__":
    app = InputWindow()
    app.mainloop()