import os.path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

books = pd.read_csv('books_with_emotions.csv')
books['large_thumbnail'] = books['thumbnail'] + "&file=w800"
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    "cover-not-found.jpg",
    books['large_thumbnail'],
)

persist_directory = "chroma_db"

if (os.path.exists(persist_directory) and os.listdir(persist_directory)):
    db_books = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings())
    print("Loaded Chroma database from disk.")
else:
    raw_documents = TextLoader('tagged_description.txt').load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=0, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    # Create a new database with persistence
    db_books = Chroma.from_documents(
        documents,
        OpenAIEmbeddings(),
        persist_directory=persist_directory
    )

    # Persist the database to disk
    print("Created and saved new vector database to disk.")

def retrieve_semantic_recommendation(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_res = books[books['isbn13'].isin(books_list)].head(final_top_k)

    if category != 'All':
        book_res = book_res[book_res['simple_categories'] == category].head(final_top_k)
    else:
        book_res = book_res.head(final_top_k)

    if tone == 'Happy':
        book_res.sort_values('joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_res.sort_values('surprise', ascending=False, inplace=True)
    elif tone == 'Angry':
        book_res.sort_values('anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        book_res.sort_values('fear', ascending=False, inplace=True)
    if tone == 'Sad':
        book_res.sort_values('sadness', ascending=False, inplace=True)

    return book_res


def recommend_books(
        query: str,
        category: str,
        tone: str
):

    recommendations = retrieve_semantic_recommendation(query, category, tone)
    result = []

    for _, row in recommendations.iterrows():
        description = row['description']
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        result.append((row['large_thumbnail'], caption))
    return result


categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ['All'] + ['Happy', 'Surprise', 'Angry', 'Suspenseful', 'Sad']

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic Book Recommender')

    with gr.Row():
        user_query = gr.Textbox(label='Enter a book description', placeholder='e.g., A story about forgiveness')
        category_dropdown = gr.Dropdown(categories, label='Select a category', value='All')
        tone_dropdown = gr.Dropdown(tones, label='Select an emotional tone', value='All')
        submit_button = gr.Button('Find recommendations')

    gr.Markdown('## Recommendations')
    output = gr.Gallery(label='Recommended books', columns=8, rows=2)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )


if __name__ == '__main__':
    dashboard.launch()