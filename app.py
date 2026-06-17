
import gradio as gr
import tempfile
import os

from rag_engine import (
    ingest_pdf,
    build_faiss_index,
    answer_question
)

# Global storage
INDEX = None
CHUNKS = None


def process_pdf(pdf_file):
    """
    Runs when user uploads a PDF.
    Creates chunks + FAISS index.
    """

    global INDEX, CHUNKS

    if pdf_file is None:
        return "Please upload a PDF first."

    try:
        chunks = ingest_pdf(pdf_file.name)

        index, stored_chunks = build_faiss_index(chunks)

        INDEX = index
        CHUNKS = stored_chunks

        return f"""
✅ Document indexed successfully

Pages indexed: {len(set(c['page'] for c in stored_chunks))}
Text chunks created: {len(stored_chunks)}

You can now ask questions.
"""

    except Exception as e:
        return f"Error: {str(e)}"


def chat_with_document(message, history):
    """
    Called whenever user sends a question.
    """

    global INDEX, CHUNKS

    if INDEX is None:
        return "Please upload and process a PDF first."

    try:

        result = answer_question(
            question=message,
            index=INDEX,
            chunks=CHUNKS
        )

        pages = sorted(result["sources"])

        response = f"""
{result['answer']}

📚 Source Pages: {pages}
"""

        return response

    except Exception as e:
        return f"Error: {str(e)}"


with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Medical Document Assistant"
) as app:

    gr.Markdown(
        """
# 🏥 Medical Document Q&A Assistant

Upload a medical PDF and ask questions about it.

⚠️ For educational and research purposes only.
Not a substitute for professional medical advice.
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"]
            )

            process_btn = gr.Button(
                "📄 Process Document",
                variant="primary"
            )

            status_output = gr.Markdown()

        with gr.Column(scale=2):

            chatbot = gr.ChatInterface(
                fn=chat_with_document,
                title="Ask Questions",
                description="Ask anything about the uploaded PDF."
            )

    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status_output
    )


if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
