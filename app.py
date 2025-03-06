from transformers import pipeline
import gradio as gr

# Load the pre-trained sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"Sentiment: {result['label']}, Confidence: {result['score']:.2f}"

# Gradio interface
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter your text"),
    outputs=gr.Textbox(label="Sentiment Analysis Result"),
    title="Sentiment Analysis API",
    description="This API analyzes sentiment using a Hugging Face model."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
