import google.generativeai as genai
import os
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Initialize Flask app
app = Flask(__name__)

# Initialize Gemini model with streaming config
model = genai.GenerativeModel('gemini-1.5-flash')

# System instructions for formatted output
SYSTEM_PROMPT = """Please format your responses according to these rules:
1. Use markdown formatting
2. For code blocks, specify the language
3. For lists, use proper markdown bullet points or numbers
4. For important information, use bold text
5. For technical terms, use inline code formatting
6. Include section headers where appropriate
"""

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')
        
        # Combine system prompt with user message
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_message}"
        
        # Generate streaming response from Gemini
        response = model.generate_content(
            full_prompt,
            stream=True
        )

        def generate():
            for chunk in response:
                if chunk.text:
                    yield f"data: {chunk.text}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype='text/event-stream'
        )

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
