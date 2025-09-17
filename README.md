# AI Agent for ElevenLabs

This is a custom LLM agent designed to work with ElevenLabs Agents Platform using Together AI as the backend.

## Setup

### 1. Get Together AI API Key

1. Go to [api.together.xyz/settings/api-keys](https://api.together.xyz/settings/api-keys)
2. Create a new API key
3. Copy the API key

### 2. Configure Environment

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your Together AI API key:
   ```
   TOGETHER_API_KEY=your_actual_api_key_here
   ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Configure ElevenLabs Agent

In your ElevenLabs Agent dashboard:

1. Go to your agent settings
2. Navigate to "Secrets" section
3. Add a new secret with your Together AI API key
4. In the LLM section, select "Custom LLM"
5. Set:
   - **Server URL**: `https://your-domain.com/v1` (or `http://localhost:8000/v1` for testing)
   - **Model ID**: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`
   - **API Key**: Select your Together AI API key from the dropdown

## Testing

Test your API endpoint:

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_together_api_key" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "messages": [
      {
        "role": "user",
        "content": "Hello, I want to rent a car"
      }
    ]
  }'
```

## Troubleshooting

- If you see "custom_llm_error: Failed to generate response from custom LLM", check:
  1. Your Together AI API key is correct
  2. The server is running and accessible
  3. The model name matches exactly
  4. Check the server logs for detailed error messages

## Models Supported

The following Together AI models are recommended for ElevenLabs Agents:

- `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` (faster, cheaper)
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` (recommended)
- `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` (most capable)
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` (latest)
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `mistralai/Mistral-7B-Instruct-v0.1`
