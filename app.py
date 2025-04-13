from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx
import json
import logging
import sys
import os
import re
import time
import asyncio
from typing import Dict, Any
import sqlite3
from pydantic import BaseModel

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Default configuration values
DEFAULT_TIMEOUT = 300.0
DATABASE_URL = os.getenv('DATABASE_URL', '/data/keywords.db')

# Ensure data directory exists
os.makedirs(os.path.dirname(DATABASE_URL), exist_ok=True)

# Initialize database
def init_db():
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    
    # Keywords table
    c.execute('''CREATE TABLE IF NOT EXISTS keywords
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  keyword TEXT NOT NULL,
                  webhook_id TEXT NOT NULL,
                  backend_type TEXT NOT NULL DEFAULT 'n8n',
                  description TEXT)''')
    
    # Configuration table
    c.execute('''CREATE TABLE IF NOT EXISTS config
                 (id TEXT PRIMARY KEY,
                  value TEXT NOT NULL,
                  description TEXT)''')
    
    # Insert default configuration if not exists
    c.execute('SELECT COUNT(*) FROM config')
    if c.fetchone()[0] == 0:
        default_configs = [
            ('llm_backend', 'llama.cpp', 'Main LLM backend type (llama.cpp or ollama)'),
            ('webhook_backend', 'n8n', 'Webhook backend type (n8n or flowise)'),
            ('llama_url', 'http://192.168.1.166:8988', 'URL for llama.cpp server'),
            ('ollama_url', 'http://192.168.1.166:11434', 'URL for Ollama server'),
            ('ollama_model', 'llama2', 'Model to use with Ollama'),
            ('n8n_url', 'http://192.168.1.166:5678', 'URL for n8n server'),
            ('n8n_use_test_webhook', 'false', 'Use test webhook for n8n automation testing'),
            ('flowise_url', 'http://192.168.1.166:3000', 'URL for Flowise server'),
            ('language', 'en', 'Default interface language'),
            ('temperature', '0.7', 'Generation temperature'),
            ('max_tokens', '400', 'Maximum tokens to generate'),
            ('timeout', '60', 'Request timeout in seconds'),
            ('debug_mode', 'false', 'Enable debug mode')
        ]
        c.executemany('INSERT INTO config (id, value, description) VALUES (?, ?, ?)', default_configs)
    else:
        # Ensure n8n_use_test_webhook exists even if other configs already exist
        c.execute('SELECT COUNT(*) FROM config WHERE id=?', ('n8n_use_test_webhook',))
        if c.fetchone()[0] == 0:
            c.execute('INSERT INTO config (id, value, description) VALUES (?, ?, ?)',
                     ('n8n_use_test_webhook', 'false', 'Use test webhook for n8n automation testing'))
    
    conn.commit()
    conn.close()

init_db()

# Helper function to get configuration value
def get_config_value(config_id: str, default: str = "") -> str:
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('SELECT value FROM config WHERE id=?', (config_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else default

# Database models
class KeywordRule(BaseModel):
    keyword: str
    webhook_id: str
    backend_type: str = "n8n"
    description: str = ""

class ConfigItem(BaseModel):
    id: str
    value: str
    description: str = ""

# Keyword CRUD endpoints
@app.post("/api/keywords")
async def create_keyword(rule: KeywordRule):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('INSERT INTO keywords (keyword, webhook_id, backend_type, description) VALUES (?, ?, ?, ?)',
              (rule.keyword, rule.webhook_id, rule.backend_type, rule.description))
    conn.commit()
    id = c.lastrowid
    conn.close()
    return {"id": id, **rule.dict()}

@app.get("/api/keywords")
async def get_keywords():
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('SELECT id, keyword, webhook_id, backend_type, description FROM keywords')
    keywords = [{"id": row[0], "keyword": row[1], "webhook_id": row[2], "backend_type": row[3], "description": row[4]} 
                for row in c.fetchall()]
    conn.close()
    return keywords

@app.put("/api/keywords/{keyword_id}")
async def update_keyword(keyword_id: int, rule: KeywordRule):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('UPDATE keywords SET keyword=?, webhook_id=?, backend_type=?, description=? WHERE id=?',
              (rule.keyword, rule.webhook_id, rule.backend_type, rule.description, keyword_id))
    conn.commit()
    conn.close()
    return {"id": keyword_id, **rule.dict()}

@app.delete("/api/keywords/{keyword_id}")
async def delete_keyword(keyword_id: int):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('DELETE FROM keywords WHERE id=?', (keyword_id,))
    conn.commit()
    conn.close()
    return {"status": "success"}

# Configuration endpoints
@app.get("/api/config")
async def get_config():
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('SELECT id, value, description FROM config')
    configs = [{"id": row[0], "value": row[1], "description": row[2]} 
               for row in c.fetchall()]
    conn.close()
    return configs

@app.put("/api/config/{config_id}")
async def update_config(config_id: str, item: ConfigItem):
    conn = sqlite3.connect(DATABASE_URL)
    c = conn.cursor()
    c.execute('UPDATE config SET value=?, description=? WHERE id=?',
              (item.value, item.description, config_id))
    conn.commit()
    conn.close()
    return {"id": config_id, **item.dict()}

# New Ollama compatibility endpoints

@app.get("/v1/models")
async def openai_models():
    """
    OpenAI API compatibility endpoint for listing available models.
    This endpoint is used by clients like Chatbox that expect OpenAI API compatibility.
    """
    print("Received OpenAI API models request", flush=True)
    try:
        # Get current configuration
        llm_backend = get_config_value("llm_backend", "llama.cpp")
        llama_url = get_config_value("llama_url", "http://192.168.1.166:8988")
        ollama_url = get_config_value("ollama_url", "http://192.168.1.166:11434")
        ollama_model = get_config_value("ollama_model", "llama2")
        timeout = float(get_config_value("timeout", "60"))
        
        # Set timeout
        TIMEOUT_CONFIG = httpx.Timeout(timeout=timeout)
        
        if llm_backend == "llama.cpp":
            # Forward request to llama.cpp
            async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                try:
                    response = await client.get(f"{llama_url}/v1/models")
                    if response.status_code == 200:
                        return JSONResponse(content=response.json())
                    else:
                        # Fallback if llama.cpp doesn't respond properly
                        return JSONResponse(content={
                            "object": "list",
                            "data": [
                                {
                                    "id": "llama2",
                                    "object": "model",
                                    "created": int(time.time()),
                                    "owned_by": "llama.cpp"
                                }
                            ]
                        })
                except Exception as e:
                    print(f"Error connecting to llama.cpp: {str(e)}", flush=True)
                    # Fallback model list
                    return JSONResponse(content={
                        "object": "list",
                        "data": [
                            {
                                "id": "llama2",
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "llama.cpp"
                            }
                        ]
                    })
                    
        elif llm_backend == "ollama":
            # Get models from Ollama and convert to OpenAI format
            async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                try:
                    response = await client.get(f"{ollama_url}/api/tags")
                    if response.status_code == 200:
                        ollama_models = response.json().get("models", [])
                        
                        # Convert to OpenAI format
                        openai_models = {
                            "object": "list",
                            "data": [
                                {
                                    "id": model.get("name", "unknown"),
                                    "object": "model",
                                    "created": int(time.time()),
                                    "owned_by": "ollama"
                                }
                                for model in ollama_models
                            ]
                        }
                        
                        return JSONResponse(content=openai_models)
                    else:
                        # Fallback if Ollama doesn't respond properly
                        return JSONResponse(content={
                            "object": "list",
                            "data": [
                                {
                                    "id": ollama_model,
                                    "object": "model",
                                    "created": int(time.time()),
                                    "owned_by": "ollama"
                                }
                            ]
                        })
                except Exception as e:
                    print(f"Error connecting to Ollama: {str(e)}", flush=True)
                    # Fallback model list
                    return JSONResponse(content={
                        "object": "list",
                        "data": [
                            {
                                "id": ollama_model,
                                "object": "model",
                                "created": int(time.time()),
                                "owned_by": "ollama"
                            }
                        ]
                    })
                    
        else:
            # Unknown backend, return a generic model list
            return JSONResponse(content={
                "object": "list",
                "data": [
                    {
                        "id": "generic-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "llm-proxy"
                    }
                ]
            })
    
    except Exception as e:
        print(f"Models API error: {str(e)}", flush=True)
        logger.error("Models API error details:", exc_info=True)
        # Return a minimal response even on error
        return JSONResponse(
            content={
                "object": "list",
                "data": [
                    {
                        "id": "fallback-model",
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "llm-proxy"
                    }
                ]
            },
            status_code=200  # Returning 200 even on error to not break clients
        )

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """
    OpenAI API compatibility endpoint for chat completions.
    This endpoint is used by clients like OpenWebUI that expect OpenAI API compatibility.
    The endpoint will route requests to either llama.cpp or Ollama depending on the configuration.
    Also checks for keywords to route to webhooks if applicable.
    """
    print("Received OpenAI API chat completions request", flush=True)
    try:
        data = await request.json()
        
        messages = data.get("messages", [])
        model = data.get("model", "")
        stream = data.get("stream", False)
        temperature = data.get("temperature", None)
        max_tokens = data.get("max_tokens", None)
        
        # Get current configuration
        llm_backend = get_config_value("llm_backend", "llama.cpp")
        webhook_backend = get_config_value("webhook_backend", "n8n")
        
        llama_url = get_config_value("llama_url", "http://192.168.1.166:8988")
        ollama_url = get_config_value("ollama_url", "http://192.168.1.166:11434")
        ollama_model = get_config_value("ollama_model", "llama2")
        n8n_url = get_config_value("n8n_url", "http://192.168.1.166:5678")
        flowise_url = get_config_value("flowise_url", "http://192.168.1.166:3000")
        
        config_temperature = float(get_config_value("temperature", "0.7"))
        config_max_tokens = int(get_config_value("max_tokens", "400"))
        timeout = float(get_config_value("timeout", "60"))
        
        # Use provided values or fall back to configuration
        if temperature is None:
            temperature = config_temperature
        if max_tokens is None:
            max_tokens = config_max_tokens
        
        # Set timeout
        TIMEOUT_CONFIG = httpx.Timeout(timeout=timeout)
        
        # Extract the last user message to check for keywords
        last_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_message = msg.get("content", "").lower()
                break
                
        # Check for keywords in database
        conn = sqlite3.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute('SELECT keyword, webhook_id, backend_type FROM keywords')
        keywords = c.fetchall()
        conn.close()
        
        for keyword, webhook_id, backend_type in keywords:
            if keyword.lower() in last_message:
                # Found a keyword match, process with webhook
                print(f"Found keyword '{keyword}' in OpenAI API request, routing to webhook", flush=True)
                
                # Extract only last message and clean it
                clean_prompt = re.sub(r'\[.*?\]', '', last_message).strip()
                print(f"Cleaned prompt: {clean_prompt}", flush=True)
                
                # Use specific backend for this keyword
                current_backend = backend_type if backend_type else webhook_backend
                
                try:
                    response_text = ""
                    
                    if current_backend == "n8n":
                        # Get test webhook setting
                        use_test_webhook = get_config_value("n8n_use_test_webhook", "false").lower() == "true"
                        
                        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                            params = {"question": clean_prompt}
                            
                            # Build appropriate URL based on setting
                            webhook_url = f"{n8n_url}/webhook-test/{webhook_id}" if use_test_webhook else f"{n8n_url}/webhook/{webhook_id}"
                            
                            webhook_response = await client.get(
                                webhook_url,
                                params=params
                            )
                            try:
                                response_json = webhook_response.json()
                                response_text = response_json.get("text", str(response_json))
                            except:
                                response_text = webhook_response.text
                    
                    elif current_backend == "flowise":
                        # Flowise implementation
                        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                            flowise_response = await client.post(
                                f"{flowise_url}/api/v1/prediction/{webhook_id}",
                                json={"question": clean_prompt}
                            )
                            response_text = flowise_response.json().get("text", "No response")
                    
                    # Return the webhook response in OpenAI format
                    if stream:
                        # Return streaming response
                        async def generate_webhook_stream():
                            # For streaming we'll split the response into smaller chunks
                            chunk_size = 15  # Number of characters per chunk
                            chunks = [response_text[i:i+chunk_size] for i in range(0, len(response_text), chunk_size)]
                            
                            for i, chunk in enumerate(chunks):
                                is_last = i == len(chunks) - 1
                                
                                openai_chunk = {
                                    "id": f"chatcmpl-{os.urandom(6).hex()}",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": "webhook",
                                    "choices": [
                                        {
                                            "delta": {"content": chunk},
                                            "index": 0,
                                            "finish_reason": "stop" if is_last else None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                
                                # Small delay to simulate realistic streaming
                                await asyncio.sleep(0.05)
                                
                            yield "data: [DONE]\n\n"
                            
                        return StreamingResponse(
                            generate_webhook_stream(),
                            media_type="text/event-stream"
                        )
                    else:
                        # Return non-streaming response
                        openai_response = {
                            "id": f"chatcmpl-{os.urandom(6).hex()}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": "webhook",
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": response_text
                                    },
                                    "finish_reason": "stop"
                                }
                            ],
                            "usage": {
                                "prompt_tokens": -1,
                                "completion_tokens": -1,
                                "total_tokens": -1
                            }
                        }
                        
                        return JSONResponse(content=openai_response)
                        
                except Exception as e:
                    print(f"Webhook error in OpenAI endpoint: {str(e)}", flush=True)
                    logger.error("Webhook error details:", exc_info=True)
                    # Continue with normal LLM processing if webhook fails
        
        # Choose the model name based on the provided model or configured model
        model_name = model if model else ollama_model
        
        if llm_backend == "llama.cpp":
            # Forward to llama.cpp
            llm_request = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }
            
            print(f"Forwarding to llama.cpp: {llm_request}", flush=True)
            
            headers = {
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                response = await client.post(
                    f"{llama_url}/v1/chat/completions",
                    json=llm_request,
                    headers=headers
                )
                
                # Return the proxied response as is
                if stream:
                    # Return streaming response
                    return StreamingResponse(
                        response.aiter_bytes(),
                        status_code=response.status_code,
                        headers={k: v for k, v in response.headers.items()},
                        media_type=response.headers.get("content-type", "text/event-stream")
                    )
                else:
                    # Return non-streaming response as JSON
                    return JSONResponse(
                        content=response.json(),
                        status_code=response.status_code
                    )
                    
        elif llm_backend == "ollama":
            # Forward to Ollama with appropriate format
            if stream:
                ollama_request = {
                    "model": model_name,
                    "messages": messages,
                    "stream": True,
                    "options": {
                        "temperature": temperature
                    }
                }
                
                print(f"Forwarding to Ollama streaming: {ollama_request}", flush=True)
                
                async def generate_openai_from_ollama():
                    async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                        try:
                            async with client.stream("POST", f"{ollama_url}/api/chat", json=ollama_request) as response:
                                async for line in response.aiter_lines():
                                    if line.strip():
                                        try:
                                            ollama_chunk = json.loads(line)
                                            # Convert Ollama format to OpenAI format
                                            content = ollama_chunk.get("message", {}).get("content", "")
                                            if content:
                                                openai_chunk = {
                                                    "id": "chatcmpl-" + os.urandom(6).hex(),
                                                    "object": "chat.completion.chunk",
                                                    "created": int(time.time()),
                                                    "model": model_name,
                                                    "choices": [
                                                        {
                                                            "delta": {"content": content},
                                                            "index": 0,
                                                            "finish_reason": "stop" if ollama_chunk.get("done", False) else None
                                                        }
                                                    ]
                                                }
                                                yield f"data: {json.dumps(openai_chunk)}\n\n"
                                            
                                            if ollama_chunk.get("done", False):
                                                yield "data: [DONE]\n\n"
                                        except json.JSONDecodeError:
                                            continue
                        except Exception as e:
                            print(f"Ollama streaming error: {str(e)}", flush=True)
                            error_chunk = {
                                "id": "chatcmpl-" + os.urandom(6).hex(),
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [
                                    {
                                        "delta": {"content": f"Error: {str(e)}"},
                                        "index": 0,
                                        "finish_reason": "stop"
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(error_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate_openai_from_ollama(),
                    media_type="text/event-stream"
                )
            else:
                # Non-streaming request to Ollama
                ollama_request = {
                    "model": model_name,
                    "messages": messages,
                    "options": {
                        "temperature": temperature
                    }
                }
                
                print(f"Forwarding to Ollama non-streaming: {ollama_request}", flush=True)
                
                async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                    response = await client.post(f"{ollama_url}/api/chat", json=ollama_request)
                    ollama_response = response.json()
                    
                    # Convert Ollama response to OpenAI format
                    openai_response = {
                        "id": "chatcmpl-" + os.urandom(6).hex(),
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": ollama_response.get("message", {}).get("content", "")
                                },
                                "finish_reason": "stop"
                            }
                        ],
                        "usage": {
                            "prompt_tokens": -1,
                            "completion_tokens": -1,
                            "total_tokens": -1
                        }
                    }
                    
                    return JSONResponse(content=openai_response)
        
        else:
            # Unknown backend
            return JSONResponse(
                content={"error": f"Unsupported LLM backend: {llm_backend}"},
                status_code=400
            )
            
    except Exception as e:
        print(f"OpenAI API error: {str(e)}", flush=True)
        logger.error("OpenAI API error details:", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/api/tags")
async def get_ollama_tags():
    """
    Get available models from Ollama when Ollama is the selected backend.
    This endpoint mimics Ollama's /api/tags endpoint for compatibility with clients like OpenWebUI.
    """
    # Check if Ollama is the selected backend
    llm_backend = get_config_value("llm_backend", "llama.cpp")
    
    if llm_backend != "ollama":
        # If not using Ollama, return a default response with just the default model
        ollama_model = get_config_value("ollama_model", "llama2")
        return {"models": [{"name": ollama_model, "modified_at": "2023-01-01T00:00:00Z", "size": 0}]}
    
    # Get Ollama URL from config
    ollama_url = get_config_value("ollama_url", "http://192.168.1.166:11434")
    timeout = float(get_config_value("timeout", "60"))
    
    # Set timeout
    TIMEOUT_CONFIG = httpx.Timeout(timeout=timeout)
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
            # Forward the request to the actual Ollama server
            response = await client.get(f"{ollama_url}/api/tags")
            
            if response.status_code == 200:
                # Return the models list from Ollama
                return response.json()
            else:
                # Return empty list if request fails
                return {"models": []}
    except Exception as e:
        print(f"Error fetching Ollama models: {str(e)}", flush=True)
        logger.error("Error fetching Ollama models:", exc_info=True)
        return {"models": []}

# Main chat endpoint for handling messages
@app.post("/api/chat")
async def chat(request: Request):
    print("Received chat request", flush=True)
    try:
        data = await request.json()
        messages = data.get("messages", [])
        last_message = messages[-1].get("content", "").lower() if messages else ""
        
        # Get current configuration
        llm_backend = get_config_value("llm_backend", "llama.cpp")
        webhook_backend = get_config_value("webhook_backend", "n8n")
        
        llama_url = get_config_value("llama_url", "http://192.168.1.166:8988")
        ollama_url = get_config_value("ollama_url", "http://192.168.1.166:11434")
        ollama_model = get_config_value("ollama_model", "llama2")
        n8n_url = get_config_value("n8n_url", "http://192.168.1.166:5678")
        flowise_url = get_config_value("flowise_url", "http://192.168.1.166:3000")
        timeout = float(get_config_value("timeout", "60"))
        temperature = float(get_config_value("temperature", "0.7"))
        max_tokens = int(get_config_value("max_tokens", "400"))
        
        # Set timeout
        TIMEOUT_CONFIG = httpx.Timeout(timeout=timeout)
        
        # Check for keywords in database
        conn = sqlite3.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute('SELECT keyword, webhook_id, backend_type FROM keywords')
        keywords = c.fetchall()
        conn.close()

        for keyword, webhook_id, backend_type in keywords:
            if keyword.lower() in last_message:
                # Extract only last message
                clean_prompt = re.sub(r'\[.*?\]', '', last_message).strip()
                print(f"Cleaned prompt: {clean_prompt}", flush=True)
                
                # Use specific backend for this keyword
                current_backend = backend_type if backend_type else webhook_backend
                
                try:
                    if current_backend == "n8n":
                        # Get test webhook setting
                        use_test_webhook = get_config_value("n8n_use_test_webhook", "false").lower() == "true"
                        
                        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                            params = {"question": clean_prompt}
                            
                            # Zbuduj odpowiedni URL w zależności od ustawienia, ale zachowaj ID
                            webhook_url = f"{n8n_url}/webhook-test/{webhook_id}" if use_test_webhook else f"{n8n_url}/webhook/{webhook_id}"
                            
                            webhook_response = await client.get(
                                webhook_url,
                                params=params
                            )
                            try:
                                response_json = webhook_response.json()
                                response_text = response_json.get("text", str(response_json))
                            except:
                                response_text = webhook_response.text
                    
                    elif current_backend == "flowise":
                        # Flowise implementation remains unchanged
                        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                            flowise_response = await client.post(
                                f"{flowise_url}/api/v1/prediction/{webhook_id}",
                                json={"question": clean_prompt}
                            )
                            response_text = flowise_response.json().get("text", "No response")
                    
                    return StreamingResponse(
                        iter([json.dumps({
                            "model": current_backend,
                            "message": {
                                "role": "assistant",
                                "content": response_text
                            },
                            "done": True
                        }) + "\n"]),
                        media_type="application/x-ndjson"
                    )
                    
                except Exception as e:
                    print(f"Webhook error: {str(e)}", flush=True)
                    logger.error("Webhook error details:", exc_info=True)

        # No keyword match, use LLM backend
        if llm_backend == "llama.cpp":
            llm_request = {
                "model": "llama2",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            print(f"Sending to llama.cpp: {llm_request}", flush=True)
            
            async def generate_llama():
                async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    }
                    try:
                        async with client.stream("POST", f"{llama_url}/v1/chat/completions", 
                                               json=llm_request, headers=headers) as response:
                            buffer = ""
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    line = line[6:]  # Remove "data: " prefix
                                    if line.strip() == "[DONE]":
                                        continue
                                    try:
                                        completion_chunk = json.loads(line)
                                        if "choices" in completion_chunk:
                                            content = completion_chunk["choices"][0].get("delta", {}).get("content", "")
                                            if content:
                                                yield json.dumps({
                                                    "model": "llama",
                                                    "message": {
                                                        "role": "assistant",
                                                        "content": content
                                                    },
                                                    "done": completion_chunk["choices"][0].get("finish_reason") is not None
                                                }) + "\n"
                                    except json.JSONDecodeError as e:
                                        print(f"JSON decode error: {line}", flush=True)
                                        continue
                    except Exception as e:
                        print(f"Stream error: {str(e)}", flush=True)
                        yield json.dumps({
                            "model": "llama",
                            "message": {
                                "role": "assistant",
                                "content": f"Sorry, there was an error communicating with the LLM service: {str(e)}"
                            },
                            "done": True
                        }) + "\n"
            
            return StreamingResponse(
                generate_llama(),
                media_type="application/x-ndjson"
            )
            
        elif llm_backend == "ollama":
            # Ollama request format
            ollama_request = {
                "model": ollama_model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }
            
            print(f"Sending to Ollama: {ollama_request}", flush=True)
            
            async def generate_ollama():
                async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                    try:
                        async with client.stream("POST", f"{ollama_url}/api/chat", json=ollama_request) as response:
                            async for line in response.aiter_lines():
                                if line.strip():
                                    yield line + "\n"
                    except Exception as e:
                        print(f"Ollama error: {str(e)}", flush=True)
                        yield json.dumps({
                            "model": "ollama",
                            "message": {
                                "role": "assistant",
                                "content": f"Sorry, there was an error communicating with Ollama: {str(e)}"
                            },
                            "done": True
                        }) + "\n"
            
            return StreamingResponse(
                generate_ollama(),
                media_type="application/x-ndjson"
            )
            
    except Exception as e:
        print(f"MAIN ERROR: {str(e)}", flush=True)
        logger.error("Error details:", exc_info=True)
        return StreamingResponse(
            iter([json.dumps({
                "error": str(e),
                "message": {
                    "role": "assistant",
                    "content": f"Sorry, an error occurred: {str(e)}"
                },
                "done": True
            }) + "\n"]),
            media_type="application/x-ndjson"
        )

# Compatibility endpoint for Amica
@app.post("/completion")
async def completion(request: Request):
    print("Received completion request", flush=True)
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        
        # Extract latest user message
        last_prompt = prompt.split("User:")[-1].strip() if "User:" in prompt else prompt
        
        # Remove emotion tags
        clean_prompt = re.sub(r'\[.*?\]', '', last_prompt).strip()
        print(f"Cleaned prompt: {clean_prompt}", flush=True)
        
        # Get configuration
        llm_backend = get_config_value("llm_backend", "llama.cpp")
        webhook_backend = get_config_value("webhook_backend", "n8n")
        
        llama_url = get_config_value("llama_url", "http://192.168.1.166:8988")
        ollama_url = get_config_value("ollama_url", "http://192.168.1.166:11434")
        ollama_model = get_config_value("ollama_model", "llama2")
        n8n_url = get_config_value("n8n_url", "http://192.168.1.166:5678")
        flowise_url = get_config_value("flowise_url", "http://192.168.1.166:3000")
        timeout = float(get_config_value("timeout", "60"))
        temperature = float(get_config_value("temperature", "0.7"))
        max_tokens = int(get_config_value("max_tokens", "400"))
        
        # Set timeout
        TIMEOUT_CONFIG = httpx.Timeout(timeout=timeout)
        
        # Check for keywords
        conn = sqlite3.connect(DATABASE_URL)
        c = conn.cursor()
        c.execute('SELECT keyword, webhook_id, backend_type FROM keywords')
        keywords = c.fetchall()
        conn.close()

        for keyword, webhook_id, backend_type in keywords:
            if keyword.lower() in clean_prompt.lower():
                try:
                    current_backend = backend_type if backend_type else webhook_backend
                    
                    if current_backend == "n8n":
                        # Get test webhook setting
                        use_test_webhook = get_config_value("n8n_use_test_webhook", "false").lower() == "true"
                        
                        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                            params = {"question": clean_prompt}
                            
                            # Zbuduj odpowiedni URL w zależności od ustawienia, ale zachowaj ID
                            webhook_url = f"{n8n_url}/webhook-test/{webhook_id}" if use_test_webhook else f"{n8n_url}/webhook/{webhook_id}"
                            
                            webhook_response = await client.get(
                                webhook_url,
                                params=params
                            )
                            try:
                                response_json = webhook_response.json()
                                response_text = response_json.get("text", str(response_json))
                            except:
                                response_text = webhook_response.text
                    
                    elif current_backend == "flowise":
                        # Flowise implementation remains unchanged
                        async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                            flowise_response = await client.post(
                                f"{flowise_url}/api/v1/prediction/{webhook_id}",
                                json={"question": clean_prompt}
                            )
                            response_text = flowise_response.json().get("text", "No response")
                    
                    return StreamingResponse(
                        iter([json.dumps({
                            "content": response_text,
                            "stop": True
                        }) + "\n"]),
                        media_type="application/x-ndjson"
                    )
                except Exception as e:
                    print(f"Webhook error: {str(e)}", flush=True)

        # If no keyword match, use LLM
        if llm_backend == "llama.cpp":
            # For llama.cpp, convert to chat format
            messages = [{"role": "user", "content": prompt}]
            llm_request = {
                "model": "llama2",
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }

            print(f"Sending to llama.cpp: {llm_request}", flush=True)

            async def generate():
                async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "text/event-stream"
                    }
                    try:
                        async with client.stream("POST", f"{llama_url}/v1/chat/completions", 
                                               json=llm_request, headers=headers) as response:
                            buffer = ""
                            async for line in response.aiter_lines():
                                if line.startswith("data: "):
                                    line = line[6:]  # Remove "data: " prefix
                                    if line.strip() == "[DONE]":
                                        continue
                                    try:
                                        completion_chunk = json.loads(line)
                                        if "choices" in completion_chunk:
                                            content = completion_chunk["choices"][0].get("delta", {}).get("content", "")
                                            if content:
                                                yield json.dumps({
                                                    "content": content,
                                                    "stop": completion_chunk["choices"][0].get("finish_reason") is not None
                                                }) + "\n"
                                    except json.JSONDecodeError as e:
                                        print(f"JSON decode error: {line}", flush=True)
                                        continue
                    except Exception as e:
                        print(f"Stream error: {str(e)}", flush=True)
                        yield json.dumps({
                            "content": f"Sorry, there was an error communicating with the LLM service: {str(e)}",
                            "stop": True
                        }) + "\n"

            return StreamingResponse(
                generate(),
                media_type="application/x-ndjson"
            )
        
        elif llm_backend == "ollama":
            # For Ollama, use the old format
            ollama_request = {
                "model": ollama_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature
                }
            }
            
            print(f"Sending to Ollama: {ollama_request}", flush=True)
            
            async def generate_ollama_completion():
                async with httpx.AsyncClient(timeout=TIMEOUT_CONFIG) as client:
                    try:
                        async with client.stream("POST", f"{ollama_url}/api/generate", json=ollama_request) as response:
                            buffer = ""
                            async for line in response.aiter_lines():
                                if line.strip():
                                    try:
                                        resp_json = json.loads(line)
                                        if "response" in resp_json:
                                            yield json.dumps({
                                                "content": resp_json["response"],
                                                "stop": resp_json.get("done", False)
                                            }) + "\n"
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        print(f"Ollama error: {str(e)}", flush=True)
                        yield json.dumps({
                            "content": f"Sorry, there was an error communicating with Ollama: {str(e)}",
                            "stop": True
                        }) + "\n"
            
            return StreamingResponse(
                generate_ollama_completion(),
                media_type="application/x-ndjson"
            )

    except Exception as e:
        print(f"MAIN ERROR: {str(e)}", flush=True)
        logger.error("Error details:", exc_info=True)
        return StreamingResponse(
            iter([json.dumps({"error": str(e), "content": f"Error: {str(e)}", "stop": True}) + "\n"]),
            media_type="application/x-ndjson"
        )

@app.get("/api/health")
async def health_check():
    try:
        # Get configurations
        llm_backend = get_config_value("llm_backend", "llama.cpp")
        webhook_backend = get_config_value("webhook_backend", "n8n")
        llama_url = get_config_value("llama_url", "http://192.168.1.166:8988")
        ollama_url = get_config_value("ollama_url", "http://192.168.1.166:11434")
        n8n_url = get_config_value("n8n_url", "http://192.168.1.166:5678")
        flowise_url = get_config_value("flowise_url", "http://192.168.1.166:3000")
        
        # Check LLM connections
        llama_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                llama_response = await client.get(f"{llama_url}/v1/models")
                llama_status = "ok" if llama_response.status_code == 200 else "error"
        except:
            llama_status = "error"
            
        ollama_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                ollama_response = await client.get(f"{ollama_url}/api/version")
                ollama_status = "ok" if ollama_response.status_code == 200 else "error"
        except:
            ollama_status = "error"
            
        # Check webhook services
        n8n_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                n8n_response = await client.get(f"{n8n_url}/healthz")
                n8n_status = "ok" if n8n_response.status_code == 200 else "error"
        except:
            n8n_status = "error"
            
        flowise_status = "unknown"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                flowise_response = await client.get(f"{flowise_url}/api/v1/health")
                flowise_status = "ok" if flowise_response.status_code == 200 else "error"
        except:
            flowise_status = "error"
            
        # Check database
        db_status = "unknown"
        keyword_count = 0
        try:
            conn = sqlite3.connect(DATABASE_URL)
            c = conn.cursor()
            c.execute('SELECT COUNT(*) FROM keywords')
            keyword_count = c.fetchone()[0]
            conn.close()
            db_status = "ok"
        except:
            db_status = "error"
            
        return {
            "status": "ok",
            "services": {
                "llama": llama_status,
                "ollama": ollama_status,
                "n8n": n8n_status,
                "flowise": flowise_status,
                "database": db_status
            },
            "config": {
                "llm_backend": llm_backend,
                "webhook_backend": webhook_backend
            },
            "keywords": {
                "count": keyword_count
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
