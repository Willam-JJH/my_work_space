"""
Anthropic -> OpenAI API format proxy for DeepSeek V4 Pro
========================================================
Flask app only — daemon/restart/PID handled by proxy_daemon.py.
"""
import json, os, time, uuid
from flask import Flask, request, Response, stream_with_context
import requests

app = Flask(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-pro"

_request_count = 0
_start_time = time.time()

def anthropic_to_openai(anthropic_body):
    messages = []
    system_prompt = None
    if "system" in anthropic_body:
        system_prompt = anthropic_body["system"]
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in system_prompt
            )
    for msg in anthropic_body.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = []
            image_parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image":
                        source = block.get("source", {})
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                            }
                        })
                elif isinstance(block, str):
                    text_parts.append(block)
            if image_parts:
                content = [{"type": "text", "text": "\n".join(text_parts)}] + image_parts
            else:
                content = "\n".join(text_parts) if text_parts else ""
        msg_entry = {"role": role, "content": content}
        messages.append(msg_entry)
    openai_body = {
        "model": DEEPSEEK_MODEL, "messages": messages,
        "stream": anthropic_body.get("stream", False),
    }
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    if "max_tokens" in anthropic_body:
        openai_body["max_tokens"] = anthropic_body["max_tokens"]
    if "temperature" in anthropic_body:
        openai_body["temperature"] = anthropic_body["temperature"]
    if "top_p" in anthropic_body:
        openai_body["top_p"] = anthropic_body["top_p"]
    if "stop_sequences" in anthropic_body:
        openai_body["stop"] = anthropic_body["stop_sequences"]
    return openai_body

def openai_to_anthropic(openai_response, model_name):
    choice = openai_response["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason", "stop")
    anthropic_stop_map = {
        "stop": "end_turn", "length": "max_tokens", "content_filter": "end_turn",
    }
    response_text = message.get("content", "") or message.get("reasoning_content", "")
    return {
        "id": f"msg_{openai_response.get('id', str(uuid.uuid4()).replace('-', ''))}",
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": response_text}],
        "model": model_name,
        "stop_reason": anthropic_stop_map.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0),
        },
    }

def openai_stream_to_anthropic_stream(line, model_name):
    line = line.strip()
    if not line or not line.startswith("data: "):
        return None
    data_str = line[6:]
    if data_str == "[DONE]":
        return 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
    try:
        chunk = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    if "choices" not in chunk or not chunk["choices"]:
        return None
    choice = chunk["choices"][0]
    delta = choice.get("delta", {})
    if "content" in delta and delta["content"]:
        event_data = {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": delta["content"]},
        }
        return f"event: content_block_delta\ndata: {json.dumps(event_data)}\n\n"
    return None

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_MAX_RETRIES = 4

def deepseek_post(url, json_body, headers, stream=False, max_retries=_MAX_RETRIES):
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=json_body, headers=headers,
                                 stream=stream, timeout=300)
            if resp.status_code == 200:
                return resp
            if resp.status_code in _RETRYABLE_STATUS:
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            elif "unavailable" in (resp.text or "").lower():
                last_error = f"model-unavailable ({resp.status_code})"
            else:
                return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            last_error = f"{type(e).__name__}: {e}"
        if attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f"[Proxy] Retry {attempt+1}/{max_retries-1} (wait {wait}s) - {last_error}")
            time.sleep(wait)
    if isinstance(last_error, str) and last_error.startswith("HTTP"):
        return requests.Response()
    raise requests.RequestException(f"Failed after {max_retries} retries: {last_error}")

@app.route("/v1/messages", methods=["POST"])
def messages():
    try:
        anthropic_body = request.get_json()
        print(f"[Proxy] Request model={anthropic_body.get('model', 'unknown')} "
              f"stream={anthropic_body.get('stream', False)}")
    except Exception as e:
        return Response(json.dumps({"error": f"Invalid JSON: {str(e)}"}),
                        status=400, mimetype="application/json")
    stream = anthropic_body.get("stream", False)
    try:
        openai_body = anthropic_to_openai(anthropic_body)
    except Exception as e:
        print(f"[Proxy] Conversion error: {e}")
        return Response(json.dumps({"error": f"Conversion error: {str(e)}"}),
                        status=400, mimetype="application/json")
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    if stream:
        openai_body["stream"] = True
        def generate():
            try:
                resp = deepseek_post(f"{DEEPSEEK_BASE}/chat/completions",
                                     json=openai_body, headers=headers, stream=True)
            except requests.RequestException as e:
                yield f"data: {json.dumps({'error': f'DeepSeek API unreachable: {str(e)}'})}\n\n"
                return
            if resp.status_code != 200:
                err = resp.text[:500] if hasattr(resp, 'text') and resp.text else f"status={resp.status_code}"
                yield f"data: {json.dumps({'error': err})}\n\n"
                return
            start = {
                "type": "message_start",
                "message": {
                    "id": f"msg_{str(uuid.uuid4()).replace('-', '')}",
                    "type": "message", "role": "assistant",
                    "content": [], "model": DEEPSEEK_MODEL,
                },
            }
            yield f"event: message_start\ndata: {json.dumps(start)}\n\n"
            yield "event: content_block_start\ndata: "
            yield json.dumps({"type": "content_block_start", "index": 0,
                              "content_block": {"type": "text", "text": ""}})
            yield "\n\n"
            yield "event: ping\ndata: {}\n\n"
            for line in resp.iter_lines(decode_unicode=True):
                if line:
                    converted = openai_stream_to_anthropic_stream(line, DEEPSEEK_MODEL)
                    if converted:
                        yield converted
            delta = {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 0},
            }
            yield f"event: message_delta\ndata: {json.dumps(delta)}\n\n"
            yield "event: message_stop\ndata: {}\n\n"
        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )
    else:
        try:
            resp = deepseek_post(f"{DEEPSEEK_BASE}/chat/completions",
                                 json=openai_body, headers=headers)
            if resp.status_code != 200:
                print(f"[Proxy] DeepSeek error: {resp.text[:300]}")
                return Response(json.dumps({"error": f"DeepSeek API error: {resp.text[:500]}"}),
                                status=resp.status_code, mimetype="application/json")
            openai_response = resp.json()
            anthropic_response = openai_to_anthropic(openai_response, DEEPSEEK_MODEL)
            return Response(json.dumps(anthropic_response), status=200, mimetype="application/json")
        except requests.RequestException as e:
            print(f"[Proxy] Request error: {e}")
            return Response(json.dumps({"error": f"Proxy error: {str(e)}"}),
                            status=502, mimetype="application/json")

@app.route("/health", methods=["GET"])
def health():
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "requests_served": _request_count,
        "target": f"{DEEPSEEK_BASE}/chat/completions",
        "model": DEEPSEEK_MODEL,
    }

@app.route("/", methods=["GET", "HEAD"])
def root():
    return {"status": "ok", "proxy": "Anthropic->OpenAI for DeepSeek V4 Pro"}

@app.before_request
def _count_request():
    global _request_count
    _request_count += 1

if __name__ == "__main__":
    print("=" * 50)
    print("  Anthropic->OpenAI Proxy for DeepSeek V4 Pro")
    print(f"  Target: {DEEPSEEK_BASE}/chat/completions")
    print(f"  Model: {DEEPSEEK_MODEL}")
    print("=" * 50)
    app.run(host="127.0.0.1", port=4000, debug=False)