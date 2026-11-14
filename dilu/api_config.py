"""API configuration for DeepSeek (single entry via .env file)."""
import os
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parents[1] / '.env'


def load_api_key_from_env():
    """从 repos/DiLu/.env 文件读取 DEEPSEEK_API_KEY，作为唯一的 API 凭据入口。"""
    if not ENV_PATH.is_file():
        raise FileNotFoundError(
            f"未找到密钥配置文件：{ENV_PATH}，请在该文件中配置 DEEPSEEK_API_KEY。"
        )
    for line in ENV_PATH.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, val = line.split('=', 1)
        if key.strip() == 'DEEPSEEK_API_KEY':
            api_key = val.strip().strip('\'"')
            if api_key:
                return api_key
    raise ValueError(f"{ENV_PATH} 中未配置 DEEPSEEK_API_KEY。")


def configure_api(config):
    key = load_api_key_from_env()

    os.environ['OPENAI_API_TYPE'] = 'openai'
    os.environ['OPENAI_API_KEY'] = key
    os.environ['OPENAI_API_BASE'] = config.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
    os.environ['OPENAI_CHAT_MODEL'] = config.get('OPENAI_CHAT_MODEL', 'deepseek-chat')
    os.environ['DILU_CHAT_PROVIDER'] = 'deepseek'
    os.environ['DILU_REFLECTION_MODEL'] = config.get('REFLECTION_MODEL') or os.environ['OPENAI_CHAT_MODEL']


def create_chat_model(temperature=0, max_tokens=2000, streaming=False, reflection=False):
    from langchain.chat_models import ChatOpenAI

    options = dict(
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=60,
        max_retries=2,
        streaming=streaming,
        model_kwargs={'thinking': {'type': 'disabled'}}
    )
    return ChatOpenAI(
        model_name=os.environ['DILU_REFLECTION_MODEL' if reflection else 'OPENAI_CHAT_MODEL'],
        openai_api_base=os.environ['OPENAI_API_BASE'],
        openai_api_key=os.environ['OPENAI_API_KEY'],
        **options
    )
