"""Offline checks for DeepSeek single .env entry."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain.schema import HumanMessage
from dilu.api_config import configure_api, create_chat_model, load_api_key_from_env


class APIConfigTests(unittest.TestCase):
    def setUp(self):
        clean = {key: value for key, value in os.environ.items()
                 if not key.startswith(('OPENAI_', 'AZURE_', 'DEEPSEEK_', 'DILU_'))}
        self.environment = patch.dict(os.environ, clean, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_missing_env_file_raises_filenotfound(self):
        with patch('dilu.api_config.ENV_PATH', Path('non_existent_env_file')):
            with self.assertRaises(FileNotFoundError):
                load_api_key_from_env()

    def test_missing_key_in_env_file_raises_valueerror(self):
        with patch.object(Path, 'is_file', return_value=True):
            with patch.object(Path, 'read_text', return_value='OTHER_KEY=123\n'):
                with self.assertRaisesRegex(ValueError, '未配置 DEEPSEEK_API_KEY'):
                    load_api_key_from_env()

    def test_key_loaded_from_env_file_correctly(self):
        with patch('dilu.api_config.load_api_key_from_env', return_value='test-deepseek-key'):
            configure_api({})
            self.assertEqual(os.environ['OPENAI_API_KEY'], 'test-deepseek-key')
            self.assertEqual(os.environ['OPENAI_CHAT_MODEL'], 'deepseek-chat')
            self.assertEqual(os.environ['OPENAI_API_BASE'], 'https://api.deepseek.com')

    def test_driver_stream_uses_deepseek_and_non_thinking(self):
        with patch('dilu.api_config.load_api_key_from_env', return_value='test-deepseek-key'):
            configure_api({})
        chunks = iter([{'choices': [{'delta': {'role': 'assistant', 'content': '#### 1'},
                                    'finish_reason': None, 'index': 0}]}])
        with patch('openai.ChatCompletion.create', return_value=chunks) as request:
            model = create_chat_model(streaming=True)
            self.assertEqual(''.join(chunk.content for chunk in model.stream([HumanMessage(content='test')])), '#### 1')
        fields = request.call_args.kwargs
        self.assertEqual(fields['api_base'], 'https://api.deepseek.com')
        self.assertEqual(fields['api_key'], 'test-deepseek-key')
        self.assertEqual(fields['model'], 'deepseek-chat')
        self.assertEqual(fields['thinking'], {'type': 'disabled'})
        self.assertTrue(fields['stream'])

    def test_reflection_uses_configured_model(self):
        with patch('dilu.api_config.load_api_key_from_env', return_value='test-deepseek-key'):
            configure_api({'REFLECTION_MODEL': 'deepseek-reasoner'})
        response = {'choices': [{'message': {'role': 'assistant', 'content': 'ok'},
                                 'finish_reason': 'stop', 'index': 0}],
                    'model': 'deepseek-reasoner', 'usage': {}}
        with patch('openai.ChatCompletion.create', return_value=response) as request:
            result = create_chat_model(reflection=True)([HumanMessage(content='test')])
        self.assertEqual(result.content, 'ok')
        self.assertEqual(request.call_args.kwargs['model'], 'deepseek-reasoner')
        self.assertFalse(request.call_args.kwargs['stream'])


if __name__ == '__main__':
    unittest.main()
