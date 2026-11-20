"""Wire scope for the pinned Turbo reference profile, without tensor imports."""
from __future__ import annotations

import re
from typing import Any

from .protocol import Request, RequestError

_FIELDS = {'model', 'state', 'questions', 'images'}
_QUESTION_FIELDS = {'type', 'instructions', 'criteria'}
_IMAGE_HEADER = re.compile(r'^data:image/(?:png|jpeg|jpg)(?:;[^,]*)?;base64,', re.IGNORECASE)


def _reject_embedded_media(value: Any) -> None:
    """Refuse structured media inputs, while preserving arbitrary JSON prose.

    Plain property names such as {"image": "an image description"} are JSON
    descriptions. Recognizable media blocks, media URLs and model media tokens
    are unsupported placements and must not silently become another text task.
    """
    if isinstance(value, dict):
        kind = value.get('type')
        if isinstance(kind, str) and kind.lower() in {'image', 'image_url', 'video', 'video_url', 'input_image', 'input_video'}:
            raise RequestError('embedded media is unsupported; use one top-level images item')
        if set(value) & {'image_url', 'video_url'}:
            raise RequestError('embedded media URLs are unsupported')
        for nested in value.values():
            _reject_embedded_media(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_embedded_media(nested)
    elif isinstance(value, str) and (
        value.lower().startswith(('data:image/', 'data:video/'))
        or any(token in value for token in ('<|image_pad|>', '<|video_pad|>', '<|vision_start|>'))
    ):
        raise RequestError('embedded media is unsupported; use one top-level images item')


def parse_request(value: dict[str, Any]) -> Request:
    unknown = set(value) - _FIELDS
    if unknown:
        raise RequestError(f'Unsupported turbo request fields: {sorted(unknown)}')
    questions = value.get('questions')
    if not isinstance(questions, dict) or not questions:
        raise RequestError('questions must be a nonempty mapping of question IDs to questions')
    if any(not isinstance(name, str) or not name for name in questions):
        raise RequestError('question IDs must be nonempty strings')
    for question in questions.values():
        if not isinstance(question, dict) or question.get('type') not in ('choice', 'noul', 'score'):
            raise RequestError('turbo supports choice, noul and score questions only')
        if set(question) - _QUESTION_FIELDS:
            raise RequestError('unsupported turbo question fields')
        _reject_embedded_media(question)
    _reject_embedded_media(value.get('state'))
    images = value.get('images')
    if images is not None:
        if not isinstance(images, list) or len(images) > 1:
            raise RequestError('turbo accepts zero or one top-level PNG/JPEG data URL image')
        if any(not isinstance(image, str) or not _IMAGE_HEADER.match(image) for image in images):
            raise RequestError('turbo images must be inline PNG/JPEG base64 data URLs')
    return Request(value.get('model'), value.get('state'), tuple(questions.items()), images)


def finish_response(expected_names, response: dict[str, Any]) -> dict[str, Any]:
    """Validate framing only; official calibration, typed errors and usage pass."""
    if not isinstance(response, dict) or set(response) != {'model', 'answers', 'usage'}:
        raise RuntimeError('the pinned turbo runtime returned an unexpected response envelope')
    answers = response['answers']
    if not isinstance(answers, dict) or tuple(answers) != tuple(expected_names):
        raise RuntimeError('answers do not match request question order')
    usage = response['usage']
    if (not isinstance(usage, dict) or set(usage) != {'input_tokens', 'output_tokens'}
            or type(usage['input_tokens']) is not int or usage['input_tokens'] < 0
            or type(usage['output_tokens']) is not int or usage['output_tokens'] != 0):
        raise RuntimeError('the pinned turbo runtime returned invalid usage accounting')
    if not isinstance(response['model'], str) or any(not isinstance(answer, dict) for answer in answers.values()):
        raise RuntimeError('the pinned turbo runtime returned invalid answers')
    return response
